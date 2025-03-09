import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import sys
import json
import yaml
import torch
import mlflow
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    BitsAndBytesConfig,
    TextStreamer
)
from trl import SFTConfig, SFTTrainer

def train_model_with_unsloth(config=None):
    """
    Train a model using Unsloth with the provided configuration.
    
    Args:
        config (dict): Configuration dictionary for training
    """
        # Load config from default location if not provided
    config_path = "configs/training/llm_finetuning.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize MLflow
    mlflow.set_experiment("llm_finetuning")
    
    # Configure MLflow
    mlflow_config = config.get("mlflow", {})
    mlflow_run_name = mlflow_config.get("run_name", "llm_finetune_run")
    mlflow_tracking_uri = mlflow_config.get("tracking_uri", "")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        # Log basic configuration parameters
        mlflow.log_params({
            "model_name": config["model"]["name"],
            "max_seq_length": config["model"]["max_seq_length"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["alpha"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "eval_batch_size": config["training"].get("eval_batch_size", 4),
            "epochs": config["training"]["num_train_epochs"],
            "validation_split": config["training"].get("validation_split", 0.1),
            "test_mode": config["training"].get("test_mode", False),
            "test_fraction": config["training"].get("test_fraction", 0.1)
        })
        
        # S3 Storage Module for downloading data
        class S3Storage:
            def __init__(self, bucket_name):
                import boto3
                self.s3 = boto3.client('s3')
                self.bucket = bucket_name

            def download_file(self, s3_path, local_path):
                try:
                    self.s3.download_file(self.bucket, s3_path, local_path)
                    print(f"Successfully downloaded {s3_path} to {local_path}")
                    return True
                except Exception as e:
                    print(f"Error downloading from S3: {e}")
                    return False
                    
            def upload_directory(self, local_dir, s3_prefix):
                try:
                    success_count = 0
                    error_count = 0
                    
                    for root, _, files in os.walk(local_dir):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_file_path, local_dir)
                            s3_key = f"{s3_prefix}/{relative_path}"
                            
                            if self.upload_file(local_file_path, s3_key):
                                success_count += 1
                            else:
                                error_count += 1
                    
                    print(f"Directory upload complete. {success_count} files uploaded, {error_count} errors.")
                    return success_count > 0 and error_count == 0
                except Exception as e:
                    print(f"Error uploading directory to S3: {e}")
                    return False
                    
            def upload_file(self, local_path, s3_path):
                try:
                    self.s3.upload_file(local_path, self.bucket, s3_path)
                    print(f"Successfully uploaded {local_path} to {s3_path}")
                    return True
                except Exception as e:
                    print(f"Error uploading to S3: {e}")
                    return False

        # Setup S3 storage
        s3_bucket = config["s3"]["bucket"]
        s3_storage = S3Storage(s3_bucket)

        # Download data from S3
        train_file = config["data"]["train_file"]
        
        # If the train file is already a full S3 path, use it directly
        # Otherwise, assume it's a relative path in the S3 bucket
        if train_file.startswith("s3://"):
            s3_data_path = train_file.replace(f"s3://{s3_bucket}/", "")
        else:
            s3_data_path = train_file

        # Extract filename from S3 path
        filename = s3_data_path.split('/')[-1]
        if not filename:
            filename = "dataset.json"

        # Create a temp directory for downloaded files if it doesn't exist
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)

        # Set the local path to the temp directory with filename
        local_data_path = os.path.join(temp_dir, filename)

        print(f"Downloading dataset from s3://{s3_bucket}/{s3_data_path}")
        print(f"Saving to local path: {local_data_path}")

        # Create mock data if download fails
        success = False
        try:
            success = s3_storage.download_file(s3_data_path, local_data_path)
        except Exception as e:
            print(f"Error in S3 download: {str(e)}")
            success = False

        if not success or not os.path.exists(local_data_path):
            print("Using mock data for demonstration since S3 download failed")
            # Create a simple dataset for testing
            mock_data = [
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": "There were 27 boys and 35 girls on the playground at recess. There were _____ children on the playground at recess."},
                        {"role": "assistant", "content": "To find the total number of children on the playground, we add the number of boys and girls together: 27 + 35 = 62 children."}
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": "What's the capital of France?"},
                        {"role": "assistant", "content": "The capital of France is Paris."}
                    ]
                }
            ]
            with open(local_data_path, "w") as f:
                json.dump(mock_data, f)

        # Load dataset
        with open(local_data_path, "r") as f:
            dataset_raw = json.load(f)

        print(f"Loaded dataset with {len(dataset_raw)} examples")
        mlflow.log_metric("dataset_size", len(dataset_raw))

        # Print sample of the raw data
        print("\n=== SAMPLE OF RAW DATA ===")
        if len(dataset_raw) > 0:
            print(json.dumps(dataset_raw[0], indent=2))

        # Check if data is already in the messages format
        is_messages_format = "messages" in dataset_raw[0] if dataset_raw else False
        print(f"\nData is{'already' if is_messages_format else ' not'} in messages format")

        # Check available GPU and memory
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
            print(f"GPU: {gpu_name}")
            print(f"Total memory: {gpu_memory_total:.2f} GB")
            print(f"Free memory: {gpu_memory_free:.2f} GB")
        else:
            print("No GPU detected, running on CPU will be very slow!")

        # Get model params from config
        model_name = config["model"]["name"]
        max_seq_length = config["model"]["max_seq_length"]
        load_in_4bit = config["model"]["load_in_4bit"]
        dtype = None  # Auto detection

        # Check if we should use a smaller model based on available GPU memory
        if torch.cuda.is_available() and gpu_memory_total < 12:
            print(f"Limited GPU memory ({gpu_memory_total:.2f}GB). Consider using a smaller model.")

        # Load model with explicit device map and CPU offload options
        print(f"Loading {model_name} with Unsloth optimization")
        try:
            # First try loading with automatic device mapping
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
            )
        except ValueError as e:
            if "modules are dispatched on the CPU or the disk" in str(e):
                print("GPU memory insufficient for full model. Enabling CPU offloading.")

                # Try again with CPU offloading enabled
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
                )

                # Try loading with manual device mapping that allows CPU offload
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    quantization_config=quantization_config,
                    device_map="auto"  # Let it automatically decide what to put where
                )
            else:
                # If it's another error, re-raise it
                raise

        # Chat formatting functions
        EOS_TOKEN = tokenizer.eos_token
        print(f"EOS token: {EOS_TOKEN}")

        def format_chat_to_chatml(messages):
            """Format messages into ChatML format using existing tokens"""
            formatted_text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "system":
                    formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

            # Add EOS token at the end to signal end of generation
            formatted_text += EOS_TOKEN
            return formatted_text

        def convert_messages_to_text(example):
            """Convert messages format to formatted text"""
            if "messages" in example:
                return {"text": format_chat_to_chatml(example["messages"])}
            return example

        # Process dataset
        if is_messages_format:
            # Create a text-based dataset by manually formatting the messages
            print("Converting messages to formatted text for training...")
            formatted_data = [convert_messages_to_text(example) for example in dataset_raw]
            dataset = Dataset.from_list(formatted_data)
        else:
            # If not in messages format, assume it's already in text format
            dataset = Dataset.from_list(dataset_raw)
            
        # Check if we're in test mode
        test_mode = config["training"].get("test_mode", False)
        test_fraction = config["training"].get("test_fraction", 0.1)
        
        if test_mode:
            print(f"\n=== RUNNING IN TEST MODE: Using {test_fraction * 100:.1f}% of the dataset ===")
            # Sample a subset of the dataset
            test_size = max(1, int(len(dataset) * test_fraction))
            # Use shuffle_indices to get a random subsample
            shuffle_indices = torch.randperm(len(dataset)).tolist()
            test_indices = shuffle_indices[:test_size]
            dataset = dataset.select(test_indices)
            print(f"Reduced dataset from {len(dataset_raw)} to {len(dataset)} examples for testing")

        # Split the dataset into training and validation sets
        validation_split = config["training"].get("validation_split", 0.1)
        print(f"Splitting dataset with validation ratio: {validation_split}")

        # Use datasets built-in train_test_split method
        dataset_splits = dataset.train_test_split(
            test_size=validation_split,
            seed=config["training"].get("seed", 3407)
        )

        # Create a DatasetDict with the splits
        datasets = DatasetDict({
            'train': dataset_splits['train'],
            'validation': dataset_splits['test']
        })

        print(f"Dataset split into {len(datasets['train'])} training examples and {len(datasets['validation'])} validation examples")

        # Add LoRA adapters
        lora_config = config.get("lora", {})
        print("Adding LoRA adapters")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get("r", 16),
            target_modules=lora_config.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=lora_config.get("use_rslora", False),
            loftq_config=None,
        )

        # Get output directory
        output_dir = config["output"].get("dir", "models/finetuned")

        # Get training parameters
        train_config = config.get("training", {})

        # Create SFT training arguments with MLflow integration
        sft_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=train_config.get("batch_size", 2),
            per_device_eval_batch_size=train_config.get("eval_batch_size", 4),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
            warmup_steps=train_config.get("warmup_steps", 5),
            num_train_epochs=train_config.get("num_train_epochs", 3),
            learning_rate=train_config.get("learning_rate", 2e-4),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            
            logging_strategy=train_config.get("logging_strategy", "steps"),
            logging_steps=train_config.get("logging_steps", 5),
            
            evaluation_strategy=train_config.get("evaluation_strategy", "steps"),
            eval_steps=train_config.get("eval_steps", 20),
            
            save_strategy=train_config.get("save_strategy", "steps"),
            save_steps=train_config.get("save_steps", 50),
            save_total_limit=train_config.get("save_total_limit", 3),
            
            optim=train_config.get("optim", "adamw_8bit"),
            weight_decay=train_config.get("weight_decay", 0.01),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "linear"),
            seed=train_config.get("seed", 3407),
            max_seq_length=max_seq_length,
            packing=False,
            
            # MLflow integration
            report_to="mlflow",
            run_name=mlflow_run_name,
        )
        
        # Create trainer with text-based dataset and validation set
        print("Creating SFTTrainer with train and validation datasets...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            args=sft_args,
            dataset_text_field="text",
        )

        # Display memory stats
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU Memory: {total_gpu_memory:.3f} GB")
            print(f"Currently Allocated: {allocated_gpu_memory:.3f} GB")
        else:
            print("No CUDA device detected, training may be slow")

        # Train model
        print("Starting training")
        trainer_stats = trainer.train()

        # Show final stats
        training_time_seconds = trainer_stats.metrics.get("train_runtime", 0)
        training_time_minutes = training_time_seconds / 60

        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak allocated memory = {peak_memory_gb:.3f} GB")

        print(f"Training completed in {training_time_minutes:.2f} minutes")

        # Save model locally
        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Make sure we save the final LoRA model as a separate directory
        final_lora_dir = f"{output_dir}_final"
        print(f"Saving final LoRA model to {final_lora_dir}")
        os.makedirs(final_lora_dir, exist_ok=True)
        model.save_pretrained(final_lora_dir)
        tokenizer.save_pretrained(final_lora_dir)

        # Upload to S3 if requested
        s3_upload = True  # Set to False if you don't want to upload to S3

        if s3_upload:
            # Get S3 model path from config
            s3_model_path = config["s3"].get(
                "model_prefix", f"models/{model_name.split('/')[-1]}-finetuned"
            )

            print(f"Uploading model to S3: s3://{s3_bucket}/{s3_model_path}")

            # Upload checkpoints directory as is
            print("Uploading checkpoints as folders")
            s3_storage.upload_directory(output_dir, f"{s3_model_path}/checkpoints")
            
            # Upload final LoRA model as a separate folder
            print("Uploading final LoRA model as a folder")
            s3_storage.upload_directory(final_lora_dir, f"{s3_model_path}/final")

        # Test inference with proper EOS token handling
        print("Testing inference with finetuned model")
        FastLanguageModel.for_inference(model)

        def generate_response(model, tokenizer, messages, max_new_tokens=200):
            """Generate a response with proper formatting and EOS token handling"""
            # Format the test prompt with EOS token included
            test_prompt = format_chat_to_chatml(messages)

            inputs = tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # Configure generation parameters to respect EOS token
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,  # Explicitly set the EOS token ID
            }

            # Stream the output
            text_streamer = TextStreamer(tokenizer)
            output = model.generate(
                **inputs,
                streamer=text_streamer,
                **generation_config
            )

            return output

        # Test with your example
        test_question = "How do I reset my ServiceNow password?"

        # Format the test prompt and generate response
        test_messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": test_question}
        ]

        print("Model response:")
        generate_response(model, tokenizer, test_messages)

        print("Finetuning process completed successfully!")
        
        # Show training metrics
        print("\n=== Testing MLflow Tracking and Displaying Training Progress ===")
        try:
            # Get the current run ID
            run_id = run.info.run_id
            
            # Query MLflow for the run
            run_info = mlflow.get_run(run_id)
            
            # Check if the run exists and has metrics
            if run_info and run_info.data.metrics:
                print(f"MLflow tracking successful! Run ID: {run_id}")
                print(f"Number of metrics logged: {len(run_info.data.metrics)}")
                print(f"Number of parameters logged: {len(run_info.data.params)}")
                
                # Display available metrics
                print(f"Metrics available: {list(run_info.data.metrics.keys())}")
            else:
                print(f"MLflow run exists but no metrics were logged.")
        except Exception as e:
            print(f"Error testing MLflow tracking: {e}")
            print("MLflow tracking may have failed.")
