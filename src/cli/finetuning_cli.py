#!/usr/bin/env python
# src/cli/finetune_cli.py
"""
Command-line interface for LLM finetuning process.
This script allows running the LLM finetuning process from the command line.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("finetune_cli")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Finetune LLM on custom dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/llm_finetuning.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/data_processing.yaml",
        help="Path to data configuration file",
    )
    parser.add_argument(
        "--data-path", type=str, help="Path to training data (overrides config)"
    )
    parser.add_argument(
        "--model", type=str, help="Model name to finetune (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for finetuned model (overrides config)",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for training (overrides config)",
    )
    parser.add_argument(
        "--s3-upload", action="store_true", help="Upload finetuned model to S3"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def override_config_with_args(config, args):
    """Override configuration with command-line arguments"""
    if args.model:
        config["model"]["name"] = args.model

    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs

    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate

    if args.output_dir:
        config["output"]["dir"] = args.output_dir

    if args.data_path:
        config["data"]["train_file"] = args.data_path

    return config


def finetune_model(config, data_config):
    """Finetune LLM with the specified configuration"""
    # We'll import all heavy dependencies here to ensure fast CLI startup
    try:
        import torch
        from datasets import Dataset
        from transformers import TextStreamer, TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported

        # Import S3 modules
        from src.cloud.auth import get_s3_client
        from src.cloud.storage import S3Storage
    except ImportError as e:
        logger.error(f"Error importing required packages: {e}")
        logger.error(
            "Please install the required packages: pip install unsloth trl datasets boto3"
        )
        sys.exit(1)

    # Get model params from config
    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]
    load_in_4bit = config["model"]["load_in_4bit"]
    dtype = None  # Auto detection

    # Get S3 info from config
    s3_bucket = data_config["s3"]["default_bucket"]
    s3_region = data_config["s3"]["region"]

    # Get output directory
    output_dir = config["output"].get("dir", "models/finetuned")

    # Connect to S3
    logger.info(f"Connecting to S3 bucket {s3_bucket} in region {s3_region}")
    s3_client = get_s3_client()  # The function doesn't accept region parameter
    s3_storage = S3Storage(s3_bucket)

    # Download data from S3
    train_file = config["data"]["train_file"]
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

    logger.info(f"Downloading dataset from s3://{s3_bucket}/{s3_data_path}")
    logger.info(f"Saving to local path: {local_data_path}")
    success = s3_storage.download_file(s3_data_path, local_data_path)
    
    if not success or not os.path.exists(local_data_path):
        raise FileNotFoundError(f"Failed to download file from S3: {s3_data_path} to {local_data_path}")

    # Load dataset
    import json

    with open(local_data_path, "r") as f:
        dataset_raw = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset_raw)} examples")

    # Load model
    logger.info(f"Loading {model_name} with Unsloth optimization")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Add LoRA adapters
    lora_config = config.get("lora", {})
    logger.info("Adding LoRA adapters")
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

    # Format dataset based on format_type
    format_type = config["data"].get("format_type", "chatml")

    def format_servicenow_qa(examples):
        formatted_examples = []

        if format_type == "chatml":
            for example in examples:
                messages = example.get("messages", [])

                # Extract messages by role
                system_content = ""
                user_content = ""
                assistant_content = ""

                for message in messages:
                    if message["role"] == "system":
                        system_content = message["content"]
                    elif message["role"] == "user":
                        user_content = message["content"]
                    elif message["role"] == "assistant":
                        assistant_content = message["content"]

                # Create formatted text with ChatML format
                formatted_text = f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_content}{tokenizer.eos_token}<|im_end|>"""

                formatted_examples.append({"text": formatted_text})
        else:
            logger.warning(
                f"Unknown format type: {format_type}, defaulting to raw text"
            )
            # Default to raw text
            for example in examples:
                formatted_examples.append({"text": str(example)})

        return formatted_examples

    # Format dataset and create HF dataset
    formatted_data = format_servicenow_qa(dataset_raw)
    train_dataset = Dataset.from_list(formatted_data)

    logger.info(f"Formatted dataset with {len(train_dataset)} examples")

    # Get training parameters
    train_config = config.get("training", {})

    # Create training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=train_config.get("batch_size", 2),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        warmup_steps=train_config.get("warmup_steps", 5),
        num_train_epochs=train_config.get("num_train_epochs", 3),
        learning_rate=train_config.get("learning_rate", 2e-4),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=train_config.get("logging_steps", 10),
        optim=train_config.get("optim", "adamw_8bit"),
        weight_decay=train_config.get("weight_decay", 0.01),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "linear"),
        seed=train_config.get("seed", 3407),
        output_dir=output_dir,
        evaluation_strategy=train_config.get("evaluation_strategy", "no"),
        save_strategy=train_config.get("save_strategy", "epoch"),
        save_total_limit=train_config.get("save_total_limit", 3),
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Display memory stats
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total GPU Memory: {total_gpu_memory:.3f} GB")
    logger.info(f"Currently Allocated: {allocated_gpu_memory:.3f} GB")

    # Train model
    logger.info("Starting training")
    trainer_stats = trainer.train()

    # Show final stats
    training_time_seconds = trainer_stats.metrics.get("train_runtime", 0)
    training_time_minutes = training_time_seconds / 60
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    logger.info(f"Training completed in {training_time_minutes:.2f} minutes")
    logger.info(f"Peak allocated memory = {peak_memory_gb:.3f} GB")

    # Save model locally
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Upload to S3 if requested
    if args.s3_upload:
        # Get S3 model path from config
        s3_config = config.get("s3", {})
        model_s3_path = s3_config.get(
            "model_prefix", f"models/{model_name.split('/')[-1]}-servicenow-qa"
        )

        # Get output formats
        output_formats = config["output"].get("save_formats", ["lora", "gguf_q4_k_m"])

        logger.info(f"Uploading model to S3: s3://{s3_bucket}/{model_s3_path}")

        # Save LoRA adapters if specified
        if "lora" in output_formats:
            logger.info("Saving LoRA adapters")
            model.save_pretrained_merged(output_dir, tokenizer, save_method="lora")
            
            # Upload directory contents to S3
            for root, _, files in os.walk(output_dir):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    # Make the S3 key relative to the output directory
                    relative_path = os.path.relpath(local_file_path, output_dir)
                    s3_key = f"{model_s3_path}/lora/{relative_path}"
                    s3_storage.upload_file(local_file_path, s3_key)

        # Save in GGUF format if specified
        gguf_formats = [fmt for fmt in output_formats if fmt.startswith("gguf_")]
        if gguf_formats:
            logger.info("Converting to GGUF format(s)")
            for gguf_format in gguf_formats:
                # Extract quantization method
                quant_method = gguf_format.replace("gguf_", "")
                logger.info(f"Creating GGUF with quantization: {quant_method}")
                model.save_pretrained_gguf(
                    output_dir, tokenizer, quantization_method=quant_method
                )
                gguf_file = f"{output_dir}-unsloth-{quant_method.upper()}.gguf"
                s3_key = f"{model_s3_path}/gguf/model-{quant_method}.gguf"
                s3_storage.upload_file(gguf_file, s3_key)

    # Test inference
    logger.info("Testing inference with finetuned model")
    FastLanguageModel.for_inference(model)

    test_question = "How do I reset my ServiceNow password?"

    # Format based on format_type
    if format_type == "chatml":
        test_prompt = f"""<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
{test_question}<|im_end|>
<|im_start|>assistant
"""
    else:
        test_prompt = f"Question: {test_question}\nAnswer:"

    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    logger.info("Model response:")
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=200)

    logger.info("Finetuning process completed successfully!")


def main():
    """Main function"""
    args = parse_args()

    # Load configurations
    try:
        logger.info(f"Loading training config from {args.config}")
        training_config = load_config(args.config)

        logger.info(f"Loading data config from {args.data_config}")
        data_config = load_config(args.data_config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command-line arguments
    training_config = override_config_with_args(training_config, args)

    # Run finetuning
    finetune_model(training_config, data_config)


if __name__ == "__main__":
    main()