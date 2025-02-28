#!/usr/bin/env python

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    config = load_config(config_path)
    
    if "aws" in config and "access_key_id" in config["aws"] and "secret_access_key" in config["aws"]:
        os.environ["AWS_ACCESS_KEY_ID"] = config["aws"]["access_key_id"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = config["aws"]["secret_access_key"]
        os.environ["AWS_REGION"] = config["aws"].get("region", "us-east-1")
    
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_sharegpt
    
    max_seq_length = config["training"].get("max_seq_length", 2048)
    dtype = None
    load_in_4bit = False
    
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    logger.info("Applying LoRA...")
    lora_config = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get("r", 16),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_alpha=lora_config.get("alpha", 16),
        lora_dropout=lora_config.get("dropout", 0),
        bias=lora_config.get("bias", "none"),
        use_gradient_checkpointing=config["training"].get("use_gradient_checkpointing", "unsloth"),
        random_state=config["training"].get("seed", 3407),
        use_rslora=lora_config.get("use_rslora", False),
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config["model"].get("chat_template", "qwen-2.5"),
    )
    
    dataset_config = config["dataset"]
    if dataset_config.get("from_s3", False):
        import boto3
        from urllib.parse import urlparse
        
        def parse_s3_path(s3_path):
            parsed = urlparse(s3_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            return bucket, key
        
        def download_from_s3(s3_path, local_path):
            bucket, key = parse_s3_path(s3_path)
            logger.info(f"Downloading {s3_path} to {local_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client = boto3.client('s3')
            s3_client.download_file(bucket, key, local_path)
            return local_path
        
        os.makedirs("tmp", exist_ok=True)
        
        train_local = os.path.join("tmp", os.path.basename(dataset_config['train_file']))
        download_from_s3(dataset_config['train_file'], train_local)
        train_dataset = torch.load(train_local)
        
        val_dataset = None
        if dataset_config.get('validation_file'):
            val_local = os.path.join("tmp", os.path.basename(dataset_config['validation_file']))
            download_from_s3(dataset_config['validation_file'], val_local)
            val_dataset = torch.load(val_local)
        
        dataset = {"train": train_dataset}
        if val_dataset:
            dataset["validation"] = val_dataset
    else:
        dataset = load_dataset(dataset_config['name'], split="train")
        
        if dataset_config.get("standardize_format", True):
            dataset = standardize_sharegpt(dataset)
        
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return {"text": texts}
        
        dataset = dataset.map(formatting_prompts_func, batched=True)
    
    training_config = config["training"]
    batch_size = training_config.get("batch_size", 1)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
    
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    
    training_args = TrainingArguments(
        output_dir=config["output"]["output_dir"],
        num_train_epochs=training_config.get("num_epochs", 1),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 100),
        evaluation_strategy="steps",
        eval_steps=training_config.get("eval_steps", 100),
        logging_steps=training_config.get("logging_steps", 10),
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=not is_bf16_supported,
        bf16=is_bf16_supported,
        optim=training_config.get("optim", "paged_adamw_8bit"),
        seed=training_config.get("seed", 3407),
        report_to=training_config.get("report_to", "none"),
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field="text" if "text" in dataset["train"].column_names else None,
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=training_config.get("dataset_num_proc", 4),
        packing=training_config.get("packing", False),
        args=training_args,
    )
    
    if training_config.get("train_on_responses_only", True):
        instruction_part = "<|im_start|>user\n"
        response_part = "<|im_start|>assistant\n"
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
    
    logger.info("Starting training...")
    trainer_stats = trainer.train()
    
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    logger.info(f"Peak reserved memory = {used_memory} GB.")
    
    logger.info(f"Saving model to {config['output']['output_dir']}...")
    trainer.save_model(config["output"]["output_dir"])
    tokenizer.save_pretrained(config["output"]["output_dir"])
    
    if config["output"].get("s3_output_path"):
        logger.info(f"Uploading model to S3")
        try:
            import boto3
            from urllib.parse import urlparse
            
            def parse_s3_path(s3_path):
                parsed = urlparse(s3_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                return bucket, key
            
            def upload_to_s3(local_path, s3_path):
                bucket, key = parse_s3_path(s3_path)
                logger.info(f"Uploading {local_path} to S3")
                s3_client = boto3.client('s3')
                
                if os.path.isdir(local_path):
                    for root, dirs, files in os.walk(local_path):
                        for file in files:
                            local_file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_file_path, local_path)
                            s3_key = os.path.join(key, relative_path).replace("\\", "/")
                            s3_client.upload_file(local_file_path, bucket, s3_key)
                else:
                    s3_client.upload_file(local_path, bucket, key)
            
            s3_output_path = config["output"]["s3_output_path"]
            model_name = os.path.basename(config["output"]["output_dir"])
            s3_full_path = os.path.join(s3_output_path, model_name).replace("\\", "/")
            upload_to_s3(config["output"]["output_dir"], s3_full_path)
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
    
    FastLanguageModel.for_inference(model)
    
    if training_config.get("test_after_training", True):
        messages = [
            {"role": "user", "content": "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=64,
            use_cache=True,
            temperature=0.7
        )
    
    logger.info("Training and evaluation completed.")
    return trainer_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/training_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    main(args.config)