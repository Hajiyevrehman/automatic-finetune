#!/usr/bin/env python
"""
Fine-tuning script for Qwen models using LoRA/QLoRA technique
with support for pre-tokenized PyTorch tensor datasets
"""

import os
import yaml
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Union, Callable
import boto3
from urllib.parse import urlparse

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer, 
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from accelerate import Accelerator
import mlflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune
    """
    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (branch name, tag name or commit id)"},
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={"help": "Trust remote code when loading the model"},
    )

@dataclass
class LoraArguments:
    """
    Arguments pertaining to LoRA configuration
    """
    r: int = field(default=16, metadata={"help": "Lora attention dimension"})
    alpha: int = field(default=32, metadata={"help": "Lora alpha"})
    dropout: float = field(default=0.05, metadata={"help": "Lora dropout"})
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "List of module names to apply LoRA to"}
    )
    bias: str = field(default="none", metadata={"help": "Bias type for LoRA (none, all or lora_only)"})

@dataclass
class QuantizationArguments:
    """
    Arguments for model quantization (QLoRA)
    """
    enabled: bool = field(default=True, metadata={"help": "Whether to use quantization"})
    bits: int = field(default=4, metadata={"help": "Number of bits for quantization"})
    group_size: int = field(default=128, metadata={"help": "Group size for quantization"})
    use_double_quant: bool = field(default=True, metadata={"help": "Whether to use double quantization"})

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation
    """
    train_file: str = field(metadata={"help": "Path to training data"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Path to validation data"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Path to test data"})
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded."},
    )
    text_column: str = field(default="text", metadata={"help": "Column containing text data"})


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def ensure_type(value, expected_type, param_name):
    """Ensure value is of expected type, convert if possible"""
    if isinstance(value, expected_type):
        return value
    
    try:
        if expected_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', 'yes', '1', 'y')
            return bool(value)
        return expected_type(value)
    except (ValueError, TypeError):
        logger.warning(f"Parameter {param_name} with value {value} couldn't be converted to {expected_type.__name__}. Using default.")
        if expected_type == bool:
            return False
        elif expected_type == int:
            return 0
        elif expected_type == float:
            return 0.0
        elif expected_type == str:
            return ""
        elif expected_type == list:
            return []
        else:
            return None

def is_s3_path(path):
    """Check if a path is an S3 path"""
    return path.startswith("s3://")

def parse_s3_path(s3_path):
    """Parse an S3 path into bucket and key"""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key

def download_from_s3(s3_path, local_path):
    """Download a file from S3 to a local path"""
    bucket, key = parse_s3_path(s3_path)
    logger.info(f"Downloading {s3_path} to {local_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, local_path)
    
    return local_path

def upload_to_s3(local_path, s3_path):
    """Upload a file or directory to S3"""
    bucket, key = parse_s3_path(s3_path)
    logger.info(f"Uploading {local_path} to {s3_path}")
    
    s3_client = boto3.client('s3')
    
    if os.path.isdir(local_path):
        # Upload a directory
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = os.path.join(key, relative_path).replace("\\", "/")
                s3_client.upload_file(local_file_path, bucket, s3_key)
    else:
        # Upload a single file
        s3_client.upload_file(local_path, bucket, key)
    
    return s3_path

def train(config_path):
    """Main training function"""
    # Load configuration
    config = load_config(config_path)
    
    # Set the seed for reproducibility
    set_seed(ensure_type(config["training"].get("seed", 42), int, "seed"))
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create a HfArgumentParser-compatible dict from the config
    model_args_dict = config["model"]
    lora_args_dict = config["lora"]
    quant_args_dict = config["quantization"]
    data_args_dict = config["dataset"]
    
    # Setup MLflow tracking
    mlflow_config = config["output"]["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    # Start MLflow run
    with mlflow.start_run() as run:
        mlflow.log_params({f"model.{k}": v for k, v in model_args_dict.items()})
        mlflow.log_params({f"lora.{k}": v for k, v in lora_args_dict.items()})
        mlflow.log_params({f"training.{k}": v for k, v in config["training"].items()})
        mlflow.log_params({f"quantization.{k}": v for k, v in quant_args_dict.items()})
        
        # Load dataset
        logger.info(f"Loading datasets from {data_args_dict['train_file']} and {data_args_dict.get('validation_file', 'None')}")
        
        # Handle S3 paths if needed
        data_files = {}
        for split, file_path in {
            "train": data_args_dict["train_file"],
            "validation": data_args_dict.get("validation_file")
        }.items():
            if file_path is None:
                continue
                
            if is_s3_path(file_path):
                # For S3 paths, download to a local temporary file
                local_path = os.path.join("tmp", os.path.basename(file_path))
                data_files[split] = download_from_s3(file_path, local_path)
            else:
                data_files[split] = file_path
        
        # Load tokenizer first
        logger.info(f"Loading tokenizer for {model_args_dict['name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args_dict["name"],
            revision=model_args_dict.get("revision", "main"),
            trust_remote_code=ensure_type(model_args_dict.get("trust_remote_code", True), bool, "trust_remote_code"),
            use_fast=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset based on file extension
        extension = data_files["train"].split(".")[-1]
        
        # Specialized handling for pre-tokenized tensor datasets
        if extension == "pt":
            logger.info("Loading pre-tokenized PyTorch tensor datasets")
            tokenized_datasets = {}
            
            for split, file_path in data_files.items():
                logger.info(f"Loading {split} dataset from {file_path}")
                dataset = torch.load(file_path)
                logger.info(f"Loaded {split} dataset successfully")
                tokenized_datasets[split] = dataset
        else:
            # Standard handling for non-tensor datasets
            logger.info(f"Loading datasets from {extension} files using Hugging Face datasets")
            datasets = load_dataset(
                extension, 
                data_files=data_files,
                num_proc=ensure_type(data_args_dict.get("preprocessing_num_workers", 4), int, "preprocessing_num_workers")
            )
            
            # Tokenize datasets
            def tokenize_function(examples):
                return tokenizer(
                    examples[data_args_dict["text_column"]],
                    padding="max_length",
                    truncation=True,
                    max_length=ensure_type(data_args_dict.get("max_seq_length", 1024), int, "max_seq_length"),
                )
            
            tokenized_datasets = datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=[col for col in datasets["train"].column_names if col != data_args_dict["text_column"]],
                desc="Tokenizing datasets",
            )
        
        # Load model with quantization if enabled
        logger.info(f"Loading model {model_args_dict['name']}")
        if ensure_type(quant_args_dict.get("enabled", True), bool, "enabled"):
            bits = ensure_type(quant_args_dict.get("bits", 4), int, "bits")
            logger.info(f"Using {bits}-bit quantization")
            
            # Correct usage of BitsAndBytesConfig from transformers
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=ensure_type(quant_args_dict.get("use_double_quant", True), bool, "use_double_quant"),
                bnb_4bit_quant_type="nf4",
            )
            
            # For Qwen, enable Flash Attention if available
            use_flash_attention = ensure_type(config["training"].get("use_flash_attention", True), bool, "use_flash_attention") 
            attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_args_dict["name"],
                    revision=model_args_dict.get("revision", "main"),
                    trust_remote_code=ensure_type(model_args_dict.get("trust_remote_code", True), bool, "trust_remote_code"),
                    quantization_config=bnb_config,
                    device_map="auto",
                    attn_implementation=attn_implementation,
                )
                logger.info("Model loaded with Flash Attention")
            except Exception as e:
                logger.warning(f"Failed to load model with Flash Attention: {e}")
                logger.info("Falling back to standard attention")
                model = AutoModelForCausalLM.from_pretrained(
                    model_args_dict["name"],
                    revision=model_args_dict.get("revision", "main"),
                    trust_remote_code=ensure_type(model_args_dict.get("trust_remote_code", True), bool, "trust_remote_code"),
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            
            model = prepare_model_for_kbit_training(model)
        else:
            # For standard precision training
            try:
                # For Qwen, enable Flash Attention if available
                use_flash_attention = ensure_type(config["training"].get("use_flash_attention", True), bool, "use_flash_attention")
                attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_args_dict["name"],
                    revision=model_args_dict.get("revision", "main"),
                    trust_remote_code=ensure_type(model_args_dict.get("trust_remote_code", True), bool, "trust_remote_code"),
                    device_map="auto",
                    attn_implementation=attn_implementation,
                )
                logger.info("Model loaded with Flash Attention")
            except Exception as e:
                logger.warning(f"Failed to load model with Flash Attention: {e}")
                logger.info("Falling back to standard attention")
                model = AutoModelForCausalLM.from_pretrained(
                    model_args_dict["name"],
                    revision=model_args_dict.get("revision", "main"),
                    trust_remote_code=ensure_type(model_args_dict.get("trust_remote_code", True), bool, "trust_remote_code"),
                    device_map="auto",
                )
        
        # Configure LoRA
        r = ensure_type(lora_args_dict.get("r", 16), int, "r")
        logger.info(f"Configuring LoRA with rank {r}")
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=ensure_type(lora_args_dict.get("alpha", 32), int, "alpha"),
            target_modules=lora_args_dict.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=ensure_type(lora_args_dict.get("dropout", 0.05), float, "dropout"),
            bias=lora_args_dict.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Print number of trainable parameters
        model.print_trainable_parameters()
        
        # Configure training arguments with proper type conversion
        training_config = config["training"]
        
        # Convert all numeric parameters to the right types
        batch_size = ensure_type(training_config.get("batch_size", 4), int, "batch_size")
        gradient_accumulation_steps = ensure_type(training_config.get("gradient_accumulation_steps", 8), int, "gradient_accumulation_steps")
        num_epochs = ensure_type(training_config.get("num_epochs", 3), float, "num_epochs")
        learning_rate = ensure_type(training_config.get("learning_rate", 2e-4), float, "learning_rate")
        weight_decay = ensure_type(training_config.get("weight_decay", 0.01), float, "weight_decay")
        warmup_ratio = ensure_type(training_config.get("warmup_ratio", 0.03), float, "warmup_ratio")
        save_steps = ensure_type(training_config.get("save_steps", 100), int, "save_steps")
        eval_steps = ensure_type(training_config.get("eval_steps", 100), int, "eval_steps")
        logging_steps = ensure_type(training_config.get("logging_steps", 10), int, "logging_steps")
        fp16 = ensure_type(training_config.get("fp16", True), bool, "fp16")
        gradient_checkpointing = ensure_type(training_config.get("gradient_checkpointing", True), bool, "gradient_checkpointing")
        
        logger.info(f"Training parameters: learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}")
                
        training_args = TrainingArguments(
            output_dir=config["output"]["output_dir"],
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=training_config.get("lr_scheduler", "cosine"),
            
            # Matching save and eval strategies
            save_strategy="steps",
            save_steps=save_steps,
            evaluation_strategy="steps",  # Must match save_strategy when load_best_model_at_end is True
            eval_steps=eval_steps,
            
            logging_steps=logging_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            report_to="mlflow",
        )
        
        # Initialize data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize Trainer - handle tensor datasets properly
        if extension == "pt":
            logger.info("Using pre-tokenized dataset for training")
            train_dataset = tokenized_datasets["train"]
            eval_dataset = tokenized_datasets.get("validation", None)
        else:
            logger.info("Using standard dataset for training")
            train_dataset = tokenized_datasets["train"]
            eval_dataset = tokenized_datasets.get("validation", None)
            
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train model
        logger.info("Starting training")
        train_result = trainer.train()
        
        # Save final model and tokenizer
        logger.info("Saving final model")
        output_dir = config["output"]["output_dir"]
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Log training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate model if validation set is available
        if eval_dataset is not None:
            logger.info("Evaluating model")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        
        # Log artifacts to MLflow
        mlflow.log_artifacts(output_dir, artifact_path="model")
        
        # Upload model to S3 if an S3 output path is provided
        if "s3_output_path" in config["output"]:
            s3_output_path = config["output"]["s3_output_path"]
            model_name = os.path.basename(output_dir)
            s3_full_path = os.path.join(s3_output_path, model_name).replace("\\", "/")
            
            logger.info(f"Uploading model to {s3_full_path}")
            upload_to_s3(output_dir, s3_full_path)
            logger.info(f"Model uploaded to S3: {s3_full_path}")
        
        logger.info(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM with LoRA/QLoRA")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training/qwen_training_config.yaml",
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    
    train(args.config)