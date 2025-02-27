"""
Trainer module for fine-tuning LLMs with LoRA/QLoRA.
"""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import datasets
import torch
import transformers
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

import mlflow
from src.cloud.storage import S3Storage
from src.training.data_loader import load_and_prepare_data, parse_s3_path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainer:
    """
    Handles the fine-tuning of LLMs with LoRA/QLoRA.
    """

    config_path: str

    def __post_init__(self):
        """Load configuration after initialization."""
        self.config = self._load_config()
        # Setup MLflow
        self._setup_mlflow()
        # Initialize S3 handler if needed
        self._setup_s3_storage()

    def _load_config(self) -> Dict:
        """Load the training configuration from YAML file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded training configuration from {self.config_path}")
        return config

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow_config = self.config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri")
        registry_uri = mlflow_config.get("registry_uri")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

        experiment_name = mlflow_config.get("experiment_name", "llm-finetuning")
        try:
            mlflow.create_experiment(name=experiment_name)
        except Exception:
            logger.info(f"Experiment {experiment_name} already exists")

        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow setup complete with experiment: {experiment_name}")

    def _setup_s3_storage(self):
        """Setup S3 storage handler if needed."""
        # Check if we need S3 storage (if any paths start with s3://)
        data_config = self.config.get("data", {})
        output_config = self.config.get("output", {})

        need_s3 = False
        s3_bucket = None

        # Check data files
        train_file = data_config.get("train_file", "")
        val_file = data_config.get("validation_file", "")

        if train_file.startswith("s3://"):
            need_s3 = True
            bucket, _ = parse_s3_path(train_file)
            s3_bucket = bucket

        if val_file and val_file.startswith("s3://"):
            need_s3 = True
            bucket, _ = parse_s3_path(val_file)
            s3_bucket = bucket

        # Check output path
        s3_output_path = output_config.get("s3_output_path", "")
        if s3_output_path and s3_output_path.startswith("s3://"):
            need_s3 = True
            bucket, _ = parse_s3_path(s3_output_path)
            s3_bucket = bucket

        if need_s3 and s3_bucket:
            self.s3_storage = S3Storage(s3_bucket)
            logger.info(f"S3 storage handler initialized with bucket: {s3_bucket}")
        else:
            self.s3_storage = None

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer for the base model."""
        model_config = self.config.get("model", {})
        base_model_name = model_config.get("base_model_name")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=model_config.get("trust_remote_code", True),
            use_fast=True,
        )

        # Set pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loaded tokenizer for {base_model_name}")
        return tokenizer

    def _prepare_model(self) -> torch.nn.Module:
        """Prepare the model with LoRA configuration."""
        model_config = self.config.get("model", {})
        lora_config = self.config.get("lora", {})
        qlora_config = self.config.get("qlora", {})

        base_model_name = model_config.get("base_model_name")

        # Determine quantization settings
        load_in_8bit = model_config.get("load_in_8bit", False)
        load_in_4bit = model_config.get("load_in_4bit", False)
        use_qlora = qlora_config.get("use_qlora", False)

        quantization_config = None
        if use_qlora and load_in_4bit:
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, qlora_config.get("compute_dtype", "float16")
                ),
                bnb_4bit_quant_type=qlora_config.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=qlora_config.get("double_quant", True),
            )
        elif load_in_8bit:
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        # Load the model with appropriate settings
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            trust_remote_code=model_config.get("trust_remote_code", True),
            torch_dtype=torch.bfloat16
            if self.config.get("settings", {}).get("bf16", True)
            else torch.float16,
            device_map="auto",
            use_flash_attention_2=model_config.get("use_flash_attention", True),
        )

        # Prepare model for k-bit training if needed
        if load_in_8bit or load_in_4bit:
            logger.info("Preparing model for quantized training")
            model = prepare_model_for_kbit_training(model)

        # Get target modules
        target_modules = lora_config.get("target_modules", [])
        if not target_modules:
            # Try to infer target modules based on model architecture
            model_type = model.config.model_type
            if model_type in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_type
                ]
                logger.info(
                    f"Using default target modules for {model_type}: {target_modules}"
                )
            else:
                logger.warning(
                    f"Model type {model_type} not found in mapping, using default target modules"
                )
                target_modules = ["q_proj", "v_proj"]

        # Setup LoRA configuration
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            modules_to_save=lora_config.get("modules_to_save", []),
        )

        # Convert model to LoRA
        logger.info("Applying LoRA adapter to model")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model

    def _get_training_args(self) -> TrainingArguments:
        """Setup training arguments for the Trainer."""
        training_config = self.config.get("training", {})
        output_config = self.config.get("output", {})
        settings_config = self.config.get("settings", {})

        # Create output directory if it doesn't exist
        output_dir = output_config.get("output_dir", "models/lora-output")
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_config.get(
                "per_device_train_batch_size", 4
            ),
            per_device_eval_batch_size=training_config.get(
                "per_device_eval_batch_size", 4
            ),
            gradient_accumulation_steps=training_config.get(
                "gradient_accumulation_steps", 8
            ),
            learning_rate=training_config.get("learning_rate", 5e-5),
            num_train_epochs=training_config.get("num_train_epochs", 3),
            max_steps=training_config.get("max_steps", -1),
            logging_steps=training_config.get("logging_steps", 10),
            save_steps=training_config.get("save_steps", 500),
            eval_steps=training_config.get("eval_steps", 100),
            evaluation_strategy="steps",
            save_strategy=output_config.get("save_strategy", "steps"),
            warmup_steps=training_config.get("warmup_steps", 100),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            weight_decay=training_config.get("weight_decay", 0.01),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            seed=settings_config.get("seed", 42),
            fp16=settings_config.get("fp16", False),
            bf16=settings_config.get("bf16", True),
            tf32=settings_config.get("tf32", True),
            report_to=settings_config.get("report_to", "mlflow"),
            push_to_hub=output_config.get("push_to_hub", False),
            hub_model_id=output_config.get("hub_model_id"),
        )

        logger.info("Training arguments configured")
        return training_args

    def _load_datasets(self, tokenizer: PreTrainedTokenizer):
        """Load datasets for training."""
        data_config = self.config.get("data", {})
        training_config = self.config.get("training", {})

        train_file = data_config.get("train_file")
        validation_file = data_config.get("validation_file")

        # Load datasets from files (local or S3)
        train_dataset, eval_dataset = load_and_prepare_data(
            train_file=train_file,
            val_file=validation_file,
            batch_size=training_config.get("per_device_train_batch_size", 4),
            eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
            use_cache=True,
            num_workers=data_config.get("preprocessing_num_workers", 4),
        )

        logger.info(
            f"Loaded datasets: {len(train_dataset)} training examples, "
            f"{len(eval_dataset) if eval_dataset else 0} validation examples"
        )

        return train_dataset, eval_dataset

    def _save_to_s3(self, local_output_dir: str):
        """Upload trained model to S3."""
        output_config = self.config.get("output", {})
        s3_output_path = output_config.get("s3_output_path")

        if s3_output_path and self.s3_storage:
            if not s3_output_path.startswith("s3://"):
                logger.error(f"Invalid S3 output path: {s3_output_path}")
                return

            # Parse bucket and prefix
            bucket, prefix = parse_s3_path(s3_output_path)

            # Make sure bucket matches the initialized S3 storage
            if bucket != self.s3_storage.bucket_name:
                logger.warning(
                    f"S3 bucket mismatch: {bucket} vs {self.s3_storage.bucket_name}"
                )
                self.s3_storage = S3Storage(bucket)

            # Create models directory if not exists
            models_dir = os.path.join(prefix, "models")
            model_name = os.path.basename(local_output_dir)
            upload_prefix = f"{models_dir}/{model_name}"

            logger.info(f"Uploading model to S3 at s3://{bucket}/{upload_prefix}")
            result = self.s3_storage.upload_directory(local_output_dir, upload_prefix)

            if result:
                logger.info(
                    f"Model uploaded to S3 successfully at s3://{bucket}/{upload_prefix}"
                )
            else:
                logger.error(f"Failed to upload model to S3")

    def train(self):
        """Run the training process."""
        # Setup components
        tokenizer = self._load_tokenizer()
        model = self._prepare_model()
        training_args = self._get_training_args()
        train_dataset, eval_dataset = self._load_datasets(tokenizer)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
        )

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Start MLflow run
        with mlflow.start_run(run_name=self.config.get("mlflow", {}).get("run_name")):
            # Log parameters
            mlflow.log_params(
                {
                    "model_name": self.config.get("model", {}).get("base_model_name"),
                    "lora_r": self.config.get("lora", {}).get("r"),
                    "lora_alpha": self.config.get("lora", {}).get("alpha"),
                    "learning_rate": self.config.get("training", {}).get(
                        "learning_rate"
                    ),
                    "num_train_epochs": self.config.get("training", {}).get(
                        "num_train_epochs"
                    ),
                    "train_batch_size": self.config.get("training", {}).get(
                        "per_device_train_batch_size"
                    ),
                    "gradient_accumulation_steps": self.config.get("training", {}).get(
                        "gradient_accumulation_steps"
                    ),
                }
            )

            # Train the model
            logger.info("Starting training...")
            train_result = trainer.train()

            # Log metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)

            # Evaluate
            if eval_dataset:
                logger.info("Running evaluation...")
                eval_results = trainer.evaluate()
                trainer.log_metrics("eval", eval_results)

            # Save model, tokenizer, and training state
            logger.info("Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)

            # Upload to S3 if configured
            self._save_to_s3(training_args.output_dir)

            logger.info("Training completed successfully!")
            return train_result
