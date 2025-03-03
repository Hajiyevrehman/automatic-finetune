#!/usr/bin/env python3
"""
Data pipeline for LLM fine-tuning.

Handles data operations with S3 storage integration for preparing datasets
including conversion, preprocessing, validation, tokenization, and splitting.
Uses pickle instead of torch.save for serialization to avoid PyTorch architecture issues.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.cloud.auth import create_bucket_if_not_exists
from src.cloud.storage import S3Storage

# Note: No torch import here to avoid architecture issues
# Instead we'll import transformers only when needed


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Data pipeline for processing, tokenizing, and preparing datasets.
    Includes S3 integration for cloud storage.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the DataPipeline.

        Args:
            config_path (str or Path): Path to the data processing configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Set up S3 storage if configured
        self.s3 = None
        if "s3" in self.config and self.config["s3"].get("default_bucket"):
            bucket_name = self.config["s3"]["default_bucket"]
            self.s3 = S3Storage(bucket_name)
            create_bucket_if_not_exists(bucket_name)
            logger.info(f"S3 integration enabled with bucket: {bucket_name}")

        # Create local directories
        self._create_local_directories()

        # Set up model info
        self.model_name = self.config.get("model", {}).get(
            "name", "Qwen/Qwen2.5-7B-Instruct"
        )
        logger.info(f"Using model: {self.model_name}")

    def _load_config(self) -> Dict:
        """
        Load configuration from file.
        Supports both YAML and JSON formats.

        Returns:
            Dict: Configuration data
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Load based on file extension
        if self.config_path.suffix.lower() in [".yaml", ".yml"]:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        elif self.config_path.suffix.lower() == ".json":
            with open(self.config_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {self.config_path}")

    def _create_local_directories(self):
        """Create local directories for data storage."""
        dirs = self.config.get("directories", {})
        for dir_name, dir_path in dirs.items():
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    def _get_path(self, dir_type: str, filename: str) -> Path:
        """
        Get a file path from configuration.

        Args:
            dir_type (str): Directory type ('raw', 'processed', 'validation')
            filename (str): Filename to append

        Returns:
            Path: File path
        """
        base_dir = self.config.get("directories", {}).get(dir_type, f"data/{dir_type}")
        return Path(base_dir) / filename

    def _download_if_needed(self, dir_type: str, filename: str) -> Optional[Path]:
        """
        Download a file from S3 if it's not available locally.

        Args:
            dir_type (str): Directory type ('raw', 'processed', 'validation')
            filename (str): Name of the file

        Returns:
            Path or None: Path to the local file if available, None otherwise
        """
        local_path = self._get_path(dir_type, filename)

        # If file exists locally, return its path
        if local_path.exists():
            return local_path

        # If S3 is not configured, we can't download
        if not self.s3:
            return None

        # Try to download from S3
        s3_key = f"data/{dir_type}/{filename}"
        if self.s3.download_file(s3_key, local_path):
            logger.info(f"Downloaded {s3_key} to {local_path}")
            return local_path

        return None

    def _upload_if_needed(self, local_path: Path, dir_type: str, filename: str) -> bool:
        """
        Upload a file to S3 if S3 integration is enabled.

        Args:
            local_path (Path): Path to the local file
            dir_type (str): Directory type ('raw', 'processed', 'validation')
            filename (str): Name for the file in S3

        Returns:
            bool: True if upload successful or not needed, False otherwise
        """
        if not self.s3:
            return True  # No S3 integration, so "upload" is successful by definition

        s3_key = f"data/{dir_type}/{filename}"
        success = self.s3.upload_file(local_path, s3_key)

        if success:
            logger.info(f"Uploaded {local_path} to {s3_key}")
        else:
            logger.error(f"Failed to upload {local_path} to {s3_key}")

        return success

    def convert_dataset(self, source_dataset: str, output_name: str) -> bool:
        """
        Convert a dataset to the format required for fine-tuning.

        Args:
            source_dataset (str): Name of the source dataset
            output_name (str): Name for the converted dataset

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Get source dataset
            source_path = self._download_if_needed("raw", f"{source_dataset}.json")
            if not source_path:
                logger.error(f"Source dataset not found: {source_dataset}")
                return False

            # Load the source dataset
            with open(source_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)

            logger.info(f"Converting {source_dataset} ({len(source_data)} samples)")

            # Convert the dataset to chat format
            converted_data = []
            for item in source_data:
                # Convert to the format expected for fine-tuning
                converted_item = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant.",
                        },
                        {"role": "user", "content": item.get("question", "")},
                        {"role": "assistant", "content": item.get("answer", "")},
                    ]
                }
                converted_data.append(converted_item)

            # Save locally
            output_path = self._get_path("processed", f"{output_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)

            # Upload to S3 if configured
            if not self._upload_if_needed(
                output_path, "processed", f"{output_name}.json"
            ):
                return False

            logger.info(f"Converted {len(converted_data)} samples to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error converting dataset: {str(e)}")
            return False

    def preprocess_data(self, dataset_name: str, output_name: str) -> bool:
        """
        Preprocess the dataset with text normalization and cleaning.

        Args:
            dataset_name (str): Name of the dataset to preprocess
            output_name (str): Name for the preprocessed dataset

        Returns:
            bool: True if preprocessing successful, False otherwise
        """
        try:
            # Get dataset
            source_path = self._download_if_needed("processed", f"{dataset_name}.json")
            if not source_path:
                logger.error(f"Dataset not found: {dataset_name}")
                return False

            # Load the dataset
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Preprocessing dataset {dataset_name} ({len(data)} samples)")

            # Preprocess the dataset
            for item in data:
                for message in item.get("messages", []):
                    # Text normalization
                    if isinstance(message.get("content"), str):
                        # Normalize whitespace
                        message["content"] = " ".join(message["content"].split())
                        # Strip extra spaces around punctuation
                        for punct in [".", ",", "!", "?", ":", ";"]:
                            message["content"] = message["content"].replace(
                                f" {punct}", punct
                            )

            # Save locally
            output_path = self._get_path("processed", f"{output_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Upload to S3 if configured
            if not self._upload_if_needed(
                output_path, "processed", f"{output_name}.json"
            ):
                return False

            logger.info(f"Preprocessed {len(data)} samples to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            return False

    def validate_data(self, dataset_name: str, output_name: str) -> bool:
        """
        Validate the dataset for quality and format.

        Args:
            dataset_name (str): Name of the dataset to validate
            output_name (str): Name for the validated dataset

        Returns:
            bool: True if validation successful, False otherwise
        """
        try:
            # Get dataset
            source_path = self._download_if_needed("processed", f"{dataset_name}.json")
            if not source_path:
                logger.error(f"Dataset not found: {dataset_name}")
                return False

            # Load the dataset
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Validating dataset {dataset_name} ({len(data)} samples)")

            valid_data = []
            validation_stats = {
                "total": len(data),
                "valid": 0,
                "invalid": 0,
                "reasons": {},
            }

            # Validate each sample
            for item in data:
                is_valid = True
                reason = None

                # Check for required message structure
                if "messages" not in item or not isinstance(item["messages"], list):
                    is_valid = False
                    reason = "invalid_message_structure"

                # Check for non-empty content
                elif any(not message.get("content") for message in item["messages"]):
                    is_valid = False
                    reason = "empty_content"

                # Check for required roles
                elif not any(
                    message.get("role") == "user" for message in item["messages"]
                ):
                    is_valid = False
                    reason = "missing_user_message"

                elif not any(
                    message.get("role") == "assistant" for message in item["messages"]
                ):
                    is_valid = False
                    reason = "missing_assistant_message"

                if is_valid:
                    valid_data.append(item)
                    validation_stats["valid"] += 1
                else:
                    validation_stats["invalid"] += 1
                    validation_stats["reasons"][reason] = (
                        validation_stats["reasons"].get(reason, 0) + 1
                    )

            # Save valid data
            output_path = self._get_path("validation", f"{output_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(valid_data, f, indent=2, ensure_ascii=False)

            # Save validation stats
            stats_path = self._get_path("validation", f"{output_name}_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(validation_stats, f, indent=2)

            # Upload to S3 if configured
            success = True
            if not self._upload_if_needed(
                output_path, "validation", f"{output_name}.json"
            ):
                success = False
            if not self._upload_if_needed(
                stats_path, "validation", f"{output_name}_stats.json"
            ):
                success = False

            logger.info(
                f"Validation completed: {validation_stats['valid']}/{validation_stats['total']} valid"
            )

            return success

        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return False

    def tokenize_data(self, dataset_name: str, output_name: str) -> bool:
        """
        Tokenize the dataset using the model's tokenizer.
        Uses pickle for serialization instead of torch.save.

        Args:
            dataset_name (str): Name of the dataset to tokenize
            output_name (str): Name for the tokenized dataset

        Returns:
            bool: True if tokenization successful, False otherwise
        """
        try:
            # Get dataset
            source_path = self._download_if_needed("validation", f"{dataset_name}.json")
            if not source_path:
                logger.error(f"Dataset not found: {dataset_name}")
                return False

            # Load the dataset
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Process and tokenize conversations
            logger.info(f"Tokenizing dataset {dataset_name} ({len(data)} samples)")
            tokenized_data = []

            # Get tokenizer settings from config
            max_length = self.config.get("model", {}).get("max_length", 2048)
            tokenizer_config = self.config.get("tokenizer", {})
            padding = tokenizer_config.get("padding", "max_length")
            truncation = tokenizer_config.get("truncation", True)

            for item in data:
                if "messages" in item:
                    try:
                        # Apply chat template to create formatted prompt
                        tokenized_result = tokenizer.apply_chat_template(
                            item["messages"],
                            return_tensors="pt",
                            add_generation_prompt=False,
                            max_length=max_length,
                            truncation=truncation,
                            padding=padding,
                        )

                        # Convert tensor to list for serialization
                        token_ids = tokenized_result.tolist()[0]

                        tokenized_item = {
                            "input_ids": token_ids,
                            # Save original messages for reference
                            "original_messages": item["messages"],
                            # Add metadata if available
                            "metadata": item.get("metadata", {}),
                        }

                        tokenized_data.append(tokenized_item)
                    except Exception as e:
                        logger.warning(f"Failed to tokenize item: {str(e)}")

            # Save tokenized data locally using pickle
            output_path = self._get_path("processed", f"{output_name}.pt")
            with open(output_path, "wb") as f:
                pickle.dump(tokenized_data, f)

            # Upload to S3 if configured
            if not self._upload_if_needed(
                output_path, "processed", f"{output_name}.pt"
            ):
                return False

            logger.info(f"Tokenized {len(tokenized_data)} samples to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error tokenizing dataset: {str(e)}")
            return False

    def split_dataset(self, dataset_name: str, train_ratio: float = None) -> bool:
        """
        Split the tokenized dataset into train/validation/test sets.
        Uses pickle for serialization instead of torch.save.

        Args:
            dataset_name (str): Name of the tokenized dataset to split
            train_ratio (float, optional): Ratio of data to use for training

        Returns:
            bool: True if splitting successful, False otherwise
        """
        try:
            # Get dataset
            source_path = self._download_if_needed("processed", f"{dataset_name}.pt")
            if not source_path:
                logger.error(f"Dataset not found: {dataset_name}")
                return False

            # Load the tokenized dataset using pickle
            with open(source_path, "rb") as f:
                tokenized_data = pickle.load(f)

            # Get split configuration
            splitting_config = self.config.get("splitting", {})
            train_ratio = train_ratio or splitting_config.get("train_ratio", 0.9)
            val_test_ratio = splitting_config.get("val_test_ratio", 0.5)
            seed = splitting_config.get("seed", 42)

            # Shuffle and split the dataset
            logger.info(
                f"Splitting dataset {dataset_name} ({len(tokenized_data)} samples)"
            )
            import random

            random.seed(seed)
            random.shuffle(tokenized_data)

            total_samples = len(tokenized_data)
            train_size = int(total_samples * train_ratio)
            val_test_size = total_samples - train_size
            val_size = int(val_test_size * val_test_ratio)
            test_size = val_test_size - val_size

            train_data = tokenized_data[:train_size]
            val_data = tokenized_data[train_size : train_size + val_size]
            test_data = tokenized_data[train_size + val_size :]

            # Save splits locally using pickle
            train_path = self._get_path("processed", f"{dataset_name}_train.pt")
            val_path = self._get_path("processed", f"{dataset_name}_val.pt")
            test_path = self._get_path("processed", f"{dataset_name}_test.pt")

            with open(train_path, "wb") as f:
                pickle.dump(train_data, f)
            with open(val_path, "wb") as f:
                pickle.dump(val_data, f)
            with open(test_path, "wb") as f:
                pickle.dump(test_data, f)

            # Upload to S3 if configured
            success = True
            if not self._upload_if_needed(
                train_path, "processed", f"{dataset_name}_train.pt"
            ):
                success = False
            if not self._upload_if_needed(
                val_path, "processed", f"{dataset_name}_val.pt"
            ):
                success = False
            if not self._upload_if_needed(
                test_path, "processed", f"{dataset_name}_test.pt"
            ):
                success = False

            logger.info(
                f"Dataset split completed: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test"
            )

            return success

        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            return False

    def run_pipeline(self, source_dataset: str, output_prefix: str = None) -> bool:
        """
        Run the complete data processing pipeline from raw dataset to tokenized splits.

        Args:
            source_dataset (str): Name of the source dataset
            output_prefix (str, optional): Prefix for output dataset names

        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        if output_prefix is None:
            output_prefix = source_dataset

        # Define stage names for the pipeline
        converted_name = f"{output_prefix}_converted"
        preprocessed_name = f"{output_prefix}_preprocessed"
        validated_name = f"{output_prefix}_validated"
        tokenized_name = f"{output_prefix}_tokenized"

        # Step 1: Convert dataset format
        logger.info("Step 1: Converting dataset format")
        if not self.convert_dataset(source_dataset, converted_name):
            logger.error("Pipeline failed at conversion stage")
            return False

        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing dataset")
        if not self.preprocess_data(converted_name, preprocessed_name):
            logger.error("Pipeline failed at preprocessing stage")
            return False

        # Step 3: Validate data
        logger.info("Step 3: Validating dataset")
        if not self.validate_data(preprocessed_name, validated_name):
            logger.error("Pipeline failed at validation stage")
            return False

        # Step 4: Tokenize data
        logger.info("Step 4: Tokenizing dataset")
        if not self.tokenize_data(validated_name, tokenized_name):
            logger.error("Pipeline failed at tokenization stage")
            return False

        # Step 5: Split dataset
        logger.info("Step 5: Splitting dataset")
        if not self.split_dataset(tokenized_name):
            logger.error("Pipeline failed at splitting stage")
            return False

        logger.info("Data pipeline completed successfully!")
        return True
