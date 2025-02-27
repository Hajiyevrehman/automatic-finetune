"""
S3-integrated data pipeline for the LLM fine-tuning project.
Handles data operations with S3 storage instead of local filesystem.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from src.cloud.auth import create_bucket_if_not_exists
from src.cloud.storage import S3Storage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class S3DataPipeline:
    """
    Data pipeline with S3 integration for processing, tokenizing, and preparing datasets.
    """

    def __init__(self, config_path: Union[str, Path], bucket_name: str):
        """
        Initialize the S3DataPipeline.

        Args:
            config_path (str or Path): Path to the data processing configuration file
            bucket_name (str): S3 bucket name for data storage
        """
        self.config_path = Path(config_path)
        self.bucket_name = bucket_name
        self.s3 = S3Storage(bucket_name)

        # Ensure bucket exists
        create_bucket_if_not_exists(bucket_name)

        # Load configuration
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            # Check if config exists in S3
            s3_config_key = "configs/data/data_processing.json"
            self.config = self.s3.download_json(s3_config_key)

            if not self.config:
                raise FileNotFoundError(
                    f"Configuration file not found at {config_path} or in S3 at {s3_config_key}"
                )

        # Create local directories for temporary storage
        self._create_local_directories()

        # Set up paths
        self.model_name = self.config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")

    def _create_local_directories(self):
        """Create local directories for temporary storage."""
        for dir_key in ["raw_dir", "processed_dir", "validation_dir"]:
            if dir_key in self.config:
                os.makedirs(self.config[dir_key], exist_ok=True)

    def _get_local_path(self, directory_key: str, filename: str) -> Path:
        """
        Get a local file path from configuration.

        Args:
            directory_key (str): Key in config for the directory
            filename (str): Filename to append

        Returns:
            Path: Local file path
        """
        base_dir = self.config.get(directory_key, f"data/{directory_key.split('_')[0]}")
        return Path(base_dir) / filename

    def _get_s3_path(self, directory_key: str, filename: str) -> str:
        """
        Get an S3 key (path) from configuration.

        Args:
            directory_key (str): Key in config for the directory
            filename (str): Filename to append

        Returns:
            str: S3 key path
        """
        # Remove '_dir' suffix if present and use as prefix
        prefix = directory_key.replace("_dir", "")
        return f"data/{prefix}/{filename}"

    def download_from_source(self, dataset_name: str) -> bool:
        """
        Download dataset from source and upload to S3.
        This implementation assumes source datasets come from local paths,
        but can be extended to download from external sources.

        Args:
            dataset_name (str): Name of the dataset to download

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # This is a placeholder for actual dataset download logic
            # In a real implementation, you might download from HuggingFace, academic sources, etc.
            logger.info(f"Downloading dataset {dataset_name} from source")

            # Example: for now just check if we have a local copy to upload to S3
            local_dataset_path = self._get_local_path("raw_dir", f"{dataset_name}.json")

            if local_dataset_path.exists():
                # Upload to S3
                s3_key = self._get_s3_path("raw_dir", f"{dataset_name}.json")
                success = self.s3.upload_file(local_dataset_path, s3_key)

                if success:
                    logger.info(f"Uploaded {dataset_name} to S3 at {s3_key}")
                    return True
                else:
                    logger.error(f"Failed to upload {dataset_name} to S3")
                    return False
            else:
                logger.error(f"Local dataset file {local_dataset_path} not found")
                return False

        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
            return False

    def convert_dataset(self, source_dataset: str, output_name: str) -> bool:
        """
        Convert a dataset to the format required for fine-tuning.
        Downloads from S3, processes locally, and uploads results back to S3.

        Args:
            source_dataset (str): Name of the source dataset
            output_name (str): Name for the converted dataset

        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Download source dataset from S3
            source_s3_key = self._get_s3_path("raw_dir", f"{source_dataset}.json")
            local_source_path = self._get_local_path(
                "raw_dir", f"{source_dataset}.json"
            )

            if not self.s3.download_file(source_s3_key, local_source_path):
                logger.error(
                    f"Failed to download source dataset from S3: {source_s3_key}"
                )
                return False

            # Load the source dataset
            with open(local_source_path, "r", encoding="utf-8") as f:
                source_data = json.load(f)

            # Convert the dataset (placeholder for actual conversion logic)
            # In a real implementation, this would transform the data structure
            logger.info(
                f"Converting {source_dataset} to format required for fine-tuning"
            )

            # Example conversion for a hypothetical dataset
            converted_data = []
            for item in source_data:
                # This is a placeholder - replace with actual conversion logic
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
            local_output_path = self._get_local_path(
                "processed_dir", f"{output_name}.json"
            )
            with open(local_output_path, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)

            # Upload to S3
            s3_output_key = self._get_s3_path("processed_dir", f"{output_name}.json")
            if not self.s3.upload_file(local_output_path, s3_output_key):
                logger.error(
                    f"Failed to upload converted dataset to S3: {s3_output_key}"
                )
                return False

            logger.info(
                f"Dataset conversion completed and uploaded to S3 at {s3_output_key}"
            )
            return True

        except Exception as e:
            logger.error(f"Error converting dataset: {str(e)}")
            return False

    def preprocess_data(self, dataset_name: str, output_name: str) -> bool:
        """
        Preprocess the dataset with text normalization and cleaning.
        Downloads from S3, processes locally, and uploads results back to S3.

        Args:
            dataset_name (str): Name of the dataset to preprocess
            output_name (str): Name for the preprocessed dataset

        Returns:
            bool: True if preprocessing successful, False otherwise
        """
        try:
            # Download dataset from S3
            source_s3_key = self._get_s3_path("processed_dir", f"{dataset_name}.json")
            local_source_path = self._get_local_path(
                "processed_dir", f"{dataset_name}.json"
            )

            if not self.s3.download_file(source_s3_key, local_source_path):
                logger.error(f"Failed to download dataset from S3: {source_s3_key}")
                return False

            # Load the dataset
            with open(local_source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Preprocess the dataset (placeholder for actual preprocessing logic)
            logger.info(f"Preprocessing dataset {dataset_name}")

            # Example preprocessing
            for item in data:
                for message in item.get("messages", []):
                    # Simple text normalization (placeholder)
                    if isinstance(message.get("content"), str):
                        # Normalize whitespace
                        message["content"] = " ".join(message["content"].split())
                        # Strip extra spaces around punctuation
                        for punct in [".", ",", "!", "?", ":", ";"]:
                            message["content"] = message["content"].replace(
                                f" {punct}", punct
                            )

            # Save locally
            local_output_path = self._get_local_path(
                "processed_dir", f"{output_name}.json"
            )
            with open(local_output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Upload to S3
            s3_output_key = self._get_s3_path("processed_dir", f"{output_name}.json")
            if not self.s3.upload_file(local_output_path, s3_output_key):
                logger.error(
                    f"Failed to upload preprocessed dataset to S3: {s3_output_key}"
                )
                return False

            logger.info(
                f"Dataset preprocessing completed and uploaded to S3 at {s3_output_key}"
            )
            return True

        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            return False

    def validate_data(self, dataset_name: str, output_name: str) -> bool:
        """
        Validate the dataset for quality and format.
        Downloads from S3, validates locally, and uploads results back to S3.

        Args:
            dataset_name (str): Name of the dataset to validate
            output_name (str): Name for the validated dataset

        Returns:
            bool: True if validation successful, False otherwise
        """
        try:
            # Download dataset from S3
            source_s3_key = self._get_s3_path("processed_dir", f"{dataset_name}.json")
            local_source_path = self._get_local_path(
                "processed_dir", f"{dataset_name}.json"
            )

            if not self.s3.download_file(source_s3_key, local_source_path):
                logger.error(f"Failed to download dataset from S3: {source_s3_key}")
                return False

            # Load the dataset
            with open(local_source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate the dataset (placeholder for actual validation logic)
            logger.info(f"Validating dataset {dataset_name}")

            valid_data = []
            validation_stats = {
                "total": len(data),
                "valid": 0,
                "invalid": 0,
                "reasons": {},
            }

            # Example validation
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

                # Add more validation checks as needed

                if is_valid:
                    valid_data.append(item)
                    validation_stats["valid"] += 1
                else:
                    validation_stats["invalid"] += 1
                    validation_stats["reasons"][reason] = (
                        validation_stats["reasons"].get(reason, 0) + 1
                    )

            # Save valid data locally
            local_output_path = self._get_local_path(
                "validation_dir", f"{output_name}.json"
            )
            with open(local_output_path, "w", encoding="utf-8") as f:
                json.dump(valid_data, f, indent=2, ensure_ascii=False)

            # Save validation stats
            local_stats_path = self._get_local_path(
                "validation_dir", f"{output_name}_stats.json"
            )
            with open(local_stats_path, "w", encoding="utf-8") as f:
                json.dump(validation_stats, f, indent=2)

            # Upload to S3
            s3_output_key = self._get_s3_path("validation_dir", f"{output_name}.json")
            s3_stats_key = self._get_s3_path(
                "validation_dir", f"{output_name}_stats.json"
            )

            success = True
            if not self.s3.upload_file(local_output_path, s3_output_key):
                logger.error(
                    f"Failed to upload validated dataset to S3: {s3_output_key}"
                )
                success = False

            if not self.s3.upload_file(local_stats_path, s3_stats_key):
                logger.error(f"Failed to upload validation stats to S3: {s3_stats_key}")
                success = False

            if success:
                logger.info(
                    f"Dataset validation completed: {validation_stats['valid']}/{validation_stats['total']} valid samples"
                )
                logger.info(f"Validated dataset uploaded to S3 at {s3_output_key}")

            return success

        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return False

    def tokenize_data(self, dataset_name: str, output_name: str) -> bool:
        """
        Tokenize the dataset using the model's tokenizer.
        Downloads from S3, tokenizes locally, and uploads results back to S3.

        Args:
            dataset_name (str): Name of the dataset to tokenize
            output_name (str): Name for the tokenized dataset

        Returns:
            bool: True if tokenization successful, False otherwise
        """
        try:
            # Download dataset from S3
            source_s3_key = self._get_s3_path("validation_dir", f"{dataset_name}.json")
            local_source_path = self._get_local_path(
                "validation_dir", f"{dataset_name}.json"
            )

            if not self.s3.download_file(source_s3_key, local_source_path):
                logger.error(f"Failed to download dataset from S3: {source_s3_key}")
                return False

            # Load the dataset
            with open(local_source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load tokenizer
            logger.info(f"Loading tokenizer for {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Process and tokenize conversations
            logger.info(f"Tokenizing dataset {dataset_name}")
            tokenized_data = []

            for item in data:
                # Format according to the Qwen2.5 chat template
                if "messages" in item:
                    try:
                        # Apply chat template to create formatted prompt
                        tokenized_result = tokenizer.apply_chat_template(
                            item["messages"],
                            return_tensors="pt",
                            add_generation_prompt=False,
                        )

                        # Convert tensor to integers and then to list
                        token_ids = tokenized_result.tolist()[0]

                        tokenized_item = {
                            "input_ids": token_ids,
                            # Save original messages for reference
                            "original_messages": item["messages"],
                        }

                        tokenized_data.append(tokenized_item)
                    except Exception as e:
                        logger.warning(f"Failed to tokenize item: {str(e)}")

            # Save tokenized data locally
            local_output_path = self._get_local_path(
                "processed_dir", f"{output_name}.pt"
            )
            torch.save(tokenized_data, local_output_path)

            # Upload to S3
            s3_output_key = self._get_s3_path("processed_dir", f"{output_name}.pt")
            if not self.s3.upload_file(local_output_path, s3_output_key):
                logger.error(
                    f"Failed to upload tokenized dataset to S3: {s3_output_key}"
                )
                return False

            logger.info(
                f"Dataset tokenization completed with {len(tokenized_data)} samples"
            )
            logger.info(f"Tokenized dataset uploaded to S3 at {s3_output_key}")
            return True

        except Exception as e:
            logger.error(f"Error tokenizing dataset: {str(e)}")
            return False

    def split_dataset(self, dataset_name: str, train_ratio: float = 0.9) -> bool:
        """
        Split the tokenized dataset into train/validation/test sets.
        Downloads from S3, splits locally, and uploads results back to S3.

        Args:
            dataset_name (str): Name of the tokenized dataset to split
            train_ratio (float): Ratio of data to use for training

        Returns:
            bool: True if splitting successful, False otherwise
        """
        try:
            # Download tokenized dataset from S3
            source_s3_key = self._get_s3_path("processed_dir", f"{dataset_name}.pt")
            local_source_path = self._get_local_path(
                "processed_dir", f"{dataset_name}.pt"
            )

            if not self.s3.download_file(source_s3_key, local_source_path):
                logger.error(
                    f"Failed to download tokenized dataset from S3: {source_s3_key}"
                )
                return False

            # Load the tokenized dataset
            tokenized_data = torch.load(local_source_path)

            # Shuffle and split the dataset
            logger.info(
                f"Splitting dataset {dataset_name} into train/validation/test sets"
            )
            import random

            random.shuffle(tokenized_data)

            total_samples = len(tokenized_data)
            train_size = int(total_samples * train_ratio)
            val_size = int(total_samples * (1 - train_ratio) / 2)
            test_size = total_samples - train_size - val_size

            train_data = tokenized_data[:train_size]
            val_data = tokenized_data[train_size : train_size + val_size]
            test_data = tokenized_data[train_size + val_size :]

            # Save splits locally
            local_train_path = self._get_local_path(
                "processed_dir", f"{dataset_name}_train.pt"
            )
            local_val_path = self._get_local_path(
                "processed_dir", f"{dataset_name}_val.pt"
            )
            local_test_path = self._get_local_path(
                "processed_dir", f"{dataset_name}_test.pt"
            )

            torch.save(train_data, local_train_path)
            torch.save(val_data, local_val_path)
            torch.save(test_data, local_test_path)

            # Upload to S3
            s3_train_key = self._get_s3_path(
                "processed_dir", f"{dataset_name}_train.pt"
            )
            s3_val_key = self._get_s3_path("processed_dir", f"{dataset_name}_val.pt")
            s3_test_key = self._get_s3_path("processed_dir", f"{dataset_name}_test.pt")

            success = True
            if not self.s3.upload_file(local_train_path, s3_train_key):
                logger.error(f"Failed to upload train split to S3: {s3_train_key}")
                success = False

            if not self.s3.upload_file(local_val_path, s3_val_key):
                logger.error(f"Failed to upload validation split to S3: {s3_val_key}")
                success = False

            if not self.s3.upload_file(local_test_path, s3_test_key):
                logger.error(f"Failed to upload test split to S3: {s3_test_key}")
                success = False

            if success:
                logger.info(
                    f"Dataset split completed: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples"
                )
                logger.info(f"Dataset splits uploaded to S3")

            return success

        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            return False

    def run_pipeline(self, source_dataset: str, output_prefix: str = None) -> bool:
        """
        Run the complete data processing pipeline from source dataset to tokenized splits.

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

        # Step 1: Download from source or use existing S3 data
        logger.info("Step 1: Downloading dataset")
        if not self.download_from_source(source_dataset):
            logger.error("Pipeline failed at download stage")
            return False

        # Step 2: Convert dataset format
        logger.info("Step 2: Converting dataset format")
        if not self.convert_dataset(source_dataset, converted_name):
            logger.error("Pipeline failed at conversion stage")
            return False

        # Step 3: Preprocess data
        logger.info("Step 3: Preprocessing dataset")
        if not self.preprocess_data(converted_name, preprocessed_name):
            logger.error("Pipeline failed at preprocessing stage")
            return False

        # Step 4: Validate data
        logger.info("Step 4: Validating dataset")
        if not self.validate_data(preprocessed_name, validated_name):
            logger.error("Pipeline failed at validation stage")
            return False

        # Step 5: Tokenize data
        logger.info("Step 5: Tokenizing dataset")
        if not self.tokenize_data(validated_name, tokenized_name):
            logger.error("Pipeline failed at tokenization stage")
            return False

        # Step 6: Split dataset
        logger.info("Step 6: Splitting dataset")
        if not self.split_dataset(tokenized_name):
            logger.error("Pipeline failed at splitting stage")
            return False

        logger.info("Data pipeline completed successfully!")
        return True
