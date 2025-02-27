"""
Data loading module for the training pipeline.
Handles loading tokenized datasets from S3 for model training.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, Features, Sequence, Value
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.cloud.storage import S3Storage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parse an S3 path into bucket name and key.

    Args:
        s3_path: S3 path in the format s3://bucket-name/key

    Returns:
        Tuple of (bucket_name, key)
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with s3://")

    path_without_prefix = s3_path[5:]  # Remove "s3://"
    parts = path_without_prefix.split("/", 1)

    bucket_name = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    return bucket_name, key


def download_from_s3(s3_path: str, local_dir: str = "data/cache") -> str:
    """
    Download a file from S3 to a local directory.

    Args:
        s3_path: S3 path in the format s3://bucket-name/key
        local_dir: Local directory to download to

    Returns:
        Local path to the downloaded file
    """
    bucket_name, key = parse_s3_path(s3_path)

    # Create cache directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Determine local filename
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)

    # Download the file
    s3_storage = S3Storage(bucket_name)
    success = s3_storage.download_file(key, local_path)

    if not success:
        raise FileNotFoundError(f"Failed to download file from {s3_path}")

    return local_path


def load_pt_dataset(
    file_path: str, use_cache: bool = True, cache_dir: str = "data/cache"
) -> Dataset:
    """
    Load a PyTorch dataset from a local file or S3 (handles .pt files).

    Args:
        file_path: Path to the dataset file (local or S3)
        use_cache: Whether to use cached files
        cache_dir: Directory for cached files

    Returns:
        Dataset: PyTorch dataset
    """
    # If S3 path, download to local cache
    if file_path.startswith("s3://"):
        if use_cache:
            # Check if file exists in cache
            filename = os.path.basename(file_path)
            cached_path = os.path.join(cache_dir, filename)

            if not os.path.exists(cached_path):
                logger.info(
                    f"File not found in cache, downloading from S3: {file_path}"
                )
                file_path = download_from_s3(file_path, cache_dir)
            else:
                logger.info(f"Using cached file: {cached_path}")
                file_path = cached_path
        else:
            # Always download from S3
            file_path = download_from_s3(file_path, cache_dir)

    # Check file extension
    if file_path.endswith(".pt") or file_path.endswith(".pth"):
        try:
            # Load PyTorch dataset
            logger.info(f"Loading PyTorch dataset from {file_path}")
            data = torch.load(file_path)

            # Process based on the data structure from your tokenization pipeline
            if isinstance(data, list):
                logger.info(f"Found list of {len(data)} tokenized samples")

                # Extract fields
                input_ids_list = []
                attention_masks = []

                for item in data:
                    if "input_ids" in item:
                        # Get input_ids
                        token_ids = item["input_ids"]
                        input_ids_list.append(token_ids)

                        # Create attention mask of all 1s if not present
                        attention_mask = [1] * len(token_ids)
                        attention_masks.append(attention_mask)

                if input_ids_list:
                    # Convert to dataset
                    dataset_dict = {
                        "input_ids": input_ids_list,
                        "attention_mask": attention_masks,
                        "labels": input_ids_list,  # Use input_ids as labels for causal LM
                    }

                    # Convert lists to tensors if needed
                    features = Features(
                        {
                            "input_ids": Sequence(Value("int32")),
                            "attention_mask": Sequence(Value("int8")),
                            "labels": Sequence(Value("int32")),
                        }
                    )

                    dataset = Dataset.from_dict(dataset_dict, features=features)
                    logger.info(
                        f"Successfully loaded PyTorch dataset with {len(dataset)} examples"
                    )
                    return dataset
                else:
                    raise ValueError("No valid input_ids found in PyTorch file")

            # Handle potential dictionary structure
            elif isinstance(data, dict) and "input_ids" in data:
                # Convert to dataset
                dataset = Dataset.from_dict(
                    {
                        "input_ids": data.get("input_ids", []),
                        "attention_mask": data.get("attention_mask", []),
                        "labels": data.get("labels", data.get("input_ids", [])),
                    }
                )
                logger.info(
                    f"Successfully loaded PyTorch dataset with {len(dataset)} examples"
                )
                return dataset
            else:
                raise ValueError(
                    f"PyTorch file does not contain expected tensor format"
                )
        except Exception as e:
            logger.error(f"Failed to load PyTorch dataset: {e}")
            raise
    else:
        raise ValueError(
            f"Unsupported file format for {file_path}. Expected .pt or .pth file."
        )


def load_and_prepare_data(
    train_file: str,
    val_file: Optional[str] = None,
    batch_size: int = 4,
    eval_batch_size: Optional[int] = None,
    use_cache: bool = True,
    num_workers: int = 4,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load, prepare, and create datasets for training.

    Args:
        train_file: Path to training data file (local or S3)
        val_file: Path to validation data file (local or S3)
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        use_cache: Whether to use cached files
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Load datasets
    logger.info(f"Loading training dataset from {train_file}")
    if train_file.endswith(".pt") or train_file.endswith(".pth"):
        train_dataset = load_pt_dataset(train_file, use_cache)
    else:
        raise ValueError(f"Unsupported file format for {train_file}")

    eval_dataset = None
    if val_file:
        logger.info(f"Loading validation dataset from {val_file}")
        if val_file.endswith(".pt") or val_file.endswith(".pth"):
            eval_dataset = load_pt_dataset(val_file, use_cache)
        else:
            raise ValueError(f"Unsupported file format for {val_file}")

    return train_dataset, eval_dataset


def create_training_dataloaders(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    batch_size: int = 4,
    eval_batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training and evaluation.

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size (defaults to batch_size if None)
        shuffle: Whether to shuffle the training data
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_dataloader, eval_dataloader
