# src/data/load.py
import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QADataset(Dataset):
    """
    Dataset class for Q&A pairs.
    """

    def __init__(self, data, tokenizer=None, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # If data is already tokenized, return tokenized data
        if "tokenized" in item:
            tokenized = item["tokenized"]
            return {
                "input_ids": torch.tensor(tokenized["input_ids"]),
                "attention_mask": torch.tensor(tokenized["attention_mask"]),
                "question": item["question"],
                "answer": item["answer"],
            }

        # Otherwise tokenize on-the-fly if tokenizer is provided
        elif self.tokenizer:
            question = item.get("question", "")
            answer = item.get("answer", "")

            # Format as instruction-following format
            formatted_text = f"USER: {question}\nASSISTANT: {answer}"

            # Tokenize
            tokenized = self.tokenizer(
                formatted_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "question": question,
                "answer": answer,
            }

        # Return raw data if no tokenizer
        else:
            return item


def load_dataset_from_json(file_path):
    """
    Load a dataset from a JSON file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        list: List of data samples
    """
    logger.info(f"Loading dataset from {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        raise


def create_dataloader(data, batch_size=8, shuffle=True, tokenizer=None, max_length=512):
    """
    Create a DataLoader for the dataset.

    Args:
        data (list): List of data samples
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        tokenizer: Optional tokenizer for on-the-fly tokenization
        max_length (int): Maximum sequence length

    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = QADataset(data, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 2,
    )

    return dataloader


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        data (list): List of data samples
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        shuffle (bool): Whether to shuffle the data before splitting

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    import random

    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10
    ), "Ratios must sum to 1"

    # Make a copy of the data
    data_copy = data.copy()

    # Shuffle if required
    if shuffle:
        random.shuffle(data_copy)

    n = len(data_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]

    logger.info(
        f"Split dataset into {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test samples"
    )

    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description="Load and split dataset")
    parser.add_argument("input", type=str, help="Path to input JSON file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save the split datasets",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of training data"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Ratio of validation data"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Ratio of test data"
    )

    args = parser.parse_args()

    # Load dataset
    data = load_dataset_from_json(args.input)

    # Split dataset
    train_data, val_data, test_data = split_dataset(
        data, args.train_ratio, args.val_ratio, args.test_ratio
    )

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save the split datasets
    with open(os.path.join(args.output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved split datasets to {args.output_dir}")


if __name__ == "__main__":
    main()
