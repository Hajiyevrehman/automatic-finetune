#!/usr/bin/env python3
"""
ServiceNow dataset converter.

This script downloads and converts the ServiceNow-AI/R1-Distill-SFT dataset
from Hugging Face to a clean Q&A format JSON.

Usage:
    python servicenow_converter.py --subset_size 1000 --version v1 --output_path data/raw/servicenow-qa.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_servicenow_dataset(subset_size=100, version="v1"):
    """
    Load the ServiceNow-AI/R1-Distill-SFT dataset from Hugging Face
    and extract a subset of the data.

    Args:
        subset_size (int): Number of samples to extract
        version (str): Dataset version ('v0' or 'v1')

    Returns:
        list: List of dataset samples
    """
    logger.info(f"Loading ServiceNow-AI/R1-Distill-SFT dataset (version {version})...")
    dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", version, split="train")

    # Take a subset
    subset = dataset.select(range(min(subset_size, len(dataset))))
    logger.info(f"Loaded {len(subset)} samples from the dataset")

    return subset


def convert_to_qa_format(dataset_samples):
    """
    Convert the ServiceNow dataset format to a clean Q&A JSON format.

    Args:
        dataset_samples: Dataset samples

    Returns:
        list: List of Q&A pairs in the format [{"question": "...", "answer": "..."}]
    """
    qa_pairs = []

    for sample in tqdm(dataset_samples, desc="Converting to Q&A format"):
        try:
            # Extract reannotated messages which contain the Q&A pair
            messages = sample.get("reannotated_messages", sample.get("messages", []))

            if len(messages) >= 2:
                # First message is typically the question (user)
                # Second message is typically the answer (assistant)
                user_msg = next(
                    (msg for msg in messages if msg.get("role") == "user"), None
                )
                assistant_msg = next(
                    (msg for msg in messages if msg.get("role") == "assistant"), None
                )

                if user_msg and assistant_msg:
                    # Get content from both messages
                    question = user_msg.get("content", "")
                    answer = assistant_msg.get("content", "")

                    # Remove the <think> sections from the answer if present
                    if "<think>" in answer and "</think>" in answer:
                        think_start = answer.find("<think>")
                        think_end = answer.find("</think>") + len("</think>")
                        answer = answer[:think_start] + answer[think_end:]
                        answer = answer.strip()

                    # Create a Q&A pair
                    qa_pair = {
                        "question": question,
                        "answer": answer,
                        "source": sample.get("source", ""),
                        "source_dataset": sample.get("source_dataset", ""),
                    }

                    qa_pairs.append(qa_pair)
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            continue

    logger.info(f"Successfully converted {len(qa_pairs)} samples to Q&A format")
    return qa_pairs


def save_qa_pairs(qa_pairs, output_path):
    """
    Save the Q&A pairs to a JSON file.

    Args:
        qa_pairs (list): List of Q&A pairs
        output_path (str): Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ServiceNow dataset to Q&A format"
    )
    parser.add_argument(
        "--subset_size", type=int, default=100, help="Number of samples to extract"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        choices=["v0", "v1"],
        help="Dataset version (v0 or v1)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/servicenow-qa.json",
        help="Path to save the output JSON file",
    )

    args = parser.parse_args()

    # Load dataset
    dataset_samples = load_servicenow_dataset(args.subset_size, args.version)

    # Convert to Q&A format
    qa_pairs = convert_to_qa_format(dataset_samples)

    # Save the Q&A pairs
    save_qa_pairs(qa_pairs, args.output_path)

    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    main()
