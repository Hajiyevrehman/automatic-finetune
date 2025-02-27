#!/usr/bin/env python3
"""
Test the complete S3-integrated data pipeline.

This script:
1. Uploads a sample dataset to S3
2. Runs each stage of the pipeline
3. Verifies outputs at each stage
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.cloud.storage import S3Storage
from src.data.s3_pipeline import S3DataPipeline

# Load environment variables
load_dotenv()


def test_pipeline():
    """Test the complete data pipeline with S3 integration."""

    # Get S3 bucket name from environment
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        print("Error: S3_BUCKET_NAME not found in environment variables")
        return False

    # Configuration path
    config_path = "configs/data/data_processing.json"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return False

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Ensure directories exist
    for dir_key in ["raw_dir", "processed_dir", "validation_dir"]:
        if dir_key in config:
            os.makedirs(config[dir_key], exist_ok=True)

    # Create S3 storage client
    s3 = S3Storage(bucket_name)

    # Check if sample data exists locally
    sample_path = Path(config["raw_dir"]) / "servicenow-qa.json"
    if not sample_path.exists():
        print(f"Error: Sample dataset not found at {sample_path}")
        return False

    # Upload sample data to S3
    print(f"Uploading sample data to S3...")
    s3_key = f"data/raw/servicenow-qa.json"
    if not s3.upload_file(sample_path, s3_key):
        print("Error: Failed to upload sample data to S3")
        return False

    # Create data pipeline
    print("Initializing data pipeline...")
    pipeline = S3DataPipeline(config_path, bucket_name)

    # Run each stage of the pipeline
    stages = [
        (
            "Converting dataset",
            lambda: pipeline.convert_dataset(
                "servicenow-qa", "servicenow-qa_converted"
            ),
        ),
        (
            "Preprocessing data",
            lambda: pipeline.preprocess_data(
                "servicenow-qa_converted", "servicenow-qa_preprocessed"
            ),
        ),
        (
            "Validating data",
            lambda: pipeline.validate_data(
                "servicenow-qa_preprocessed", "servicenow-qa_validated"
            ),
        ),
        (
            "Tokenizing data",
            lambda: pipeline.tokenize_data(
                "servicenow-qa_validated", "servicenow-qa_tokenized"
            ),
        ),
        (
            "Splitting dataset",
            lambda: pipeline.split_dataset("servicenow-qa_tokenized"),
        ),
    ]

    for stage_name, stage_func in stages:
        print(f"\n--- {stage_name} ---")
        if not stage_func():
            print(f"Error: {stage_name} failed")
            return False
        print(f"{stage_name} completed successfully")

    print("\n✅ Full pipeline test completed successfully!")
    return True


if __name__ == "__main__":
    test_pipeline()
