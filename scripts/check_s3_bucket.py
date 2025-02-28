#!/usr/bin/env python
"""
Script to check the contents of the S3 bucket and create necessary directories.
This ensures all required paths exist for the training pipeline.
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.auth import create_bucket_if_not_exists
from src.cloud.storage import S3Storage


def check_s3_bucket(bucket_name, create_missing=True):
    """
    Check if the S3 bucket exists and has the necessary directory structure.

    Args:
        bucket_name (str): Name of the S3 bucket
        create_missing (bool): Create missing directories if True
    """
    print(f"Checking S3 bucket: {bucket_name}")

    # Ensure bucket exists
    if not create_bucket_if_not_exists(bucket_name):
        print(f"Error: Failed to find or create bucket {bucket_name}")
        return False

    # Initialize S3 storage
    s3 = S3Storage(bucket_name)

    # Define required directories
    required_dirs = [
        "data/raw/",
        "data/processed/",
        "data/validation/",
        "models/",
        "configs/data/",
        "configs/model/",
        "configs/training/",
    ]

    # Check each directory
    for dir_path in required_dirs:
        # List objects with this prefix
        objects = s3.list_objects(prefix=dir_path)

        if not objects and create_missing:
            # Create an empty placeholder object
            placeholder_path = f"{dir_path}placeholder.txt"
            print(f"Creating directory: {dir_path}")
            success = s3.upload_json(
                {"info": f"Created directory {dir_path}"}, placeholder_path
            )

            if not success:
                print(f"Error: Failed to create directory {dir_path}")
        elif not objects:
            print(f"Warning: Directory doesn't exist: {dir_path}")
        else:
            object_count = len(objects)
            print(f"Directory exists: {dir_path} ({object_count} objects)")

            # List first 5 objects in this directory
            if object_count > 0:
                print("  Contents:")
                for i, obj in enumerate(objects[:5]):
                    print(f"    - {obj}")
                if object_count > 5:
                    print(f"    ... and {object_count - 5} more")

    # Special check for processed directory - check for tokenized files
    processed_files = s3.list_objects(prefix="data/processed/")
    tokenized_files = [
        f for f in processed_files if f.endswith(".pt") or f.endswith(".pth")
    ]

    if tokenized_files:
        print("\nFound tokenized files:")
        for file in tokenized_files:
            print(f"  - {file}")
    else:
        print("\nWarning: No tokenized (.pt/.pth) files found in data/processed/")

    print("\nS3 bucket check completed.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check S3 bucket for required directories"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="llm-finetuning-rahman-1234",
        help="S3 bucket name",
    )
    parser.add_argument(
        "--no-create", action="store_true", help="Don't create missing directories"
    )

    args = parser.parse_args()
    check_s3_bucket(args.bucket, not args.no_create)


if __name__ == "__main__":
    main()
