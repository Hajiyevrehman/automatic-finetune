#!/usr/bin/env python3
"""
Command-line interface for the S3-integrated data pipeline.

This module provides commands to run individual stages of the data pipeline
or the entire pipeline end-to-end, with S3 integration.

Usage:
    # Run the complete pipeline
    python -m src.cli.data_cli run-pipeline --bucket my-llm-data-bucket --dataset servicenow-qa

    # Run individual stages
    python -m src.cli.data_cli download --bucket my-llm-data-bucket --dataset servicenow-qa
    python -m src.cli.data_cli convert --bucket my-llm-data-bucket --dataset servicenow-qa --output servicenow-qa_converted
    # ...and so on
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.cloud.auth import create_bucket_if_not_exists, validate_s3_connection
from src.data.s3_pipeline import S3DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def setup_pipeline(args):
    """
    Set up the S3DataPipeline instance.

    Args:
        args (Namespace): Command line arguments

    Returns:
        S3DataPipeline: Configured pipeline instance
    """
    # Ensure the S3 bucket exists
    if not create_bucket_if_not_exists(args.bucket):
        logger.error(f"Failed to create or access S3 bucket: {args.bucket}")
        return None

    # Validate the connection
    if not validate_s3_connection(args.bucket):
        logger.error(f"Cannot connect to S3 bucket: {args.bucket}")
        return None

    # Create the pipeline
    try:
        return S3DataPipeline(args.config, args.bucket)
    except Exception as e:
        logger.error(f"Failed to create data pipeline: {str(e)}")
        return None


def download_cmd(args):
    """Execute the download stage of the pipeline."""
    pipeline = setup_pipeline(args)
    if pipeline:
        return pipeline.download_from_source(args.dataset)
    return False


def convert_cmd(args):
    """Execute the convert stage of the pipeline."""
    pipeline = setup_pipeline(args)
    if pipeline:
        return pipeline.convert_dataset(args.dataset, args.output)
    return False


def preprocess_cmd(args):
    """Execute the preprocess stage of the pipeline."""
    pipeline = setup_pipeline(args)
    if pipeline:
        return pipeline.preprocess_data(args.dataset, args.output)
    return False


def validate_cmd(args):
    """Execute the validate stage of the pipeline."""
    pipeline = setup_pipeline(args)
    if pipeline:
        return pipeline.validate_data(args.dataset, args.output)
    return False


def tokenize_cmd(args):
    """Execute the tokenize stage of the pipeline."""
    pipeline = setup_pipeline(args)
    if pipeline:
        return pipeline.tokenize_data(args.dataset, args.output)
    return False


def split_cmd(args):
    """Execute the split stage of the pipeline."""
    pipeline = setup_pipeline(args)
    if pipeline:
        return pipeline.split_dataset(args.dataset, args.train_ratio)
    return False


def run_pipeline_cmd(args):
    """Execute the complete data pipeline."""
    pipeline = setup_pipeline(args)
    if pipeline:
        return pipeline.run_pipeline(args.dataset, args.output_prefix)
    return False


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="S3-integrated data pipeline for LLM fine-tuning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--bucket", required=True, help="S3 bucket name for data storage"
    )
    common_parser.add_argument(
        "--config",
        default="configs/data/data_processing.json",
        help="Path to data processing configuration file",
    )

    # Download command
    download_parser = subparsers.add_parser(
        "download", parents=[common_parser], help="Download dataset from source"
    )
    download_parser.add_argument(
        "--dataset", required=True, help="Name of the dataset to download"
    )
    download_parser.set_defaults(func=download_cmd)

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", parents=[common_parser], help="Convert dataset format"
    )
    convert_parser.add_argument(
        "--dataset", required=True, help="Name of the source dataset"
    )
    convert_parser.add_argument(
        "--output", required=True, help="Name for the converted dataset"
    )
    convert_parser.set_defaults(func=convert_cmd)

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess", parents=[common_parser], help="Preprocess dataset"
    )
    preprocess_parser.add_argument(
        "--dataset", required=True, help="Name of the dataset to preprocess"
    )
    preprocess_parser.add_argument(
        "--output", required=True, help="Name for the preprocessed dataset"
    )
    preprocess_parser.set_defaults(func=preprocess_cmd)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", parents=[common_parser], help="Validate dataset"
    )
    validate_parser.add_argument(
        "--dataset", required=True, help="Name of the dataset to validate"
    )
    validate_parser.add_argument(
        "--output", required=True, help="Name for the validated dataset"
    )
    validate_parser.set_defaults(func=validate_cmd)

    # Tokenize command
    tokenize_parser = subparsers.add_parser(
        "tokenize", parents=[common_parser], help="Tokenize dataset"
    )
    tokenize_parser.add_argument(
        "--dataset", required=True, help="Name of the dataset to tokenize"
    )
    tokenize_parser.add_argument(
        "--output", required=True, help="Name for the tokenized dataset"
    )
    tokenize_parser.set_defaults(func=tokenize_cmd)

    # Split command
    split_parser = subparsers.add_parser(
        "split",
        parents=[common_parser],
        help="Split dataset into train/validation/test sets",
    )
    split_parser.add_argument(
        "--dataset", required=True, help="Name of the tokenized dataset to split"
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (default: 0.9)",
    )
    split_parser.set_defaults(func=split_cmd)

    # Run complete pipeline command
    pipeline_parser = subparsers.add_parser(
        "run-pipeline", parents=[common_parser], help="Run the complete data pipeline"
    )
    pipeline_parser.add_argument(
        "--dataset", required=True, help="Name of the source dataset"
    )
    pipeline_parser.add_argument(
        "--output-prefix", help="Prefix for output dataset names"
    )
    pipeline_parser.set_defaults(func=run_pipeline_cmd)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    success = args.func(args)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
