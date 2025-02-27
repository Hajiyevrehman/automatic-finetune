#!/usr/bin/env python3
"""
Create a DVC pipeline configuration for the S3-integrated data pipeline.

This script:
1. Creates a dvc.yaml file with commands that use the S3 data pipeline
2. Configures the pipeline stages properly with dependencies and outputs
3. Adds parameters from configuration files

Usage:
    python scripts/create_dvc_s3_pipeline.py --bucket my-llm-data-bucket --dataset servicenow-qa
"""

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load a configuration file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration data
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {config_path}")


def create_dvc_pipeline(bucket_name, dataset_name, output_prefix=None):
    """
    Create a DVC pipeline configuration for the S3-integrated data pipeline.

    Args:
        bucket_name (str): S3 bucket name for data storage
        dataset_name (str): Name of the source dataset
        output_prefix (str, optional): Prefix for output dataset names
    """
    if output_prefix is None:
        output_prefix = dataset_name

    # Load data processing config
    config_path = "configs/data/data_processing.json"
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        # Create a basic config if not found
        config = {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "validation_dir": "data/validation",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
        }
        os.makedirs("configs/data", exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Define stage names
    converted_name = f"{output_prefix}_converted"
    preprocessed_name = f"{output_prefix}_preprocessed"
    validated_name = f"{output_prefix}_validated"
    tokenized_name = f"{output_prefix}_tokenized"

    # Create DVC pipeline configuration
    dvc_config = {
        "stages": {
            "download": {
                "cmd": f"python -m src.cli.data_cli download --bucket {bucket_name} --dataset {dataset_name} --config {config_path}",
                "deps": [
                    "src/data/s3_pipeline.py",
                    "src/cloud/storage.py",
                    "src/cloud/auth.py",
                    config_path,
                ],
                "params": [f"{config_path}:raw_dir", f"{config_path}:model_name"],
                "outs": [f"{config['raw_dir']}/{dataset_name}.json.dvc"],
            },
            "convert": {
                "cmd": f"python -m src.cli.data_cli convert --bucket {bucket_name} --dataset {dataset_name} --output {converted_name} --config {config_path}",
                "deps": [
                    "src/data/s3_pipeline.py",
                    "src/cloud/storage.py",
                    f"{config['raw_dir']}/{dataset_name}.json.dvc",
                ],
                "params": [f"{config_path}:raw_dir", f"{config_path}:processed_dir"],
                "outs": [f"{config['processed_dir']}/{converted_name}.json.dvc"],
            },
            "preprocess": {
                "cmd": f"python -m src.cli.data_cli preprocess --bucket {bucket_name} --dataset {converted_name} --output {preprocessed_name} --config {config_path}",
                "deps": [
                    "src/data/s3_pipeline.py",
                    "src/cloud/storage.py",
                    f"{config['processed_dir']}/{converted_name}.json.dvc",
                ],
                "params": [f"{config_path}:processed_dir"],
                "outs": [f"{config['processed_dir']}/{preprocessed_name}.json.dvc"],
            },
            "validate": {
                "cmd": f"python -m src.cli.data_cli validate --bucket {bucket_name} --dataset {preprocessed_name} --output {validated_name} --config {config_path}",
                "deps": [
                    "src/data/s3_pipeline.py",
                    "src/cloud/storage.py",
                    f"{config['processed_dir']}/{preprocessed_name}.json.dvc",
                ],
                "params": [
                    f"{config_path}:processed_dir",
                    f"{config_path}:validation_dir",
                ],
                "outs": [
                    f"{config['validation_dir']}/{validated_name}.json.dvc",
                    f"{config['validation_dir']}/{validated_name}_stats.json.dvc",
                ],
            },
            "tokenize": {
                "cmd": f"python -m src.cli.data_cli tokenize --bucket {bucket_name} --dataset {validated_name} --output {tokenized_name} --config {config_path}",
                "deps": [
                    "src/data/s3_pipeline.py",
                    "src/cloud/storage.py",
                    f"{config['validation_dir']}/{validated_name}.json.dvc",
                ],
                "params": [
                    f"{config_path}:validation_dir",
                    f"{config_path}:processed_dir",
                    f"{config_path}:model_name",
                ],
                "outs": [f"{config['processed_dir']}/{tokenized_name}.pt.dvc"],
            },
            "split": {
                "cmd": f"python -m src.cli.data_cli split --bucket {bucket_name} --dataset {tokenized_name} --config {config_path}",
                "deps": [
                    "src/data/s3_pipeline.py",
                    "src/cloud/storage.py",
                    f"{config['processed_dir']}/{tokenized_name}.pt.dvc",
                ],
                "params": [f"{config_path}:processed_dir"],
                "outs": [
                    f"{config['processed_dir']}/{tokenized_name}_train.pt.dvc",
                    f"{config['processed_dir']}/{tokenized_name}_val.pt.dvc",
                    f"{config['processed_dir']}/{tokenized_name}_test.pt.dvc",
                ],
            },
        }
    }

    # Write DVC pipeline configuration to dvc.yaml
    with open("dvc.yaml", "w") as f:
        yaml.dump(dvc_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Created DVC pipeline configuration: dvc.yaml")
    logger.info(f"Pipeline configured with S3 bucket: {bucket_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("To run the pipeline: dvc repro")


def main():
    """Main function to create a DVC pipeline configuration."""
    parser = argparse.ArgumentParser(
        description="Create a DVC pipeline configuration for the S3-integrated data pipeline"
    )
    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name for data storage"
    )
    parser.add_argument("--dataset", required=True, help="Name of the source dataset")
    parser.add_argument("--output-prefix", help="Prefix for output dataset names")

    args = parser.parse_args()

    create_dvc_pipeline(args.bucket, args.dataset, args.output_prefix)
    return 0


if __name__ == "__main__":
    exit(main())
