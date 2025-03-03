#!/usr/bin/env python3
"""
Setup DVC with S3 remote storage.

This script:
1. Initializes DVC if needed
2. Configures S3 remote storage
3. Creates a DVC pipeline for the data processing workflow

Usage:
    python scripts/utils/setup_dvc.py --bucket your-bucket-name --dataset servicenow-qa [--init] [--no-scm]
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(command, check=True):
    """
    Run a shell command and log the output.

    Args:
        command (list): Command to run as a list of strings
        check (bool): Whether to raise an exception if the command fails

    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=check, text=True, capture_output=True)
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.info(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        return False


def is_dvc_initialized():
    """
    Check if DVC is initialized in the current directory.

    Returns:
        bool: True if DVC is initialized, False otherwise
    """
    return Path(".dvc").is_dir()


def init_dvc(no_scm=False):
    """
    Initialize DVC in the current directory.

    Args:
        no_scm (bool): Whether to initialize DVC without Git integration

    Returns:
        bool: True if initialization succeeded, False otherwise
    """
    if is_dvc_initialized():
        logger.info("DVC is already initialized")
        return True

    command = ["dvc", "init"]
    if no_scm:
        command.append("--no-scm")

    return run_command(command)


def configure_s3_remote(bucket_name):
    """
    Configure DVC to use S3 as remote storage.

    Args:
        bucket_name (str): Name of the S3 bucket to use

    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    # Add S3 remote
    s3_url = f"s3://{bucket_name}"
    add_remote_cmd = ["dvc", "remote", "add", "--default", "storage", s3_url]
    if not run_command(add_remote_cmd):
        return False

    # Configure remote to use environment variables for credentials
    config_cmd = [
        "dvc",
        "remote",
        "modify",
        "storage",
        "endpointurl",
        "https://s3.amazonaws.com",
    ]
    if not run_command(config_cmd):
        return False

    logger.info(f"DVC configured with S3 remote at {s3_url}")
    return True


def create_dvc_pipeline(
    bucket_name, dataset_name, config_path="configs/data/data_processing.yaml"
):
    """
    Create a DVC pipeline configuration for the data processing workflow.

    Args:
        bucket_name (str): S3 bucket name
        dataset_name (str): Name of the source dataset
        config_path (str): Path to configuration file
    """
    # Define output names
    output_prefix = dataset_name
    converted_name = f"{output_prefix}_converted"
    preprocessed_name = f"{output_prefix}_preprocessed"
    validated_name = f"{output_prefix}_validated"
    tokenized_name = f"{output_prefix}_tokenized"

    # Create DVC pipeline configuration
    dvc_config = {
        "stages": {
            "download": {
                "cmd": f"python scripts/dataset_converters/{dataset_name}_converter.py --output_path data/raw/{dataset_name}.json",
                "deps": [f"scripts/dataset_converters/{dataset_name}_converter.py"],
                "outs": [f"data/raw/{dataset_name}.json"],
            },
            "convert": {
                "cmd": f"python -m src.cli.data_cli convert --dataset {dataset_name} --output {converted_name} --config {config_path}",
                "deps": [
                    "src/data/pipeline.py",
                    f"data/raw/{dataset_name}.json",
                    config_path,
                ],
                "outs": [f"data/processed/{converted_name}.json"],
            },
            "preprocess": {
                "cmd": f"python -m src.cli.data_cli preprocess --dataset {converted_name} --output {preprocessed_name} --config {config_path}",
                "deps": [
                    "src/data/pipeline.py",
                    f"data/processed/{converted_name}.json",
                    config_path,
                ],
                "outs": [f"data/processed/{preprocessed_name}.json"],
            },
            "validate": {
                "cmd": f"python -m src.cli.data_cli validate --dataset {preprocessed_name} --output {validated_name} --config {config_path}",
                "deps": [
                    "src/data/pipeline.py",
                    f"data/processed/{preprocessed_name}.json",
                    config_path,
                ],
                "outs": [
                    f"data/validation/{validated_name}.json",
                    f"data/validation/{validated_name}_stats.json",
                ],
            },
            "tokenize": {
                "cmd": f"python -m src.cli.data_cli tokenize --dataset {validated_name} --output {tokenized_name} --config {config_path}",
                "deps": [
                    "src/data/pipeline.py",
                    f"data/validation/{validated_name}.json",
                    config_path,
                ],
                "outs": [f"data/processed/{tokenized_name}.pt"],
            },
            "split": {
                "cmd": f"python -m src.cli.data_cli split --dataset {tokenized_name} --config {config_path}",
                "deps": [
                    "src/data/pipeline.py",
                    f"data/processed/{tokenized_name}.pt",
                    config_path,
                ],
                "outs": [
                    f"data/processed/{tokenized_name}_train.pt",
                    f"data/processed/{tokenized_name}_val.pt",
                    f"data/processed/{tokenized_name}_test.pt",
                ],
            },
        }
    }

    # Write DVC pipeline configuration to dvc.yaml
    with open("dvc.yaml", "w") as f:
        yaml.dump(dvc_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Created DVC pipeline configuration: dvc.yaml")
    logger.info(f"Pipeline configured for dataset: {dataset_name}")


def create_gitignore_entries():
    """
    Create or update .gitignore to exclude DVC files if needed.

    Returns:
        bool: True if update succeeded, False otherwise
    """
    gitignore_path = Path(".gitignore")
    entries = [
        "/data/raw",
        "/data/processed",
        "/data/validation",
        "*.dvc",
        "/.dvc/tmp",
        "/.dvc/plots",
        "/.dvc/cache",
    ]

    try:
        existing_content = ""
        if gitignore_path.exists():
            existing_content = gitignore_path.read_text()

        # Add entries that don't already exist
        with open(gitignore_path, "a") as f:
            for entry in entries:
                if entry not in existing_content:
                    f.write(f"\n{entry}")

        logger.info("Updated .gitignore with DVC-related entries")
        return True

    except Exception as e:
        logger.error(f"Failed to update .gitignore: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Set up DVC with S3 remote and create pipeline"
    )
    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name for DVC remote storage"
    )
    parser.add_argument("--dataset", required=True, help="Name of the source dataset")
    parser.add_argument(
        "--config",
        default="configs/data/data_processing.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--init", action="store_true", help="Initialize DVC if not already initialized"
    )
    parser.add_argument(
        "--no-scm", action="store_true", help="Initialize DVC without Git integration"
    )

    args = parser.parse_args()

    # Initialize DVC if requested
    if args.init or not is_dvc_initialized():
        if not init_dvc(args.no_scm):
            logger.error("Failed to initialize DVC. Aborting setup.")
            return 1

    # Configure S3 remote
    if not configure_s3_remote(args.bucket):
        logger.error("Failed to configure S3 remote. Aborting setup.")
        return 1

    # Create DVC pipeline
    create_dvc_pipeline(args.bucket, args.dataset, args.config)

    # Create gitignore entries
    if not args.no_scm:
        create_gitignore_entries()

    # Update config file with bucket name
    try:
        config_path = Path(args.config)
        if config_path.exists():
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                # Ensure s3 section exists
                if "s3" not in config:
                    config["s3"] = {}

                # Update bucket name
                config["s3"]["default_bucket"] = args.bucket

                # Write updated config
                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

                logger.info(
                    f"Updated configuration file with bucket name: {args.bucket}"
                )
    except Exception as e:
        logger.error(f"Error updating config file: {str(e)}")

    logger.info(
        "DVC setup complete. Use 'dvc repro' to run the pipeline, "
        "'dvc push' to upload data to S3, and 'dvc pull' to download data from S3."
    )
    return 0


if __name__ == "__main__":
    exit(main())
