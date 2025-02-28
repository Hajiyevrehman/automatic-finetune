#!/usr/bin/env python
"""
Script for fine-tuning Qwen 2.5 0.5B in Google Colab
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(cmd, desc=None):
    """Run a shell command and log the output"""
    if desc:
        logger.info(desc)

    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}")
        logger.error(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")

    logger.info(result.stdout)
    return result.stdout


def setup_colab_environment(
    repo_url, aws_access_key=None, aws_secret_key=None, aws_region=None
):
    """Set up the Colab environment for training"""

    # Check for GPU
    logger.info("Checking for GPU...")
    run_command("nvidia-smi", "GPU information:")

    # Clone repository
    logger.info(f"Cloning repository: {repo_url}")
    run_command(f"git clone {repo_url} llm-finetuning")
    os.chdir("llm-finetuning")

    # Install dependencies
    logger.info("Installing dependencies...")
    run_command("pip install -r requirements.txt")

    # Set up AWS credentials if provided
    if aws_access_key and aws_secret_key:
        logger.info("Setting up AWS credentials...")
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        os.environ["AWS_REGION"] = aws_region or "us-east-1"

        # Test S3 access
        logger.info("Testing S3 access...")
        try:
            import boto3

            s3 = boto3.client("s3")
            buckets = s3.list_buckets()
            logger.info(
                f"S3 access successful. Found {len(buckets['Buckets'])} buckets."
            )
        except Exception as e:
            logger.error(f"S3 access failed: {str(e)}")

    # Create directories
    logger.info("Creating directories...")
    os.makedirs("models/qwen-0.5b-finetuned", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Add current directory to Python path
    sys.path.append(".")

    logger.info("Colab environment setup complete!")


def update_config_for_colab():
    """Update the training configuration for Colab environment"""
    logger.info("Updating configuration for Colab...")

    config_path = "configs/training/qwen_training_config.yaml"

    # Check if config exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update for Colab environment
    config["training"]["batch_size"] = 2  # Smaller batch size for Colab
    config["training"]["gradient_accumulation_steps"] = 4
    config["training"]["fp16"] = True
    config["training"]["num_epochs"] = 1  # For testing, use just 1 epoch

    # Save updated config
    colab_config_path = "configs/training/qwen_colab_config.yaml"
    with open(colab_config_path, "w") as f:
        yaml.dump(config, f)

    logger.info(f"Updated configuration saved to: {colab_config_path}")
    return colab_config_path


def run_training(config_path):
    """Run the training script with the specified configuration"""
    logger.info(f"Starting training with config: {config_path}")

    cmd = f"PYTHONPATH=. python src/training/train.py --config {config_path}"
    run_command(cmd, "Running training:")

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Run Qwen fine-tuning in Colab")
    parser.add_argument("--repo", type=str, required=True, help="Git repository URL")
    parser.add_argument("--aws-access-key", type=str, help="AWS access key ID")
    parser.add_argument("--aws-secret-key", type=str, help="AWS secret access key")
    parser.add_argument(
        "--aws-region", type=str, default="us-east-1", help="AWS region"
    )

    args = parser.parse_args()

    try:
        # Set up environment
        setup_colab_environment(
            args.repo, args.aws_access_key, args.aws_secret_key, args.aws_region
        )

        # Update configuration
        config_path = update_config_for_colab()

        # Run training
        run_training(config_path)

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.exception(f"Error during execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
