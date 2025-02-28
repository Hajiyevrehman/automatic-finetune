#!/usr/bin/env python
"""
Command-line script to launch LLM training jobs on Lambda Labs
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cloud.lambda_labs import launch_training_job

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch LLM training on Lambda Labs")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/qwen_training_config.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Lambda Labs API key (or set LAMBDA_API_KEY env var)",
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key for Lambda Labs (or set LAMBDA_SSH_KEY env var)",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(project_root),
        help="Project directory to upload",
    )
    parser.add_argument(
        "--no-upload-data",
        action="store_true",
        help="Skip uploading datasets (useful if they're large and already on instance)",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        help="AWS profile to use for S3 access (or set AWS_PROFILE env var)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        logger.error(
            "Lambda Labs API key must be provided via --api-key or LAMBDA_API_KEY env var"
        )
        sys.exit(1)

    # Get SSH key path from args or environment
    ssh_key_path = args.ssh_key or os.environ.get("LAMBDA_SSH_KEY")
    if not ssh_key_path:
        logger.error(
            "SSH key path must be provided via --ssh-key or LAMBDA_SSH_KEY env var"
        )
        sys.exit(1)

    # Set AWS profile if provided
    aws_profile = args.aws_profile or os.environ.get("AWS_PROFILE")
    if aws_profile:
        logger.info(f"Using AWS profile: {aws_profile}")
        os.environ["AWS_PROFILE"] = aws_profile
    else:
        # Check for AWS credentials
        if not (
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        ):
            logger.warning(
                "AWS credentials not found in environment variables. S3 access may fail."
            )

    # Check if configuration file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Launch training job
    logger.info(f"Launching training job with configuration: {args.config}")

    # Check if data directories should be excluded from upload
    exclude_patterns = [".git", "__pycache__", ".env", "venv", "*.pyc", "*.log"]

    if args.no_upload_data:
        logger.info("Skipping upload of data directories")
        exclude_patterns.extend(["data/raw/*", "data/processed/*"])

    try:
        success = launch_training_job(
            args.config,
            api_key,
            ssh_key_path,
            args.project_dir,
        )

        if success:
            logger.info("Training job completed successfully")
        else:
            logger.error("Training job failed")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Error launching training job: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
