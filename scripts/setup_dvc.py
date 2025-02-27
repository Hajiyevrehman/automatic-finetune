#!/usr/bin/env python3
"""
Configure DVC with S3 remote storage.

This script:
1. Sets up DVC remote storage pointing to an S3 bucket
2. Configures DVC to use AWS credentials from environment variables
3. Provides options for initializing a new DVC project or updating an existing one

Usage:
    python scripts/setup_dvc_s3.py --bucket my-llm-data-bucket [--init] [--no-scm]
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

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

    # Configure to use AWS CLI profiles if available
    profile_cmd = ["dvc", "remote", "modify", "storage", "profile", "default"]
    if not run_command(profile_cmd, check=False):
        # Not critical if this fails
        pass

    logger.info(f"DVC configured with S3 remote at {s3_url}")
    return True


def configure_s3_credentials():
    """
    Configure DVC to use AWS credentials from environment variables.

    Returns:
        bool: True if configuration succeeded, False otherwise
    """
    # Check if AWS credentials are available in environment
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    ):
        logger.warning(
            "AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not found in environment. "
            "Make sure these are set when using DVC with S3."
        )

    # Configure DVC to use environment variables
    cmds = [
        ["dvc", "remote", "modify", "storage", "credentialpath", ""],
        ["dvc", "remote", "modify", "storage", "use_ssl", "true"],
    ]

    for cmd in cmds:
        if not run_command(cmd, check=False):
            # Not critical if these fail
            pass

    return True


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
    """Main function to set up DVC with S3 remote."""
    parser = argparse.ArgumentParser(description="Configure DVC with S3 remote storage")
    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name for DVC remote storage"
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

    # Configure credentials
    configure_s3_credentials()

    # Create gitignore entries
    if not args.no_scm:
        create_gitignore_entries()

    logger.info(
        "DVC setup with S3 remote complete. Use 'dvc push' to upload data, "
        "'dvc pull' to download data, and 'dvc repro' to run the pipeline."
    )
    return 0


if __name__ == "__main__":
    exit(main())
