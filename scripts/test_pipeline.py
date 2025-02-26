#!/usr/bin/env python
# scripts/test_pipeline.py

import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_file_exists(file_path, min_size_kb=1):
    """Check if a file exists and has a minimum size."""
    path = Path(file_path)

    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    size_kb = path.stat().st_size / 1024
    if size_kb < min_size_kb:
        logger.error(f"File too small: {file_path} ({size_kb:.2f} KB)")
        return False

    logger.info(f"File validation successful: {file_path} ({size_kb:.2f} KB)")
    return True


def run_command(command):
    """Run a shell command and log output."""
    logger.info(f"Running command: {command}")

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        logger.info(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_json_content(file_path, expected_keys):
    """Check if a JSON file contains expected keys in each item."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            logger.error(f"JSON file {file_path} is not a non-empty list")
            return False

        # Check first item for expected keys
        first_item = data[0]
        missing_keys = [key for key in expected_keys if key not in first_item]

        if missing_keys:
            logger.error(f"Missing expected keys in {file_path}: {missing_keys}")
            return False

        logger.info(
            f"JSON content validation successful: {file_path} ({len(data)} items)"
        )
        return True
    except Exception as e:
        logger.error(f"Error checking JSON content in {file_path}: {e}")
        return False


def test_pipeline():
    """Test the entire data processing pipeline."""
    logger.info("Starting pipeline test")

    # Create required directories
    for dir_path in ["data/raw", "data/processed", "data/validation", "configs/data"]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create data processing config if it doesn't exist
    config_path = "configs/data/data_processing.json"
    if not Path(config_path).exists():
        config = {
            "paths": {
                "raw_data": "data/raw",
                "processed_data": "data/processed",
                "validation_data": "data/validation",
            },
            "preprocessing": {
                "text_normalization": {
                    "lowercase": False,
                    "remove_special_chars": False,
                }
            },
            "tokenization": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "max_length": 512,
                "truncation": True,
                "padding": "max_length",
            },
            "validation": {
                "min_question_length": 5,
                "max_question_length": 1000,
                "min_answer_length": 5,
                "max_answer_length": 10000,
            },
            "splits": {
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "shuffle": True,
            },
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Created config file: {config_path}")

    # Test steps

    # 1. Check if servicenow_qa.json exists or run the converter
    if not check_file_exists("data/raw/servicenow_qa.json"):
        logger.info("servicenow_qa.json not found, running converter")
        success = run_command(
            "python -m scripts.dataset_converter --subset_size 100 --version v1 --output_path data/raw/servicenow_qa.json"
        )
        if not success:
            logger.error("Failed to convert dataset")
            return False

        # Verify the output
        if not check_file_exists("data/raw/servicenow_qa.json"):
            logger.error("Converter did not produce the expected output")
            return False

    # 2. Check if we're in a Git repository
    git_initialized = Path(".git").is_dir()

    # 3. Run the DVC pipeline
    logger.info("Running DVC pipeline")

    # Check if DVC is initialized
    if not Path(".dvc").exists():
        # Initialize DVC with --no-scm if not in a Git repo
        if git_initialized:
            success = run_command("dvc init")
        else:
            logger.warning(
                "Not in a Git repository. Initializing DVC with --no-scm flag"
            )
            success = run_command("dvc init --no-scm")

        if not success:
            logger.error("Failed to initialize DVC")
            return False

    # Create a simple dvc.yaml if it doesn't exist
    if not Path("dvc.yaml").exists():
        logger.info("Creating a basic dvc.yaml file")
        dvc_yaml = """
stages:
  preprocess:
    cmd: python -m src.data.preprocess data/raw/servicenow_qa.json data/processed/servicenow_preprocessed.json
    deps:
      - src/data/preprocess.py
      - data/raw/servicenow_qa.json
    outs:
      - data/processed/servicenow_preprocessed.json

  validate:
    cmd: python -m src.data.validate data/processed/servicenow_preprocessed.json --output data/validation/servicenow_valid.json
    deps:
      - src/data/validate.py
      - data/processed/servicenow_preprocessed.json
    outs:
      - data/validation/servicenow_valid.json

  tokenize:
    cmd: python -m src.data.tokenize data/validation/servicenow_valid.json data/processed/servicenow_tokenized.json --config configs/data/data_processing.json --model Qwen/Qwen2.5-7B-Instruct
    deps:
      - src/data/tokenize.py
      - data/validation/servicenow_valid.json
      - configs/data/data_processing.json
    outs:
      - data/processed/servicenow_tokenized.json
        """
        with open("dvc.yaml", "w") as f:
            f.write(dvc_yaml)

    # Add the data to DVC tracking
    run_command("dvc add data/raw/servicenow_qa.json")

    # Run individual commands instead of the full pipeline at first
    logger.info("Running preprocessing step")
    success = run_command(
        "python -m src.data.preprocess data/raw/servicenow_qa.json data/processed/servicenow_preprocessed.json"
    )
    if not success:
        logger.error("Preprocessing step failed")
        return False

    logger.info("Running validation step")
    success = run_command(
        "python -m src.data.validate data/processed/servicenow_preprocessed.json --output data/validation/servicenow_valid.json"
    )
    if not success:
        logger.error("Validation step failed")
        return False

    logger.info("Running tokenization step")
    success = run_command(
        "python -m src.data.tokenize data/validation/servicenow_valid.json data/processed/servicenow_tokenized.json --config configs/data/data_processing.json --model Qwen/Qwen2.5-7B-Instruct"
    )
    if not success:
        logger.error("Tokenization step failed")
        return False

    # Try to run DVC pipeline only if previous steps succeeded
    logger.info("Attempting to run DVC pipeline")
    run_command(
        "dvc repro"
    )  # We don't fail if this doesn't work, since we've already run the commands directly

    # 4. Verify the outputs
    logger.info("Verifying pipeline outputs")

    # Check preprocessed data
    preprocessed_path = "data/processed/servicenow_preprocessed.json"
    if not check_file_exists(preprocessed_path):
        logger.error("Preprocessing step did not produce output file")
        return False

    # Check validated data
    validated_path = "data/validation/servicenow_valid.json"
    if not check_file_exists(validated_path):
        logger.error("Validation step did not produce output file")
        return False

    # Check tokenized data
    tokenized_path = "data/processed/servicenow_tokenized.json"
    if not check_file_exists(tokenized_path):
        logger.error("Tokenization step did not produce output file")
        return False

    logger.info("Pipeline test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_pipeline()
    if not success:
        logger.error("Pipeline test failed")
        exit(1)
