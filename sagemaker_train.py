#!/usr/bin/env python
"""
Training script for AWS SageMaker.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import your training modules
from src.training.trainer import ModelTrainer


def parse_args():
    """Parse arguments passed to the script."""
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )

    # Training specific arguments
    parser.add_argument(
        "--config", type=str, default="configs/training/cloud_training_config.yaml"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("S3_BUCKET", "llm-finetuning-rahman-1234"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)

    return parser.parse_args()


def main():
    """Main training function for SageMaker."""
    args = parse_args()

    # Create dirs
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Update config with SageMaker specific paths
    config["output"]["output_dir"] = args.model_dir
    config["training"]["num_train_epochs"] = args.epochs
    config["training"]["per_device_train_batch_size"] = args.batch_size

    # Save updated config
    updated_config_path = os.path.join(args.output_data_dir, "training_config.yaml")
    with open(updated_config_path, "w") as f:
        yaml.dump(config, f)

    # Initialize and run trainer
    trainer = ModelTrainer(config_path=updated_config_path)
    result = trainer.train()

    # Save metrics to output
    metrics_file = os.path.join(args.output_data_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(result.metrics, f)

    # Copy model to proper location for SageMaker
    logger.info(f"Training completed. Model saved to {args.model_dir}")
    return


if __name__ == "__main__":
    main()
