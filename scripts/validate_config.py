import argparse
from pathlib import Path

import yaml


def validate_config(config_path):
    """Basic validation of config files"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Perform basic validation
        if config_path.name == "model_config.yaml":
            assert "model" in config, "Missing 'model' section"
            assert "base_model" in config["model"], "Missing 'base_model' field"

        elif config_path.name == "training_config.yaml":
            assert "training" in config, "Missing 'training' section"
            assert "batch_size" in config["training"], "Missing 'batch_size' field"
            assert (
                "learning_rate" in config["training"]
            ), "Missing 'learning_rate' field"

        print(f"✅ Configuration file {config_path} is valid")
        return True

    except Exception as e:
        print(f"❌ Error validating {config_path}: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate configuration files")
    parser.add_argument("config_path", help="Path to configuration file")
    args = parser.parse_args()

    validate_config(Path(args.config_path))
