#!/usr/bin/env python3
"""
Run the Unsloth training script for fine-tuning LLMs.
"""
import os
import sys
import yaml
import argparse
from pathlib import Path

def load_env_from_file(env_path):
    """Load environment variables from a .env file"""
    if not os.path.exists(env_path):
        return False
    
    print(f"Loading environment variables from {env_path}")
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run Unsloth training")
    parser.add_argument('--config', default="configs/training/llm_finetuning.yaml", 
                        help="Path to config file")
    parser.add_argument('--env-file', default="scripts/.env",
                        help="Path to .env file with AWS credentials")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    env_paths = [
        args.env_file,
        ".env",
        "scripts/.env"
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if load_env_from_file(env_path):
            env_loaded = True
            break
    
    if not env_loaded:
        print("Warning: No .env file found. AWS credentials may be missing.")
    
    # Check AWS credentials
    if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("AWS credentials loaded from environment")
    else:
        print("AWS credentials not found. S3 operations may fail.")
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        default_configs = [
            Path("configs/training/llm_finetuning.yaml"),
            Path("configs/training/default.yaml")
        ]
        
        for cfg_path in default_configs:
            if cfg_path.exists():
                config_path = cfg_path
                print(f"Using default config at {config_path}")
                break
        else:
            print("No default config found. Exiting.")
            sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import and run the training function
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.finetuning.unsloth_trainer import train_model_with_unsloth
    
    # Run the training
    train_model_with_unsloth(config)

if __name__ == "__main__":
    main()
