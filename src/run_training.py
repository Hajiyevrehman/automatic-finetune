#!/usr/bin/env python3
"""
Run the Unsloth training script for fine-tuning LLMs.
"""
import os
import sys
import yaml
import argparse

# Make sure the current directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Run Unsloth training")
    parser.add_argument('--config', default="configs/training/llm_finetuning.yaml", 
                        help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up AWS credentials if provided as environment variables
    if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("Using AWS credentials from environment variables")
    
    # Import and run the training function
    from src.finetuning.unsloth_trainer import train_model_with_unsloth
    
    # Run the training
    train_model_with_unsloth(config)

if __name__ == "__main__":
    main()
