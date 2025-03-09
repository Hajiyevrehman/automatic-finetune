#!/usr/bin/env python3
"""
Setup script for configuring the environment for LLM fine-tuning on Lambda Labs cloud.
This script handles setting up credentials and installing required packages.
"""

import os
import sys
import argparse
import subprocess
import getpass
import yaml
import json
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    requirements = [
        "pyyaml",
        "requests",
        "boto3",
    ]
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + requirements, check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_credentials(args):
    """Setup cloud credentials"""
    print("\nSetting up cloud credentials...")
    
    # Create credentials directory
    creds_dir = os.path.expanduser("~/.cloud-finetuning")
    os.makedirs(creds_dir, exist_ok=True)
    creds_path = os.path.join(creds_dir, "credentials.yaml")
    
    # Check if credentials file exists
    creds = {}
    if os.path.exists(creds_path):
        try:
            with open(creds_path, 'r') as f:
                creds = yaml.safe_load(f) or {}
                print("Found existing credentials file")
        except Exception as e:
            print(f"Error reading existing credentials: {e}")
            creds = {}
    
    # Initialize credentials sections if needed
    if 'aws' not in creds:
        creds['aws'] = {}
    if 'lambda' not in creds:
        creds['lambda'] = {}
    
    # Get Lambda API key
    if args.lambda_api_key:
        creds['lambda']['api_key'] = args.lambda_api_key
    elif 'api_key' not in creds['lambda'] or not creds['lambda']['api_key']:
        api_key = getpass.getpass("Enter Lambda Labs API Key (or press Enter to skip): ")
        if api_key:
            creds['lambda']['api_key'] = api_key
    
    # Get AWS credentials
    if args.aws_access_key:
        creds['aws']['access_key_id'] = args.aws_access_key
    elif 'access_key_id' not in creds['aws'] or not creds['aws']['access_key_id']:
        access_key = getpass.getpass("Enter AWS Access Key ID (or press Enter to skip): ")
        if access_key:
            creds['aws']['access_key_id'] = access_key
    
    if args.aws_secret_key:
        creds['aws']['secret_access_key'] = args.aws_secret_key
    elif 'secret_access_key' not in creds['aws'] or not creds['aws']['secret_access_key']:
        secret_key = getpass.getpass("Enter AWS Secret Access Key (or press Enter to skip): ")
        if secret_key:
            creds['aws']['secret_access_key'] = secret_key
    
    if args.aws_region:
        creds['aws']['region'] = args.aws_region
    elif 'region' not in creds['aws'] or not creds['aws']['region']:
        region = input("Enter AWS Region (default: us-east-1): ").strip() or "us-east-1"
        creds['aws']['region'] = region
    
    # Save credentials
    try:
        with open(creds_path, 'w') as f:
            yaml.dump(creds, f)
        
        # Set permissions to user only
        os.chmod(creds_path, 0o600)
        print(f"✅ Credentials saved to {creds_path}")
        
        # Also set environment variables for current session
        if creds.get('aws', {}).get('access_key_id'):
            os.environ['AWS_ACCESS_KEY_ID'] = creds['aws']['access_key_id']
        if creds.get('aws', {}).get('secret_access_key'):
            os.environ['AWS_SECRET_ACCESS_KEY'] = creds['aws']['secret_access_key']
        if creds.get('aws', {}).get('region'):
            os.environ['AWS_REGION'] = creds['aws']['region']
        if creds.get('lambda', {}).get('api_key'):
            os.environ['LAMBDA_API_KEY'] = creds['lambda']['api_key']
            
        return True
    except Exception as e:
        print(f"❌ Error saving credentials: {e}")
        return False

def setup_config_files(args):
    """Setup configuration files if they don't exist"""
    print("\nChecking configuration files...")
    
    # Check if configs directory exists
    configs_dir = Path("configs")
    if not configs_dir.exists():
        print("Creating configs directory...")
        os.makedirs(configs_dir / "training", exist_ok=True)
        os.makedirs(configs_dir / "data", exist_ok=True)
    
    # Check if training config exists
    training_config_path = configs_dir / "training" / "llm_finetuning.yaml"
    if not training_config_path.exists() or args.overwrite:
        print("Creating default training configuration...")
        
        # Use the config from the provided file
        training_yaml = """# LLM Finetuning Configuration
# Model configuration
model:
  name: "unsloth/Qwen2.5-3B" # Model to finetune
  max_seq_length: 2048 # Maximum sequence length
  load_in_4bit: true # Whether to use 4-bit quantization

# LoRA configuration
lora:
  r: 16 # LoRA rank
  alpha: 16 # LoRA alpha
  dropout: 0 # LoRA dropout
  target_modules: # Modules to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  use_rslora: false # Whether to use rank-stabilized LoRA

# Training configuration
training:
  batch_size: 16 # Batch size per device
  eval_batch_size: 4 # Evaluation batch size per device
  gradient_accumulation_steps: 8 # Gradient accumulation steps
  num_train_epochs: 3 # Number of training epochs
  learning_rate: 0.0002 # Learning rate
  warmup_steps: 5 # Warmup steps
  weight_decay: 0.01 # Weight decay
  lr_scheduler_type: "linear" # Learning rate scheduler
  seed: 3407 # Random seed
  logging_steps: 5 # Log metrics every X steps
  logging_strategy: "steps" # Log based on steps
  evaluation_strategy: "steps" # Evaluate based on steps
  eval_steps: 20 # Run evaluation every X steps
  save_strategy: "steps" # Save based on steps
  save_steps: 50 # Save every X steps
  save_total_limit: 5 # Maximum number of saved checkpoints
  optim: "adamw_8bit" # Optimizer
  validation_split: 0.1 # Validation split ratio
  test_mode: True # Whether to run in test mode (using only a fraction of the data)
  test_fraction: 0.2 # Fraction of data to use in test mode

# Data configuration
data:
  train_file: "data/processed/servicenow-qa_converted.json" # Training data file
  format_type: "chatml" # Data format type (chatml, alpaca, etc.)

# Output configuration
output:
  dir: "models/finetuned" # Output directory
  save_formats: # Formats to save the model in
    - "lora" # LoRA adapters only
  push_to_hub: false # Whether to push to Hugging Face Hub
  hub_model_id: "" # Hugging Face Hub model ID
  hub_token: "" # Hugging Face Hub token

# S3 configuration
s3:
  bucket: "llm-finetuning-rahman-1234" # S3 bucket name
  model_prefix: "models/qwen2.5-3b-servicenow-qa" # S3 model prefix

# MLflow configuration
mlflow:
  tracking_uri: "" # MLflow tracking URI (leave empty for local)
  run_name: "qwen2.5-3b-finetune" # MLflow run name

# Cloud configuration (for Lambda Labs)
cloud:
  lambda:
    region_name: "europe-central-1"
    instance_type_name: "gpu_1x_a10"
    name: "LLM-Finetuning"
"""
        
        with open(training_config_path, "w") as f:
            f.write(training_yaml)
        print(f"✅ Created {training_config_path}")
    else:
        print(f"✓ Training config already exists at {training_config_path}")
    
    # Check if data config exists
    data_config_path = configs_dir / "data" / "data_processing.yaml"
    if not data_config_path.exists() or args.overwrite:
        print("Creating default data configuration...")
        
        data_yaml = """# Data Processing Configuration
# S3 configuration
s3:
  default_bucket: "llm-finetuning-rahman-1234" # Default S3 bucket
  region: "us-east-1" # AWS region
"""
        
        with open(data_config_path, "w") as f:
            f.write(data_yaml)
        print(f"✅ Created {data_config_path}")
    else:
        print(f"✓ Data config already exists at {data_config_path}")
    
    return True

def create_auth_module(args):
    """Create or update the auth module to load credentials"""
    print("\nSetting up auth module...")
    
    # Create src/cloud directory if it doesn't exist
    cloud_dir = Path("src/cloud")
    os.makedirs(cloud_dir, exist_ok=True)
    
    # Create auth.py file
    auth_file = cloud_dir / "auth.py"
    
    auth_code = """#!/usr/bin/env python3
\"\"\"
Authentication module for cloud services.
Handles loading credentials from environment variables or config files.
\"\"\"

import os
import yaml
from pathlib import Path

def get_cloud_credentials():
    \"\"\"
    Get cloud credentials from environment variables or config file.
    
    Returns:
        dict: Dictionary containing credentials for different cloud providers
    \"\"\"
    # Initialize credentials dictionary
    credentials = {
        'aws': {},
        'lambda': {}
    }
    
    # Check environment variables first
    if os.environ.get('AWS_ACCESS_KEY_ID'):
        credentials['aws']['access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID')
    
    if os.environ.get('AWS_SECRET_ACCESS_KEY'):
        credentials['aws']['secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    if os.environ.get('AWS_REGION'):
        credentials['aws']['region'] = os.environ.get('AWS_REGION')
    
    if os.environ.get('LAMBDA_API_KEY'):
        credentials['lambda']['api_key'] = os.environ.get('LAMBDA_API_KEY')
    
    # If any credentials missing, try to load from config file
    creds_file = Path(os.path.expanduser("~/.cloud-finetuning/credentials.yaml"))
    if creds_file.exists():
        try:
            with open(creds_file, 'r') as f:
                file_creds = yaml.safe_load(f) or {}
            
            # Update AWS credentials if needed
            if 'aws' in file_creds:
                for key, value in file_creds['aws'].items():
                    if key not in credentials['aws'] or not credentials['aws'][key]:
                        credentials['aws'][key] = value
            
            # Update Lambda credentials if needed
            if 'lambda' in file_creds:
                for key, value in file_creds['lambda'].items():
                    if key not in credentials['lambda'] or not credentials['lambda'][key]:
                        credentials['lambda'][key] = value
                        
        except Exception as e:
            print(f"Error loading credentials file: {e}")
    
    return credentials

def update_env_from_credentials():
    \"\"\"
    Update environment variables from credentials file.
    \"\"\"
    credentials = get_cloud_credentials()
    
    # Set AWS credentials in environment
    if credentials.get('aws', {}).get('access_key_id'):
        os.environ['AWS_ACCESS_KEY_ID'] = credentials['aws']['access_key_id']
    
    if credentials.get('aws', {}).get('secret_access_key'):
        os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['aws']['secret_access_key']
    
    if credentials.get('aws', {}).get('region'):
        os.environ['AWS_REGION'] = credentials['aws']['region']
    
    # Set Lambda credentials in environment
    if credentials.get('lambda', {}).get('api_key'):
        os.environ['LAMBDA_API_KEY'] = credentials['lambda']['api_key']

if __name__ == "__main__":
    # Simple test if run directly
    creds = get_cloud_credentials()
    print("AWS Access Key ID:", "****" + creds['aws'].get('access_key_id', '')[-4:] if creds['aws'].get('access_key_id') else "Not found")
    print("AWS Secret Access Key:", "****" if creds['aws'].get('secret_access_key') else "Not found")
    print("AWS Region:", creds['aws'].get('region', 'Not found'))
    print("Lambda API Key:", "****" + creds['lambda'].get('api_key', '')[-4:] if creds['lambda'].get('api_key') else "Not found")
"""
    
    with open(auth_file, "w") as f:
        f.write(auth_code)
    
    # Create empty __init__.py file if it doesn't exist
    init_file = cloud_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, "w") as f:
            f.write("# Cloud modules package\n")
    
    print(f"✅ Created auth module at {auth_file}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Setup environment for LLM fine-tuning on Lambda Labs cloud")
    
    parser.add_argument('--lambda-api-key', help="Lambda Labs API key")
    parser.add_argument('--aws-access-key', help="AWS access key ID")
    parser.add_argument('--aws-secret-key', help="AWS secret access key")
    parser.add_argument('--aws-region', help="AWS region")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing config files")
    parser.add_argument('--skip-installs', action='store_true', help="Skip installing packages")
    
    args = parser.parse_args()
    
    print("=== LLM Fine-tuning Cloud Setup ===\n")
    
    # Install requirements
    if not args.skip_installs:
        install_requirements()
    
    # Setup credentials
    setup_credentials(args)
    
    # Setup config files
    setup_config_files(args)
    
    # Create auth module
    create_auth_module(args)
    
    print("\n=== Setup Complete! ===")
    print("You can now use the following commands to manage your fine-tuning workflow:")
    print("  python -m src.cli.finetuning_cli list-instances")
    print("  python -m src.cli.finetuning_cli create-instance")
    print("  python -m src.cli.finetuning_cli train <instance_id>")
    print("  python -m src.cli.finetuning_cli download <instance_id>")
    print("  python -m src.cli.finetuning_cli terminate <instance_id>")
    print("\nOr run the complete workflow:")
    print("  python -m src.cli.finetuning_cli run-workflow")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())