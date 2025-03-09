#!/usr/bin/env python3
"""
CLI for LLM fine-tuning on Lambda Labs cloud instances.
This enhances the existing fine-tuning CLI with cloud management features.
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from src.cloud.auth import get_cloud_credentials
    from scripts.setup_cloud_finetuning import LambdaCloudManager
except ImportError:
    # Handle relative imports when run as script
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.cloud.auth import get_cloud_credentials
    from scripts.setup_cloud_finetuning import LambdaCloudManager


class FineTuningWorkflow:
    """
    Manages the end-to-end workflow for fine-tuning LLMs on Lambda Cloud
    """
    
    def __init__(self, config_path=None):
        """Initialize with optional configuration path"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set up cloud credentials
        self._setup_cloud_credentials()
        
        # Initialize Lambda Cloud Manager if credentials available
        self.lambda_manager = None
        if os.environ.get('LAMBDA_API_KEY'):
            try:
                self.lambda_manager = LambdaCloudManager(config_path=config_path)
            except Exception as e:
                logger.warning(f"Failed to initialize Lambda Cloud Manager: {e}")
    
    def _load_config(self):
        """Load configuration from file"""
        if not self.config_path:
            default_paths = [
                "configs/training/llm_finetuning.yaml",
                "configs/training/default.yaml"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    self.config_path = path
                    break
            
            if not self.config_path:
                logger.warning("No configuration file found. Using default settings.")
                return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _setup_cloud_credentials(self):
        """Set up cloud credentials from environment or auth module"""
        # Try to get credentials using the auth module
        try:
            credentials = get_cloud_credentials()
            
            # Set AWS credentials if available
            if 'aws' in credentials:
                aws_creds = credentials['aws']
                if 'access_key_id' in aws_creds and 'secret_access_key' in aws_creds:
                    os.environ['AWS_ACCESS_KEY_ID'] = aws_creds['access_key_id']
                    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_creds['secret_access_key']
                    if 'region' in aws_creds:
                        os.environ['AWS_REGION'] = aws_creds['region']
            
            # Set Lambda credentials if available
            if 'lambda' in credentials and 'api_key' in credentials['lambda']:
                os.environ['LAMBDA_API_KEY'] = credentials['lambda']['api_key']
                
        except Exception as e:
            logger.debug(f"Could not load credentials from auth module: {e}")
            # Continue with environment variables if set
    
    def create_instance(self, args):
        """Create a new Lambda Cloud instance"""
        if not self.lambda_manager:
            logger.error("Lambda Cloud Manager not available. Check API key.")
            return False
        
        # Build config overrides from arguments
        config_override = {}
        if args.name:
            config_override['name'] = args.name
        if args.region:
            config_override['region_name'] = args.region
        if args.instance_type:
            config_override['instance_type_name'] = args.instance_type
        if args.ssh_key:
            config_override['ssh_key_names'] = [args.ssh_key]
        
        # Generate setup script with repo and requirements
        repo_url = args.repo or "https://github.com/Hajiyevrehman/automatic-finetune.git"
        user_data = self.lambda_manager.generate_setup_script(
            repo_url=repo_url,
            requirements="requirements.txt" if os.path.exists("requirements.txt") else None
        )
        config_override['user_data'] = user_data
        
        # Launch the instance
        logger.info("Launching Lambda Cloud instance...")
        instance_id = self.lambda_manager.launch_instance(config_override)
        if not instance_id:
            logger.error("Failed to launch instance")
            return False
        
        try:
            logger.info(f"Waiting for instance {instance_id} to be ready...")
            ip_address = self.lambda_manager.wait_for_instance(instance_id)
            logger.info(f"Instance {instance_id} is ready at {ip_address}")
            
            # Save instance details to file
            instance_info = {
                'instance_id': instance_id,
                'ip_address': ip_address,
                'launched_at': time.time(),
                'config': self.lambda_manager.config
            }
            
            with open('.lambda_instance.json', 'w') as f:
                json.dump(instance_info, f, indent=2)
            
            logger.info(f"Instance details saved to .lambda_instance.json")
            return True
            
        except Exception as e:
            logger.error(f"Error waiting for instance: {e}")
            return False
    
    def prepare_training(self, args):
        """Prepare the instance for training by uploading files"""
        # Check if we have instance info
        instance_info = self._get_instance_info(args.instance_id)
        if not instance_info:
            return False
        
        ip_address = instance_info.get('ip_address')
        if not ip_address:
            logger.error(f"No IP address found for instance {args.instance_id}")
            return False
        
        # Build scp commands to upload necessary files
        import subprocess
        
        # Determine SSH key path
        key_path = args.key_path
        if not key_path:
            # Try to find a key in standard locations
            for possible_key in ['~/.ssh/id_rsa', '~/.ssh/id_ed25519']:
                expanded = os.path.expanduser(possible_key)
                if os.path.exists(expanded):
                    key_path = expanded
                    break
        
        if not key_path:
            logger.warning("No SSH key found. SCP may fail if key authentication is required.")
        
        # Files to upload
        files_to_upload = [
            self.config_path
        ]
        
        # Additional data files if specified
        if args.data_files:
            files_to_upload.extend(args.data_files.split(','))
        
        # Upload each file
        for file_path in files_to_upload:
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist, skipping")
                continue
                
            # Build scp command
            scp_cmd = ['scp']
            if key_path:
                scp_cmd.extend(['-i', key_path])
            
            # Determine remote path - maintain directory structure under ~/automatic-finetune
            remote_path = f"ubuntu@{ip_address}:~/automatic-finetune/{file_path}"
            scp_cmd.extend([file_path, remote_path])
            
            logger.info(f"Uploading {file_path} to instance...")
            try:
                subprocess.run(scp_cmd, check=True)
                logger.info(f"Successfully uploaded {file_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to upload {file_path}: {e}")
                return False
        
        logger.info("All files uploaded successfully")
        return True
    
    def run_training(self, args):
        """Run the training on the instance"""
        # Check if we have instance info
        instance_info = self._get_instance_info(args.instance_id)
        if not instance_info:
            return False
            
        ip_address = instance_info.get('ip_address')
        if not ip_address:
            logger.error(f"No IP address found for instance {args.instance_id}")
            return False
        
        # Determine SSH key path
        key_path = args.key_path
        if not key_path:
            # Try to find a key in standard locations
            for possible_key in ['~/.ssh/id_rsa', '~/.ssh/id_ed25519']:
                expanded = os.path.expanduser(possible_key)
                if os.path.exists(expanded):
                    key_path = expanded
                    break
        
        # Build SSH command for running training
        import subprocess
        ssh_cmd = ['ssh']
        if key_path:
            ssh_cmd.extend(['-i', key_path])
        
        # Support for running in tmux to prevent disconnection from killing job
        use_tmux = args.use_tmux
        
        # Base training command
        base_cmd = f"cd ~/automatic-finetune && "
        
        # Set environment variables
        env_vars = ""
        for var in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION']:
            if os.environ.get(var):
                env_vars += f"{var}='{os.environ.get(var)}' "
        
        # Add environment variables to command
        if env_vars:
            base_cmd += env_vars + " "
        
        # Training script/command
        if args.script:
            train_cmd = f"python {args.script}"
        else:
            # Use default training script from your paste.txt
            train_cmd = f"python -c \"$(cat <<'EOT'\n{self._get_default_training_script()}\nEOT\n)\""
        
        # Add config path if provided
        if self.config_path:
            config_basename = os.path.basename(self.config_path)
            remote_config_path = f"~/automatic-finetune/{self.config_path}"
            
            # If we're running a custom script, add the config as an argument
            if args.script:
                train_cmd += f" --config {remote_config_path}"
        
        # Combine commands
        if use_tmux:
            # Run in a new tmux session that can be attached to later
            full_cmd = f"{base_cmd} tmux new-session -d -s training '{train_cmd}'"
            attach_cmd = f"ssh {'-i ' + key_path if key_path else ''} ubuntu@{ip_address} 'tmux attach-session -t training'"
        else:
            # Run directly
            full_cmd = f"{base_cmd}{train_cmd}"
            attach_cmd = None
        
        ssh_cmd.extend([f'ubuntu@{ip_address}', full_cmd])
        
        logger.info(f"Starting training on instance {args.instance_id} at {ip_address}")
        try:
            subprocess.run(ssh_cmd, check=True)
            
            if use_tmux:
                logger.info("Training started in tmux session 'training'")
                logger.info(f"To view the training progress, run:\n{attach_cmd}")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start training: {e}")
            return False
    
    def terminate_instance(self, args):
        """Terminate an instance"""
        if not self.lambda_manager:
            logger.error("Lambda Cloud Manager not available. Check API key.")
            return False
        
        logger.info(f"Terminating instance {args.instance_id}...")
        try:
            self.lambda_manager.terminate_instance(args.instance_id)
            logger.info(f"Instance {args.instance_id} termination request sent")
            
            # Remove instance info file if it exists
            if os.path.exists('.lambda_instance.json'):
                stored_info = self._get_instance_info()
                if stored_info and stored_info.get('instance_id') == args.instance_id:
                    os.remove('.lambda_instance.json')
                    logger.info("Removed stored instance info")
            
            return True
        except Exception as e:
            logger.error(f"Failed to terminate instance: {e}")
            return False
    
    def list_instances(self, args):
        """List running instances"""
        if not self.lambda_manager:
            logger.error("Lambda Cloud Manager not available. Check API key.")
            return False
        
        try:
            instances = self.lambda_manager.list_instances()
            if not instances:
                logger.info("No instances found")
                return True
                
            print("\nRunning Instances:")
            print("-" * 80)
            print(f"{'ID':<24} {'Name':<30} {'Status':<10} {'IP Address':<15} {'Type':<15}")
            print("-" * 80)
            for instance in instances:
                print(f"{instance.get('id', 'N/A'):<24} {instance.get('name', 'N/A'):<30} "
                      f"{instance.get('status', 'N/A'):<10} {instance.get('ip', 'N/A'):<15} "
                      f"{instance.get('instance_type', {}).get('name', 'N/A'):<15}")
            print("-" * 80)
            return True
        except Exception as e:
            logger.error(f"Failed to list instances: {e}")
            return False

    def download_results(self, args):
        """Download training results from instance"""
        # Check if we have instance info
        instance_info = self._get_instance_info(args.instance_id)
        if not instance_info:
            return False
            
        ip_address = instance_info.get('ip_address')
        if not ip_address:
            logger.error(f"No IP address found for instance {args.instance_id}")
            return False
        
        # Determine SSH key path
        key_path = args.key_path
        if not key_path:
            # Try to find a key in standard locations
            for possible_key in ['~/.ssh/id_rsa', '~/.ssh/id_ed25519']:
                expanded = os.path.expanduser(possible_key)
                if os.path.exists(expanded):
                    key_path = expanded
                    break
        
        # Create local results directory
        results_dir = args.output_dir or "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Build scp command to download
        import subprocess
        scp_cmd = ['scp', '-r']
        if key_path:
            scp_cmd.extend(['-i', key_path])
        
        # Remote path to download from
        remote_path = f"ubuntu@{ip_address}:~/automatic-finetune/models/finetuned/*"
        
        # Build full command
        scp_cmd.extend([remote_path, results_dir])
        
        logger.info(f"Downloading training results from instance {args.instance_id}...")
        try:
            subprocess.run(scp_cmd, check=True)
            logger.info(f"Results downloaded to {results_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download results: {e}")
            return False
    
    def _get_instance_info(self, instance_id=None):
        """Get instance info from file or Lambda API"""
        # Try to get from file first
        try:
            if os.path.exists('.lambda_instance.json'):
                with open('.lambda_instance.json', 'r') as f:
                    instance_info = json.load(f)
                
                # If instance_id is provided, verify it matches
                if instance_id and instance_info.get('instance_id') != instance_id:
                    # Try to get info from Lambda API
                    if self.lambda_manager:
                        ip_address = self.lambda_manager.get_instance_ip(instance_id)
                        if ip_address:
                            return {'instance_id': instance_id, 'ip_address': ip_address}
                    return None
                
                return instance_info
        except Exception:
            pass
            
        # If not found in file but we have instance_id, try Lambda API
        if instance_id and self.lambda_manager:
            ip_address = self.lambda_manager.get_instance_ip(instance_id)
            if ip_address:
                return {'instance_id': instance_id, 'ip_address': ip_address}
        
        logger.error("No instance information found. Run list-instances to see available instances.")
        return None
    
    def _get_default_training_script(self):
        """Get the default training script (from your paste.txt)"""
        # For brevity, we'll return a placeholder that imports the full code
        # from the repository instead of embedding the entire script
        return """
import os
import sys
import yaml

# Load training config
config_path = "configs/training/llm_finetuning.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Import and run Unsloth fine-tuning
print("Starting LLM fine-tuning with Unsloth...")

# Import unsloth and required libraries  
import unsloth
from unsloth import FastLanguageModel
import torch
import mlflow

# Run the main training function - this executes the full pipeline
# similar to your pasted code but imports from the repo
print("Importing from repository...")
from src.finetuning.unsloth_trainer import train_model_with_unsloth

# Run the training
print("Starting training...")
train_model_with_unsloth(config)
"""


def main():
    """CLI entry point for fine-tuning workflow"""
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Workflow on Lambda Cloud")
    
    # Global arguments
    parser.add_argument('--config', '-c', help="Path to configuration file")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose logging")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create instance command
    create_parser = subparsers.add_parser('create-instance', help='Create a new Lambda Cloud instance')
    create_parser.add_argument('--name', help='Instance name')
    create_parser.add_argument('--region', help='Region name')
    create_parser.add_argument('--instance-type', help='Instance type')
    create_parser.add_argument('--ssh-key', help='SSH key name')
    create_parser.add_argument('--repo', help='Git repository to clone (default: your repo)')
    create_parser.set_defaults(func=lambda w, a: w.create_instance(a))
    
    # List instances command
    list_parser = subparsers.add_parser('list-instances', help='List running Lambda Cloud instances')
    list_parser.set_defaults(func=lambda w, a: w.list_instances(a))
    
    # Prepare training command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare instance for training')
    prepare_parser.add_argument('instance_id', help='Instance ID')
    prepare_parser.add_argument('--key-path', help='Path to SSH private key')
    prepare_parser.add_argument('--data-files', help='Comma-separated list of data files to upload')
    prepare_parser.set_defaults(func=lambda w, a: w.prepare_training(a))
    
    # Run training command
    train_parser = subparsers.add_parser('train', help='Run training on instance')
    train_parser.add_argument('instance_id', help='Instance ID')
    train_parser.add_argument('--key-path', help='Path to SSH private key')
    train_parser.add_argument('--script', help='Custom training script path (relative to repo root)')
    train_parser.add_argument('--use-tmux', action='store_true', help='Run in tmux session to prevent disconnect')
    train_parser.set_defaults(func=lambda w, a: w.run_training(a))
    
    # Download results command
    download_parser = subparsers.add_parser('download', help='Download training results')
    download_parser.add_argument('instance_id', help='Instance ID')
    download_parser.add_argument('--key-path', help='Path to SSH private key')
    download_parser.add_argument('--output-dir', help='Local directory to save results')
    download_parser.set_defaults(func=lambda w, a: w.download_results(a))
    
    # Terminate instance command
    terminate_parser = subparsers.add_parser('terminate', help='Terminate an instance')
    terminate_parser.add_argument('instance_id', help='Instance ID to terminate')
    terminate_parser.set_defaults(func=lambda w, a: w.terminate_instance(a))
    
    # All-in-one workflow
    workflow_parser = subparsers.add_parser('run-workflow', help='Run end-to-end fine-tuning workflow')
    workflow_parser.add_argument('--instance-type', default='gpu_1x_a10', help='Instance type')
    workflow_parser.add_argument('--key-path', help='Path to SSH private key')
    workflow_parser.add_argument('--keep-instance', action='store_true', help='Do not terminate instance after training')
    workflow_parser.add_argument('--output-dir', default='results', help='Directory for downloading results')
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize workflow
    workflow = FineTuningWorkflow(config_path=args.config)
    
    # Run appropriate function based on command
    if args.command == 'run-workflow':
        # Create instance
        create_args = argparse.Namespace(
            name=f"LLM-Finetune-{int(time.time())}",
            region=None,
            instance_type=args.instance_type,
            ssh_key=None,
            repo=None
        )
        logger.info("Step 1: Creating Lambda Cloud instance...")
        if not workflow.create_instance(create_args):
            return 1
        
        # Get instance ID from stored info
        with open('.lambda_instance.json', 'r') as f:
            instance_info = json.load(f)
            instance_id = instance_info['instance_id']
        
        # Wait for instance to be fully initialized
        logger.info("Waiting 60 seconds for instance initialization...")
        time.sleep(60)
        
        # Prepare instance
        prepare_args = argparse.Namespace(
            instance_id=instance_id,
            key_path=args.key_path,
            data_files=None
        )
        logger.info("Step 2: Preparing instance for training...")
        if not workflow.prepare_training(prepare_args):
            return 1
        
        # Run training
        train_args = argparse.Namespace(
            instance_id=instance_id,
            key_path=args.key_path,
            script=None,
            use_tmux=True
        )
        logger.info("Step 3: Running training...")
        if not workflow.run_training(train_args):
            return 1
            
        # Wait for completion (simple approach - more sophisticated monitoring could be added)
        logger.info("Training is running in tmux session. Waiting 10 minutes before downloading results...")
        logger.info(f"You can check progress by running: ssh ubuntu@{instance_info['ip_address']} 'tmux attach -t training'")
        time.sleep(600)  # Wait 10 minutes - this could be much longer for real training
        
        # Download results
        download_args = argparse.Namespace(
            instance_id=instance_id,
            key_path=args.key_path,
            output_dir=args.output_dir
        )
        logger.info("Step 4: Downloading training results...")
        workflow.download_results(download_args)
        
        # Terminate instance if not keeping
        if not args.keep_instance:
            terminate_args = argparse.Namespace(
                instance_id=instance_id
            )
            logger.info("Step 5: Terminating instance...")
            workflow.terminate_instance(terminate_args)
        else:
            logger.info(f"Instance {instance_id} kept running as requested.")
        
        logger.info("Workflow completed successfully!")
        
    elif args.command and hasattr(args, 'func'):
        # Call the appropriate function with the workflow and args
        if not args.func(workflow, args):
            return 1
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())