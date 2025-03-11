#!/usr/bin/env python3
"""
Lambda Cloud Instance Manager for LLM Fine-tuning
Handles launching, configuring, and connecting to Lambda Labs instances.
"""

import os
import sys
import json
import time
import argparse
import requests
import subprocess
import yaml
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "region_name": "europe-central-1",
    "instance_type_name": "gpu_1x_a10",  # Cheapest A10 instance
    "ssh_key_names": [],  # Will be populated from user config
    "file_system_names": [],  # Optional file systems
    "name": "LLM-Finetuning-Instance",
    "image": {"id": "ea87a5ee-5257-4286-aeb0-8607525801e2"},  # Lambda ML stack
    "user_data": ""  # Bootstrap script can be added here
}

class LambdaCloudManager:
    """Manages Lambda Cloud instances for LLM fine-tuning"""
    
    def __init__(self, api_key=None, config_path=None):
        """Initialize with API key and optional config"""
        self.api_key = api_key or os.environ.get('LAMBDA_API_KEY')
        if not self.api_key:
            raise ValueError("Lambda API key not found. Set LAMBDA_API_KEY environment variable or pass as argument.")
        
        self.base_url = "https://cloud.lambdalabs.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Load config if provided
        self.config = DEFAULT_CONFIG.copy()
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path):
        """Load instance configuration from YAML file"""
        with open(config_path, 'r') as f:
            cloud_config = yaml.safe_load(f)
        
        # If cloud section exists, update config
        if 'cloud' in cloud_config:
            lambda_config = cloud_config.get('cloud', {}).get('lambda', {})
            if lambda_config:
                for key, value in lambda_config.items():
                    if key in self.config:
                        self.config[key] = value
        
        print(f"Loaded Lambda config: {self.config}")
    
    def list_ssh_keys(self):
        """List available SSH keys"""
        response = requests.get(f"{self.base_url}/ssh-keys", headers=self.headers)
        response.raise_for_status()
        return response.json().get('data', [])
    
    def list_instances(self):
        """List running instances"""
        response = requests.get(f"{self.base_url}/instances", headers=self.headers)
        response.raise_for_status()
        return response.json().get('data', [])
    
    def get_instance_types(self):
        """Get available instance types and pricing"""
        response = requests.get(f"{self.base_url}/instance-types", headers=self.headers)
        response.raise_for_status()
        return response.json().get('data', {})
    
    def launch_instance(self, config_override=None):
        """Launch a new instance with given configuration"""
        launch_config = self.config.copy()
        
        # Override with any provided config
        if config_override:
            launch_config.update(config_override)
        
        # Validate SSH keys exist
        if not launch_config.get('ssh_key_names'):
            ssh_keys = self.list_ssh_keys()
            if not ssh_keys:
                raise ValueError("No SSH keys found in your Lambda account. Add one via the Lambda console.")
            launch_config['ssh_key_names'] = [ssh_keys[0]['name']]
            print(f"Using SSH key: {launch_config['ssh_key_names'][0]}")
        
        response = requests.post(
            f"{self.base_url}/instance-operations/launch",
            headers=self.headers,
            json=launch_config
        )
        
        if response.status_code != 200:
            print(f"Error launching instance: {response.text}")
            return None
        
        instance_id = response.json().get('data', {}).get('instance_ids', [None])[0]
        if not instance_id:
            print("Failed to get instance ID from response")
            return None
        
        print(f"Instance launching with ID: {instance_id}")
        return instance_id
    
    def wait_for_instance(self, instance_id, timeout=300):
        """Wait for instance to be ready, with timeout in seconds"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            instances = self.list_instances()
            for instance in instances:
                if instance.get('id') == instance_id:
                    status = instance.get('status')
                    ip_address = instance.get('ip')
                    
                    print(f"Instance status: {status}, IP: {ip_address}")
                    
                    if status == 'active' and ip_address:
                        return ip_address
            
            print("Waiting for instance to be ready...")
            time.sleep(15)
        
        raise TimeoutError(f"Instance {instance_id} did not become ready within {timeout} seconds")
    
    def terminate_instance(self, instance_id):
        """Terminate an instance by ID"""
        response = requests.post(
            f"{self.base_url}/instance-operations/terminate",
            headers=self.headers,
            json={"instance_ids": [instance_id]}
        )
        response.raise_for_status()
        return response.json()
    
    def get_instance_ip(self, instance_id):
        """Get the IP address of an instance"""
        instances = self.list_instances()
        for instance in instances:
            if instance.get('id') == instance_id:
                return instance.get('ip')
        return None
    
    def generate_setup_script(self, repo_url=None, requirements=None):
        """Generate a setup script for the instance"""
        script = "#!/bin/bash\n\n"
        script += "# Auto-generated setup script for LLM fine-tuning\n\n"
        
        # System updates and common packages
        script += "# Update system and install dependencies\n"
        script += "sudo apt-get update\n"
        script += "sudo apt-get install -y git tmux htop\n\n"
        
        # Clone repository if provided
        if repo_url:
            script += f"# Clone repository\n"
            script += f"git clone {repo_url} ~/automatic-finetune\n"
            script += f"cd ~/automatic-finetune\n\n"
        
        # Install Python requirements
        if requirements:
            script += "# Install Python requirements\n"
            if isinstance(requirements, str) and os.path.exists(requirements):
                # If it's a requirements file path
                with open(requirements, 'r') as f:
                    req_content = f.read()
                script += "cat > ~/requirements.txt << 'EOL'\n"
                script += req_content
                script += "\nEOL\n"
                script += "pip install -r ~/requirements.txt\n\n"
            elif isinstance(requirements, list):
                # If it's a list of packages
                script += f"pip install {' '.join(requirements)}\n\n"
        else:
            # Default packages for LLM fine-tuning
            script += "# Install default LLM fine-tuning packages\n"
            script += "pip install -U torch==2.3.1 torchvision==0.17.2\n"
            script += "pip install unsloth datasets boto3 pyyaml mlflow python-dotenv\n\n"
        
        # Setup AWS credentials if provided
        script += "# Setup AWS credentials if provided as environment variables\n"
        script += 'if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then\n'
        script += '    mkdir -p ~/.aws\n'
        script += '    cat > ~/.aws/credentials << EOL\n'
        script += '[default]\n'
        script += 'aws_access_key_id = $AWS_ACCESS_KEY_ID\n'
        script += 'aws_secret_access_key = $AWS_SECRET_ACCESS_KEY\n'
        script += 'EOL\n'
        script += '    if [ -n "$AWS_REGION" ]; then\n'
        script += '        cat > ~/.aws/config << EOL\n'
        script += '[default]\n'
        script += 'region = $AWS_REGION\n'
        script += 'EOL\n'
        script += '    fi\n'
        script += 'fi\n\n'
        
        return script


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Manage Lambda Cloud instances for LLM fine-tuning")
    
    # Main command arguments
    parser.add_argument('--config', '-c', help="Path to YAML config file")
    parser.add_argument('--api-key', help="Lambda Cloud API key (or set LAMBDA_API_KEY env var)")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List instances command
    list_parser = subparsers.add_parser('list', help='List running instances')
    
    # List instance types/pricing command
    pricing_parser = subparsers.add_parser('pricing', help='List instance types and pricing')
    
    # Launch instance command
    launch_parser = subparsers.add_parser('launch', help='Launch a new instance')
    launch_parser.add_argument('--name', help='Instance name')
    launch_parser.add_argument('--region', help='Region name')
    launch_parser.add_argument('--instance-type', help='Instance type')
    launch_parser.add_argument('--ssh-key', help='SSH key name')
    launch_parser.add_argument('--repo', help='Git repository to clone')
    launch_parser.add_argument('--requirements', help='Path to requirements.txt or comma-separated package list')
    
    # Terminate instance command
    terminate_parser = subparsers.add_parser('terminate', help='Terminate an instance')
    terminate_parser.add_argument('instance_id', help='Instance ID to terminate')
    
    # SSH command
    ssh_parser = subparsers.add_parser('ssh', help='SSH into an instance')
    ssh_parser.add_argument('instance_id', help='Instance ID to SSH into')
    ssh_parser.add_argument('--key-path', help='Path to SSH private key')
    ssh_parser.add_argument('--command', help='Command to run on the instance')
    
    # Execute training command
    train_parser = subparsers.add_parser('train', help='Run training on an instance')
    train_parser.add_argument('instance_id', help='Instance ID to run training on')
    train_parser.add_argument('--key-path', help='Path to SSH private key')
    train_parser.add_argument('--config-path', help='Path to training config')
    
    args = parser.parse_args()
    
    # Create Lambda Cloud manager
    try:
        manager = LambdaCloudManager(api_key=args.api_key, config_path=args.config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Handle commands
    if args.command == 'list':
        instances = manager.list_instances()
        if not instances:
            print("No instances found")
            return 0
            
        print("\nRunning Instances:")
        print("-" * 80)
        print(f"{'ID':<24} {'Name':<30} {'Status':<10} {'IP Address':<15} {'Instance Type':<20}")
        print("-" * 80)
        for instance in instances:
            print(f"{instance.get('id', 'N/A'):<24} {instance.get('name', 'N/A'):<30} "
                  f"{instance.get('status', 'N/A'):<10} {instance.get('ip', 'N/A'):<15} "
                  f"{instance.get('instance_type', {}).get('name', 'N/A'):<20}")
        print("-" * 80)
    
    elif args.command == 'pricing':
        instance_types = manager.get_instance_types()
        
        print("\nAvailable Instance Types:")
        print("-" * 80)
        print(f"{'Name':<20} {'Description':<40} {'Price ($/hr)':<15} {'Regions':<20}")
        print("-" * 80)
        
        for instance_type_name, instance_type in instance_types.items():
            regions = ", ".join(instance_type.get('regions_with_capacity_available', []))
            if not regions:
                regions = "No capacity"
            
            print(f"{instance_type_name:<20} {instance_type.get('description', 'N/A'):<40} "
                  f"${instance_type.get('price_cents_per_hour', 0)/100:<14.2f} {regions[:20]:<20}")
        print("-" * 80)
    
    elif args.command == 'launch':
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
        
        # Generate setup script if repo provided
        if args.repo:
            user_data = manager.generate_setup_script(
                repo_url=args.repo,
                requirements=args.requirements
            )
            config_override['user_data'] = user_data
        
        # Launch the instance
        instance_id = manager.launch_instance(config_override)
        if not instance_id:
            print("Failed to launch instance")
            return 1
        
        try:
            ip_address = manager.wait_for_instance(instance_id)
            print(f"\n🚀 Instance {instance_id} is ready!")
            print(f"IP Address: {ip_address}")
            print(f"SSH command: ssh ubuntu@{ip_address}")
            
            # Save instance details to file for reference
            with open('.lambda_instance.json', 'w') as f:
                json.dump({
                    'instance_id': instance_id,
                    'ip_address': ip_address,
                    'launched_at': time.time()
                }, f)
            
            print(f"Instance details saved to .lambda_instance.json")
        except TimeoutError as e:
            print(f"Error: {e}")
            return 1
    
    elif args.command == 'terminate':
        print(f"Terminating instance {args.instance_id}...")
        manager.terminate_instance(args.instance_id)
        print(f"Instance {args.instance_id} termination request sent")
    
    elif args.command == 'ssh':
        # Get the IP address
        ip_address = manager.get_instance_ip(args.instance_id)
        if not ip_address:
            print(f"Error: Could not find IP address for instance {args.instance_id}")
            return 1
        
        # Build the SSH command
        ssh_cmd = ['ssh']
        if args.key_path:
            ssh_cmd.extend(['-i', args.key_path])
        
        ssh_cmd.append(f'ubuntu@{ip_address}')
        
        if args.command:
            ssh_cmd.append(args.command)
            
        print(f"Connecting to instance {args.instance_id} at {ip_address}...")
        subprocess.run(ssh_cmd)
    
    elif args.command == 'train':
        # Get the IP address
        ip_address = manager.get_instance_ip(args.instance_id)
        if not ip_address:
            print(f"Error: Could not find IP address for instance {args.instance_id}")
            return 1
        
        # Build the SSH command for running training
        ssh_cmd = ['ssh']
        if args.key_path:
            ssh_cmd.extend(['-i', args.key_path])
        
        # Default training command
        training_cmd = 'cd ~/automatic-finetune && python -m src.cli.finetuning_cli train'
        
        # Add config path if provided
        if args.config_path:
            # Need to upload the config first
            scp_cmd = ['scp']
            if args.key_path:
                scp_cmd.extend(['-i', args.key_path])
            scp_cmd.extend([args.config_path, f'ubuntu@{ip_address}:~/training_config.yaml'])
            
            print(f"Uploading training config to instance...")
            subprocess.run(scp_cmd)
            
            # Add config argument to training command
            training_cmd += ' --config ~/training_config.yaml'
        
        ssh_cmd.extend([f'ubuntu@{ip_address}', training_cmd])
        
        print(f"Starting training on instance {args.instance_id} at {ip_address}...")
        subprocess.run(ssh_cmd)
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())