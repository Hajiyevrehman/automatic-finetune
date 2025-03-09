#!/usr/bin/env python3
"""
Improved script to launch an A100 instance on Lambda Cloud.
Features:
- .env file support for API keys (AWS and Lambda)
- Better SSH key management
- Detailed connection instructions
- Automatic setup script generation
"""

import os
import sys
import time
import requests
import json
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

# Configuration
API_BASE_URL = "https://cloud.lambdalabs.com/api/v1"

def load_env_variables(env_path=None):
    """Load environment variables from .env file"""
    try:
        # Try to import dotenv
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
        from dotenv import load_dotenv
    
    # Default paths to look for .env files
    paths_to_check = [
        env_path,
        "scripts/.env",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", ".env"),
        ".env"
    ]
    
    # Filter out None values
    paths_to_check = [p for p in paths_to_check if p]
    
    # Try to find and load .env file
    env_loaded = False
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"Loading environment variables from {path}")
            load_dotenv(path)
            env_loaded = True
            break
    
    # Check for required variables
    if os.environ.get('LAMBDA_API_KEY'):
        print("✓ Lambda API key loaded from .env file")
    elif env_loaded:
        print("⚠️ .env file found but LAMBDA_API_KEY not set in the file")
    
    if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("✓ AWS credentials loaded from .env file")
    
    # Return all environment variables as a dict
    return {key: val for key, val in os.environ.items()}

def get_api_key(api_key_arg=None):
    """Get Lambda API key from argument, environment or prompt user"""
    # First load from .env files
    load_env_variables()
    
    # Try to get from args, then environment
    api_key = api_key_arg or os.environ.get('LAMBDA_API_KEY')
    
    if not api_key:
        # Try to read from credentials file
        try:
            import yaml
            creds_path = os.path.expanduser("~/.cloud-finetuning/credentials.yaml")
            if os.path.exists(creds_path):
                with open(creds_path, 'r') as f:
                    creds = yaml.safe_load(f)
                    if creds and 'lambda' in creds and 'api_key' in creds['lambda']:
                        api_key = creds['lambda']['api_key']
        except Exception as e:
            print(f"Error reading credentials: {e}")
        
        # If still not found, prompt user
        if not api_key:
            api_key = input("Enter your Lambda API key: ").strip()
            
            # Ask if user wants to save to .env file
            save_to_env = input("Do you want to save this API key to your .env file? (y/n): ").strip().lower()
            if save_to_env == 'y':
                env_path = input("Enter path to .env file (default: scripts/.env): ").strip() or "scripts/.env"
                
                # Create directory if needed
                os.makedirs(os.path.dirname(env_path), exist_ok=True)
                
                # Check if file exists
                env_content = ""
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        env_content = f.read()
                
                # Check if LAMBDA_API_KEY already in file
                if "LAMBDA_API_KEY=" in env_content:
                    # Replace existing key
                    env_lines = env_content.splitlines()
                    for i, line in enumerate(env_lines):
                        if line.strip().startswith("LAMBDA_API_KEY="):
                            env_lines[i] = f"LAMBDA_API_KEY={api_key}"
                            break
                    env_content = "\n".join(env_lines)
                else:
                    # Add new key
                    if env_content and not env_content.endswith("\n"):
                        env_content += "\n"
                    env_content += f"LAMBDA_API_KEY={api_key}\n"
                
                # Write back to file
                with open(env_path, 'w') as f:
                    f.write(env_content)
                
                print(f"API key saved to {env_path}")
            
    if not api_key:
        print("Error: Lambda API key is required")
        sys.exit(1)
    
    # Save to environment variable for convenience
    os.environ['LAMBDA_API_KEY'] = api_key
    return api_key

def get_headers(api_key):
    """Get headers for API requests"""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def get_ssh_keys(headers):
    """Get list of SSH keys"""
    response = requests.get(f"{API_BASE_URL}/ssh-keys", headers=headers)
    if response.status_code != 200:
        print(f"Error getting SSH keys: {response.text}")
        return []
    
    ssh_keys = response.json().get('data', [])
    return ssh_keys

def display_ssh_keys(ssh_keys):
    """Display SSH keys in a formatted way"""
    if not ssh_keys:
        print("No SSH keys found in your Lambda account")
        return
    
    print(f"\nFound {len(ssh_keys)} SSH keys:")
    print("-" * 80)
    print(f"{'Name':<30} {'Created':<25} {'ID':<20}")
    print("-" * 80)
    for i, key in enumerate(ssh_keys):
        # Format creation date if available
        created = key.get('created_at', 'Unknown')
        if isinstance(created, (int, float)):
            from datetime import datetime
            created = datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{i+1}. {key.get('name', 'Unnamed'):<27} {created:<25} {key.get('id', 'No ID'):<20}")
    print("-" * 80)

def create_ssh_key(headers, key_name=None):
    """Create a new SSH key and upload to Lambda"""
    if not key_name:
        key_name = f"lambda-key-{int(time.time())}"
    
    # Create a temporary directory for the SSH key
    with tempfile.TemporaryDirectory() as temp_dir:
        key_path = os.path.join(temp_dir, "lambda_key")
        pub_key_path = f"{key_path}.pub"
        
        # Create SSH key
        print(f"\nGenerating new SSH key: {key_name}")
        try:
            subprocess.run(
                ["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", "", "-C", key_name],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error generating SSH key: {e.stderr.decode('utf-8') if hasattr(e, 'stderr') else str(e)}")
            return None, None
        
        # Read public key
        with open(pub_key_path, 'r') as f:
            public_key = f.read().strip()
        
        # Upload to Lambda
        print("Uploading public key to Lambda...")
        response = requests.post(
            f"{API_BASE_URL}/ssh-keys",
            headers=headers,
            json={"name": key_name, "public_key": public_key}
        )
        
        if response.status_code != 200:
            print(f"Error uploading SSH key: {response.text}")
            return None, None
        
        key_info = response.json().get('data', {})
        print(f"SSH key created and uploaded: {key_info.get('name', key_name)}")
        
        # Save private key to ~/.ssh
        ssh_dir = os.path.expanduser("~/.ssh")
        os.makedirs(ssh_dir, exist_ok=True)
        
        local_key_path = os.path.join(ssh_dir, f"lambda_{key_name}")
        with open(key_path, 'r') as src, open(local_key_path, 'w') as dst:
            dst.write(src.read())
        
        # Set permissions
        os.chmod(local_key_path, 0o600)
        
        print(f"Private key saved to: {local_key_path}")
        return key_info, local_key_path

def get_instance_types(headers, gpu_count=None, filter_type=None):
    """Get list of available instance types and check which have capacity"""
    response = requests.get(f"{API_BASE_URL}/instance-types", headers=headers)
    if response.status_code != 200:
        print(f"Error getting instance types: {response.text}")
        return {}
    
    instance_types = response.json().get('data', {})
    
    # Filter instance types based on criteria
    filtered_types = {}
    for name, data in instance_types.items():
        # Apply GPU type filter (a100, h100, etc)
        if filter_type and filter_type.lower() not in name.lower():
            continue
            
        # Filter by GPU count if specified
        if gpu_count is not None:
            # Look for patterns like "gpu_1x" or "gpu_8x" in the name
            gpu_count_pattern = f"gpu_{gpu_count}x"
            if gpu_count_pattern not in name.lower():
                continue
        
        filtered_types[name] = data
    
    if not filtered_types:
        print(f"No matching instance types found with the specified criteria.")
        
        if filter_type and gpu_count:
            print(f"No {filter_type} instances found with {gpu_count}x GPUs.")
        
        print("\nAvailable GPU instance types:")
        for name in sorted(instance_types.keys()):
            if "gpu" in name.lower():
                count_match = re.search(r'gpu_(\d+)x', name.lower())
                if count_match:
                    count = count_match.group(1)
                    print(f"  - {name} ({count}x GPUs)")
        return {}

    print(f"\nAvailable matching instance types:")
    print("-" * 100)
    print(f"{'Name':<25} {'Description':<40} {'Price ($/hr)':<15} {'Regions with Capacity':<20}")
    print("-" * 100)
    
    for name, data in sorted(filtered_types.items()):
        regions = []
        if 'regions_with_capacity_available' in data:
            regions_data = data['regions_with_capacity_available']
            
            # Check if the list is not empty before accessing elements
            if regions_data:
                if isinstance(regions_data[0], dict):
                    regions = [r.get('name', 'Unknown') for r in regions_data]
                else:
                    regions = regions_data
        
        # Extract pricing if available
        price = data.get('price_cents_per_hour', 0) / 100.0
        
        region_str = ", ".join(regions) if regions else "No capacity"
        if len(region_str) > 20:
            region_str = region_str[:17] + "..."
        
        print(f"{name:<25} {data.get('description', 'No description'):<40} "
              f"${price:<14.2f} {region_str:<20}")
    
    print("-" * 100)
    return filtered_types

def generate_setup_script(repo_url=None, install_packages=None):
    """Generate a setup script to run on instance startup"""
    script = "#!/bin/bash\n\n"
    script += "# Auto-generated setup script for LLM fine-tuning\n\n"
    
    # Basic system updates
    script += "echo 'Running system updates...'\n"
    script += "sudo apt-get update\n"
    script += "sudo apt-get install -y git tmux htop\n\n"
    
    # Clone repository if provided
    if repo_url:
        script += "echo 'Cloning repository...'\n"
        script += f"git clone {repo_url} ~/automatic-finetune\n"
        script += "cd ~/automatic-finetune\n\n"
    
    # Create .env file with AWS credentials if available
    script += "# Setting up environment variables\n"
    script += "mkdir -p ~/automatic-finetune/scripts\n\n"
    
    script += "cat > ~/automatic-finetune/scripts/.env << 'EOL'\n"
    
    # Add environment variables if they exist
    for var in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'LAMBDA_API_KEY']:
        if os.environ.get(var):
            script += f"{var}={os.environ.get(var)}\n"
    
    script += "EOL\n\n"
    
    # Install common ML packages
    script += "echo 'Installing ML packages...'\n"
    script += "pip install -U torch==2.3.1 torchvision==0.17.2\n"
    script += "pip install python-dotenv\n"
    
    # Install additional packages if specified
    if install_packages:
        package_list = " ".join(install_packages)
        script += f"pip install {package_list}\n"
    else:
        script += "pip install unsloth datasets boto3 pyyaml mlflow python-dotenv\n"
    
    script += "\necho 'Setup complete!'\n"
    return script

def launch_instance(headers, instance_type, region, image_id, ssh_key_name, name=None, setup_script=None):
    """Launch an instance with the specified parameters"""
    if not name:
        name = f"LLM-Finetuning-{int(time.time())}"
    
    # Make sure we have a valid SSH key name
    ssh_key_names = [ssh_key_name] if ssh_key_name else []
    
    launch_config = {
        "region_name": region,
        "instance_type_name": instance_type,
        "ssh_key_names": ssh_key_names,
        "file_system_names": [],
        "name": name,
        "image": {"id": image_id}
    }
    
    # Add user data (startup script) if provided
    if setup_script:
        launch_config["user_data"] = setup_script
    
    print(f"\nLaunching instance with:")
    print(f"  Instance Type: {instance_type}")
    print(f"  Region: {region}")
    print(f"  Image ID: {image_id}")
    print(f"  Name: {name}")
    print(f"  SSH Key: {ssh_key_name}")
    if setup_script:
        print(f"  Startup Script: Yes (setup script will run on first boot)")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/instance-operations/launch",
            headers=headers,
            json=launch_config
        )
        
        if response.status_code == 200:
            data = response.json().get('data', {})
            instance_ids = data.get('instance_ids', [])
            
            if instance_ids:
                instance_id = instance_ids[0]
                print(f"✅ SUCCESS! Instance launched with ID: {instance_id}")
                return instance_id
            else:
                print(f"❌ No instance ID returned in response: {json.dumps(data)}")
                return None
        else:
            error = response.json().get('error', {})
            print(f"❌ Failed to launch instance: {error.get('message', response.text)}")
            print(f"   Suggestion: {error.get('suggestion', 'No suggestion')}")
            return None
    except Exception as e:
        print(f"❌ Error launching instance: {e}")
        return None

def wait_for_instance(headers, instance_id, timeout=3000):
    """Wait for instance to be ready"""
    print(f"Waiting for instance {instance_id} to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_BASE_URL}/instances", headers=headers)
            if response.status_code == 200:
                instances = response.json().get('data', [])
                for instance in instances:
                    if instance.get('id') == instance_id:
                        status = instance.get('status')
                        ip = instance.get('ip')
                        
                        print(f"  Status: {status}, IP: {ip or 'Not assigned yet'}")
                        
                        if status == 'active' and ip:
                            print(f"✅ Instance is ready! IP address: {ip}")
                            return ip
            else:
                print(f"  Warning: Failed to get instance status, will retry...")
        except Exception as e:
            print(f"  Warning: Error checking instance status: {e}")
        
        time.sleep(15)
    
    print(f"❌ Timed out waiting for instance to be ready")
    return None

def save_instance_info(instance_id, ip_address, instance_type, region, image_id, ssh_key_name, ssh_key_path, output_file):
    """Save instance information to a file"""
    info = {
        "instance_id": instance_id,
        "ip_address": ip_address,
        "instance_type": instance_type,
        "region": region,
        "image_id": image_id,
        "ssh_key_name": ssh_key_name,
        "ssh_key_path": ssh_key_path,
        "created_at": time.time(),
        # Include AWS credentials for reference
        "aws_credentials": {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "aws_region": os.environ.get("AWS_REGION", "")
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Instance information saved to {output_file}")

def print_connection_instructions(ip_address, ssh_key_path):
    """Print detailed connection instructions"""
    print("\n" + "=" * 80)
    print("CONNECTION INSTRUCTIONS")
    print("=" * 80)
    
    ssh_cmd = f"ssh -i {ssh_key_path} ubuntu@{ip_address}"
    
    print(f"\n1. Connect to your instance:")
    print(f"   {ssh_cmd}")
    
    print(f"\n2. If you get a permission error, make sure your key has the right permissions:")
    print(f"   chmod 600 {ssh_key_path}")
    
    print(f"\n3. To transfer files to your instance:")
    print(f"   scp -i {ssh_key_path} path/to/local/file ubuntu@{ip_address}:~/path/on/instance")
    
    print(f"\n4. For persistent sessions, use tmux:")
    print(f"   {ssh_cmd} 'tmux new -s training'")
    print(f"   # To detach: press Ctrl+B, then D")
    print(f"   # To reattach: {ssh_cmd} 'tmux attach -t training'")
    
    print("\n5. To run your training script:")
    print(f"   {ssh_cmd} 'cd ~/automatic-finetune && python src/run_training.py'")
    
    print("\n6. Environment variables from your .env file have been transferred to the instance")
    print("   This includes your AWS credentials for accessing S3")
    
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Launch a GPU instance on Lambda Cloud")
    
    # Instance configuration
    parser.add_argument('--image', default="ea87a5ee-5257-4286-aeb0-8607525801e2", 
                        help="Image ID (default: Lambda Stack Ubuntu 22.04)")
    parser.add_argument('--instance-type', default=None, help="Instance type (e.g., gpu_8x_a100_80gb_sxm4)")
    parser.add_argument('--region', default=None, help="Region name (e.g., us-east-1)")
    parser.add_argument('--name', default=None, help="Instance name")
    parser.add_argument('--output', default=".lambda_instance.json", help="Output file for instance info")
    parser.add_argument('--gpu-count', type=int, help="Filter by GPU count (e.g., 1, 8)")
    parser.add_argument('--gpu-type', default="a100", help="GPU type (default: a100, options: a100, h100, a10, etc)")
    
    # API and SSH key options
    parser.add_argument('--api-key', help="Lambda Cloud API key")
    parser.add_argument('--env-file', help="Path to .env file for credentials")
    parser.add_argument('--ssh-key-name', help="Name of existing SSH key to use")
    parser.add_argument('--ssh-key-path', help="Path to SSH private key")
    parser.add_argument('--create-ssh-key', action='store_true', help="Create a new SSH key")
    
    # Startup script options
    parser.add_argument('--repo', default="https://github.com/Hajiyevrehman/automatic-finetune.git", 
                       help="Git repository to clone on startup")
    parser.add_argument('--packages', help="Additional packages to install, comma separated")
    
    # Operation modes
    parser.add_argument('--list-ssh-keys', action='store_true', help="List available SSH keys and exit")
    parser.add_argument('--list-only', action='store_true', help="Only list available instances, don't launch")
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    if args.env_file:
        load_env_variables(args.env_file)
    else:
        load_env_variables()
    
    # Get API key and headers
    api_key = get_api_key(args.api_key)
    headers = get_headers(api_key)
    
    # Get SSH keys
    ssh_keys = get_ssh_keys(headers)
    
    # If requested, just list SSH keys and exit
    if args.list_ssh_keys:
        display_ssh_keys(ssh_keys)
        return 0
    
    # SSH key setup
    ssh_key_name = args.ssh_key_name
    ssh_key_path = args.ssh_key_path
    
    # If create-ssh-key flag is set, create a new key
    if args.create_ssh_key:
        key_info, key_path = create_ssh_key(headers)
        if key_info:
            ssh_key_name = key_info.get('name')
            ssh_key_path = key_path
            # Update ssh_keys list
            ssh_keys = get_ssh_keys(headers)
        else:
            print("Failed to create SSH key. Exiting.")
            return 1
    
    # If no SSH key name specified, show available keys and prompt
    if not ssh_key_name:
        if not ssh_keys:
            print("No SSH keys found in your Lambda account")
            create_key = input("Would you like to create a new SSH key? (y/n): ").strip().lower()
            if create_key == 'y':
                key_info, key_path = create_ssh_key(headers)
                if key_info:
                    ssh_key_name = key_info.get('name')
                    ssh_key_path = key_path
                else:
                    print("Failed to create SSH key. Exiting.")
                    return 1
            else:
                print("Please add an SSH key through the Lambda console or using --create-ssh-key")
                return 1
        else:
            display_ssh_keys(ssh_keys)
            
            # Try to select the first key by default
            ssh_key_name = ssh_keys[0]['name']
            print(f"Using SSH key: {ssh_key_name}")
            
            # If there's more than one key, allow selection
            if len(ssh_keys) > 1:
                use_default = input(f"Use this SSH key? (y/n): ").strip().lower()
                if use_default != 'y':
                    key_idx = int(input(f"Enter key number (1-{len(ssh_keys)}): ").strip()) - 1
                    if 0 <= key_idx < len(ssh_keys):
                        ssh_key_name = ssh_keys[key_idx]['name']
                        print(f"Using SSH key: {ssh_key_name}")
                    else:
                        print("Invalid key selection. Exiting.")
                        return 1
    
    # Get available instance types and check which regions have capacity
    instance_types = get_instance_types(headers, args.gpu_count, args.gpu_type)
    
    # If list-only mode, exit after listing
    if args.list_only:
        return 0
    
    # If no instances were found, exit
    if not instance_types:
        print("No suitable instances found with your criteria. Try different options.")
        return 1
    
    # Determine instance type and region
    selected_instance_type = args.instance_type
    selected_region = args.region
    
    # If instance type not specified, use the first one with capacity
    if not selected_instance_type:
        for name, data in instance_types.items():
            regions = []
            if 'regions_with_capacity_available' in data:
                regions_data = data['regions_with_capacity_available']
                if regions_data:  # Check if not empty
                    if isinstance(regions_data[0], dict):
                        regions = [r.get('name', 'Unknown') for r in regions_data]
                    else:
                        regions = regions_data
                    
            if regions:
                selected_instance_type = name
                if not selected_region:
                    selected_region = regions[0]
                break
    
    # If no instance type with capacity was found
    if not selected_instance_type:
        print("Error: No instance types with capacity available")
        return 1
    
    # If instance type specified but no region specified
    if selected_instance_type and not selected_region:
        instance_data = instance_types.get(selected_instance_type, {})
        regions = []
        if 'regions_with_capacity_available' in instance_data:
            regions_data = instance_data['regions_with_capacity_available']
            if regions_data:  # Check if not empty
                if isinstance(regions_data[0], dict):
                    regions = [r.get('name', 'Unknown') for r in regions_data]
                else:
                    regions = regions_data
                
        if regions:
            selected_region = regions[0]
        else:
            print(f"Error: No capacity available for {selected_instance_type}")
            return 1
    
    # Print selected configuration
    print(f"\nSelected configuration:")
    print(f"  Instance Type: {selected_instance_type}")
    print(f"  Region: {selected_region}")
    print(f"  Image ID: {args.image}")
    print(f"  SSH Key: {ssh_key_name}")
    
    # Check for AWS credentials
    aws_creds_found = False
    if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
        aws_creds_found = True
        print(f"  AWS Credentials: ✓ (will be transferred to instance)")
    else:
        print(f"  AWS Credentials: ✗ (not found in environment)")
    
    # Confirm before launching expensive instances
    if "8x" in selected_instance_type or "4x" in selected_instance_type or "h100" in selected_instance_type.lower():
        price = instance_types.get(selected_instance_type, {}).get('price_cents_per_hour', 0) / 100.0
        if price > 5.0:  # If more than $5 per hour
            confirm = input(f"\n⚠️ This instance costs ${price:.2f}/hour. Proceed? (y/n): ")
            if confirm.lower() != 'y':
                print("Operation cancelled by user.")
                return 0
    
    # Create setup script if requested
    setup_script = None
    if args.repo:
        packages = args.packages.split(',') if args.packages else None
        setup_script = generate_setup_script(args.repo, packages)
    
    # Launch the instance
    instance_id = launch_instance(
        headers, 
        selected_instance_type, 
        selected_region, 
        args.image, 
        ssh_key_name,
        args.name,
        setup_script
    )
    
    if not instance_id:
        return 1
    
    # Wait for the instance to be ready
    ip_address = wait_for_instance(headers, instance_id)
    if not ip_address:
        return 1
    
    # Save instance information
    save_instance_info(
        instance_id,
        ip_address,
        selected_instance_type,
        selected_region,
        args.image,
        ssh_key_name,
        ssh_key_path,
        args.output
    )
    
    # Print detailed connection instructions
    print_connection_instructions(ip_address, ssh_key_path or "~/.ssh/id_rsa")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())