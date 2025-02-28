#!/usr/bin/env python
"""
Lambda Labs cloud integration for LLM training
This module provides functions to launch, monitor, and terminate Lambda Labs instances
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import paramiko
import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class LambdaLabsConfig:
    api_key: str
    instance_type: str = "gpu_a10"
    region: str = "us-west-1"
    ssh_key_name: str = "default"
    name: str = "llm-training"
    ssh_key_path: Optional[str] = None
    file_mounts: Dict[str, str] = None


class LambdaLabsClient:
    """Client for interacting with Lambda Labs API and instances"""

    API_BASE_URL = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, config: LambdaLabsConfig):
        self.config = config
        self.api_key = config.api_key
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

        # Validate SSH key exists on Lambda Labs
        self._validate_ssh_key()

    def _validate_ssh_key(self):
        """Ensure that the specified SSH key exists on Lambda Labs"""
        response = self.session.get(f"{self.API_BASE_URL}/ssh-keys")
        response.raise_for_status()

        keys = response.json().get("data", [])
        key_names = [key["name"] for key in keys]

        if self.config.ssh_key_name not in key_names:
            raise ValueError(
                f"SSH key '{self.config.ssh_key_name}' not found in Lambda Labs account. "
                f"Available keys: {', '.join(key_names)}"
            )

        logger.info(f"SSH key '{self.config.ssh_key_name}' validated")

    def list_instances(self) -> List[Dict[str, Any]]:
        """List all running instances"""
        response = self.session.get(f"{self.API_BASE_URL}/instances")
        response.raise_for_status()
        return response.json().get("data", [])

    def get_instance_types(self) -> Dict[str, Dict[str, Any]]:
        """Get available instance types and their specs"""
        response = self.session.get(f"{self.API_BASE_URL}/instance-types")
        response.raise_for_status()
        return response.json().get("data", {})

    def launch_instance(self) -> Dict[str, Any]:
        """Launch a new instance for training"""
        payload = {
            "region_name": self.config.region,
            "instance_type_name": self.config.instance_type,
            "ssh_key_names": [self.config.ssh_key_name],
            "name": self.config.name,
            "quantity": 1,
        }

        logger.info(
            f"Launching Lambda Labs instance ({self.config.instance_type}) in {self.config.region}"
        )
        response = self.session.post(f"{self.API_BASE_URL}/instances", json=payload)

        if response.status_code != 200:
            error_msg = f"Failed to launch instance: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

        instance_data = response.json()
        logger.info(f"Instance launched: {instance_data}")
        return instance_data.get("data", [])[0]

    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance by ID"""
        logger.info(f"Terminating instance {instance_id}")
        response = self.session.delete(f"{self.API_BASE_URL}/instances/{instance_id}")

        if response.status_code != 200:
            logger.error(f"Failed to terminate instance: {response.text}")
            return False

        logger.info(f"Instance {instance_id} termination initiated")
        return True

    def wait_for_instance_ready(
        self, instance_id: str, timeout: int = 300
    ) -> Dict[str, Any]:
        """Wait for an instance to be in the 'active' state"""
        logger.info(f"Waiting for instance {instance_id} to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.API_BASE_URL}/instances/{instance_id}")
            if response.status_code != 200:
                logger.warning(f"Failed to check instance status: {response.text}")
                time.sleep(5)
                continue

            instance = response.json().get("data", {})
            status = instance.get("status")

            if status == "active":
                logger.info(f"Instance {instance_id} is now ready")
                return instance

            logger.info(f"Instance status: {status}, waiting...")
            time.sleep(10)

        raise TimeoutError(
            f"Timed out waiting for instance {instance_id} to become ready"
        )

    def get_ssh_client(self, instance_ip: str) -> paramiko.SSHClient:
        """Create an SSH client connected to the instance"""
        if not self.config.ssh_key_path:
            raise ValueError("SSH key path not provided in config")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        logger.info(f"Connecting to instance at {instance_ip}")

        # Try to connect with retries
        max_retries = 5
        retry_delay = 10

        for i in range(max_retries):
            try:
                client.connect(
                    instance_ip,
                    username="ubuntu",
                    key_filename=self.config.ssh_key_path,
                    timeout=10,
                )
                logger.info("SSH connection established")
                return client
            except Exception as e:
                if i == max_retries - 1:
                    raise
                logger.warning(
                    f"SSH connection failed (attempt {i+1}/{max_retries}): {str(e)}"
                )
                time.sleep(retry_delay)

    def run_command(self, ssh_client: paramiko.SSHClient, command: str) -> tuple:
        """Run a command on the instance and return stdout and stderr"""
        logger.info(f"Running command: {command}")
        stdin, stdout, stderr = ssh_client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()

        stdout_str = stdout.read().decode("utf-8")
        stderr_str = stderr.read().decode("utf-8")

        if exit_code != 0:
            logger.warning(f"Command failed with exit code {exit_code}")
            logger.warning(f"stderr: {stderr_str}")

        return stdout_str, stderr_str, exit_code

    def upload_file(
        self, ssh_client: paramiko.SSHClient, local_path: str, remote_path: str
    ):
        """Upload a file to the instance"""
        logger.info(f"Uploading {local_path} to {remote_path}")

        sftp = ssh_client.open_sftp()

        # Create remote directory if needed
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            logger.info(f"Creating remote directory: {remote_dir}")
            self.run_command(ssh_client, f"mkdir -p {remote_dir}")

        # Upload file
        sftp.put(local_path, remote_path)
        sftp.close()
        logger.info(f"Upload complete: {remote_path}")

    def upload_directory(
        self,
        ssh_client: paramiko.SSHClient,
        local_dir: str,
        remote_dir: str,
        exclude=None,
    ):
        """Upload a directory to the instance"""
        exclude = exclude or []
        logger.info(f"Uploading directory {local_dir} to {remote_dir}")

        # Create remote directory
        self.run_command(ssh_client, f"mkdir -p {remote_dir}")

        # Upload files
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                # Skip excluded files/patterns
                if any(file.endswith(ext) for ext in exclude):
                    continue

                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                remote_path = os.path.join(remote_dir, rel_path)

                # Create remote subdirectories if needed
                remote_subdir = os.path.dirname(remote_path)
                self.run_command(ssh_client, f"mkdir -p {remote_subdir}")

                # Upload file
                self.upload_file(ssh_client, local_path, remote_path)

    def download_directory(
        self, ssh_client: paramiko.SSHClient, remote_dir: str, local_dir: str
    ):
        """Download a directory from the instance"""
        logger.info(f"Downloading directory {remote_dir} to {local_dir}")

        sftp = ssh_client.open_sftp()

        # Create local directory
        os.makedirs(local_dir, exist_ok=True)

        # Get list of files in remote directory
        stdin, stdout, stderr = ssh_client.exec_command(f"find {remote_dir} -type f")
        files = stdout.read().decode("utf-8").splitlines()

        # Download each file
        for remote_path in tqdm(files, desc="Downloading files"):
            rel_path = os.path.relpath(remote_path, remote_dir)
            local_path = os.path.join(local_dir, rel_path)

            # Create local subdirectories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file
            sftp.get(remote_path, local_path)

        sftp.close()
        logger.info(f"Download complete: {local_dir}")


def setup_instance_for_training(ssh_client, project_dir="/home/ubuntu/llm-training"):
    """Set up the instance with necessary dependencies for training"""
    commands = [
        # Update system and install dependencies
        "sudo apt-get update && sudo apt-get install -y python3-pip git awscli",
        # Create project directory
        f"mkdir -p {project_dir}",
        # Install Python dependencies
        f"cd {project_dir} && pip install -r requirements.txt",
        # Set up environment variables
        'echo "export PYTHONPATH=$PYTHONPATH:' + project_dir + '" >> ~/.bashrc',
        'echo "export MLFLOW_TRACKING_URI=file://'
        + os.path.join(project_dir, "mlruns")
        + '" >> ~/.bashrc',
    ]

    # Add AWS credentials if available
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")

        commands.extend(
            [
                f'echo "export AWS_ACCESS_KEY_ID={aws_access_key}" >> ~/.bashrc',
                f'echo "export AWS_SECRET_ACCESS_KEY={aws_secret_key}" >> ~/.bashrc',
                f'echo "export AWS_REGION={aws_region}" >> ~/.bashrc',
            ]
        )

        # Also configure AWS CLI
        commands.extend(
            [
                f"aws configure set aws_access_key_id {aws_access_key}",
                f"aws configure set aws_secret_access_key {aws_secret_key}",
                f"aws configure set region {aws_region}",
                f"aws configure set output json",
            ]
        )

    commands.append("source ~/.bashrc")

    for cmd in commands:
        logger.info(f"Running setup command: {cmd}")
        stdout, stderr, exit_code = ssh_client.run_command(cmd)

        if exit_code != 0:
            logger.error(f"Setup command failed: {cmd}")
            logger.error(f"Error: {stderr}")
            raise RuntimeError(f"Failed to set up instance: {stderr}")

        logger.info(f"Command output: {stdout}")

    logger.info("Instance setup completed successfully")


def launch_training_job(config_path, api_key, ssh_key_path, project_dir="."):
    """Launch a training job on Lambda Labs"""
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize Lambda Labs client
    lambda_config = LambdaLabsConfig(
        api_key=api_key,
        instance_type=config["cloud"]["instance_type"],
        region=config["cloud"]["region"],
        ssh_key_name=config["cloud"]["ssh_key_name"],
        ssh_key_path=ssh_key_path,
    )

    client = LambdaLabsClient(lambda_config)

    try:
        # Launch instance
        instance = client.launch_instance()
        instance_id = instance["id"]

        logger.info(f"Instance launched with ID: {instance_id}")

        # Wait for instance to be ready
        instance = client.wait_for_instance_ready(instance_id)
        instance_ip = instance["ip"]

        logger.info(f"Instance ready with IP: {instance_ip}")

        # Connect to instance
        ssh_client = client.get_ssh_client(instance_ip)

        # Set up instance
        remote_project_dir = "/home/ubuntu/llm-training"
        setup_instance_for_training(client, remote_project_dir)

        # Upload project files
        client.upload_directory(
            ssh_client,
            project_dir,
            remote_project_dir,
            exclude=[".git", "__pycache__", ".env", "venv", "*.pyc", "*.log"],
        )

        # Launch training
        logger.info("Starting training job")
        training_cmd = f"cd {remote_project_dir} && python src/training/train.py --config {config_path}"

        stdout, stderr, exit_code = client.run_command(ssh_client, training_cmd)

        if exit_code != 0:
            logger.error(f"Training failed: {stderr}")
            return False

        # Download results
        logger.info("Training completed successfully, downloading results")
        client.download_directory(
            ssh_client,
            os.path.join(remote_project_dir, config["output"]["output_dir"]),
            config["output"]["output_dir"],
        )

        client.download_directory(
            ssh_client,
            os.path.join(
                remote_project_dir, config["output"]["mlflow"]["tracking_uri"]
            ),
            config["output"]["mlflow"]["tracking_uri"],
        )

        logger.info("Training job completed and results downloaded")
        return True

    except Exception as e:
        logger.error(f"Error during training job: {str(e)}")
        return False
    finally:
        # Terminate instance if requested
        if input("Terminate instance? (y/n): ").lower() == "y":
            client.terminate_instance(instance_id)
            logger.info(f"Instance {instance_id} terminated")


def main():
    """Command-line interface for Lambda Labs training"""
    import argparse

    parser = argparse.ArgumentParser(description="Launch training job on Lambda Labs")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Lambda Labs API key (or set LAMBDA_API_KEY env var)",
    )
    parser.add_argument(
        "--ssh-key",
        type=str,
        help="Path to SSH private key for Lambda Labs (or set LAMBDA_SSH_KEY env var)",
    )
    parser.add_argument(
        "--project-dir", type=str, default=".", help="Project directory to upload"
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        raise ValueError(
            "Lambda Labs API key must be provided via --api-key or LAMBDA_API_KEY env var"
        )

    # Get SSH key path from args or environment
    ssh_key_path = args.ssh_key or os.environ.get("LAMBDA_SSH_KEY")
    if not ssh_key_path:
        raise ValueError(
            "SSH key path must be provided via --ssh-key or LAMBDA_SSH_KEY env var"
        )

    # Launch training job
    launch_training_job(args.config, api_key, ssh_key_path, args.project_dir)


if __name__ == "__main__":
    main()
