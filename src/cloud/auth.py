#!/usr/bin/env python3
"""
Authentication module for cloud services.
Handles loading credentials from environment variables or config files.
"""

import os
import yaml
from pathlib import Path

def get_cloud_credentials():
    """
    Get cloud credentials from environment variables or config file.
    
    Returns:
        dict: Dictionary containing credentials for different cloud providers
    """
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
    """
    Update environment variables from credentials file.
    """
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
