#!/usr/bin/env python3
"""
Script to list available Lambda Cloud images
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

# Configuration
API_BASE_URL = "https://cloud.lambdalabs.com/api/v1"

def load_credentials():
    """Load API key from environment"""
    # Try loading from .env files
    for env_path in ["scripts/.env", ".env"]:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
    
    api_key = os.environ.get('LAMBDA_API_KEY')
    if not api_key:
        api_key = input("Enter your Lambda API key: ")
    
    return api_key

def get_images(api_key):
    """Get list of available images"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(f"{API_BASE_URL}/images", headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    return response.json().get('data', [])

def main():
    api_key = load_credentials()
    if not api_key:
        print("No API key found")
        return 1
    
    images = get_images(api_key)
    if not images:
        print("No images found or error occurred")
        return 1
    
    print("\nAvailable Images:")
    print("-" * 100)
    print(f"{'ID':<38} {'Name':<30} {'Description':<30}")
    print("-" * 100)
    
    for image in images:
        image_id = image.get('id', 'Unknown')
        name = image.get('name', 'Unknown')
        description = image.get('description', '')
        
        print(f"{image_id:<38} {name:<30} {description:<30}")
    
    print("-" * 100)
    print("\nTo use an image, run your script with --image <IMAGE_ID>")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())