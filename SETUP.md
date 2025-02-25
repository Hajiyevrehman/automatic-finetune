# Environment Setup Guide

## Prerequisites
- Python 3.10+
- Git LFS

## Step 1: Clone the Repository
git clone https://github.com/your-username/llm-finetuning-project.git



cd llm-finetuning-project


## Step 2: Set Up Virtual Environment

python3.10 -m venv env
source env/bin/activate  # On Unix/MacOS
OR
env\Scripts\activate  # On Windows


## Step 3: Install Dependencies
pip install -r requirements.txt


## Step 4: Configure Cloud Credentials
Create a `.env` file in the root directory (don't commit this file):
CLOUD_PROVIDER_API_KEY=your_api_key
CLOUD_PROVIDER_SECRET=your_secret


## Step 5: Verify Setup
python -m pytest tests/smoke_test.py
