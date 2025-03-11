#!/bin/bash
# Setup script for creating an isolated Python environment for unsloth fine-tuning

set -e  # Exit on error

# Configuration
VENV_DIR=~/pyenv/unsloth-env
TORCH_VERSION=2.2.0
TORCHVISION_VERSION=0.17.0
TRANSFORMERS_VERSION=4.38.0
UNSLOTH_VERSION=2025.3.1

# Create directories
mkdir -p ~/pyenv
mkdir -p configs/training
mkdir -p configs/data
mkdir -p scripts
mkdir -p src/finetuning

# Create virtual environment
echo "Creating virtual environment at $VENV_DIR..."
python -m venv --clear $VENV_DIR

# Activate it
source $VENV_DIR/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION
pip install transformers==$TRANSFORMERS_VERSION datasets boto3 pyyaml mlflow
pip install unsloth==$UNSLOTH_VERSION

# Copy necessary files into the environment
echo "Setting up project structure..."
cp -r src $VENV_DIR/
cp -r configs $VENV_DIR/
cp -r scripts $VENV_DIR/
cp *.py $VENV_DIR/ 2>/dev/null || true

# If .env file exists, copy it
if [ -f ".env" ]; then
    cp .env $VENV_DIR/
fi

# If .env file exists in scripts directory, copy it
if [ -f "scripts/.env" ]; then
    mkdir -p $VENV_DIR/scripts
    cp scripts/.env $VENV_DIR/scripts/
fi

echo "Environment setup complete at $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"