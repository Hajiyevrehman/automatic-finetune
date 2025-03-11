#!/bin/bash
# Master script to set up environment and run training

set -e # Exit on error

# Check if script is run from the correct directory
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
cd "$REPO_ROOT"

# Source paths
SCRIPTS_DIR="scripts"
SRC_DIR="src"
VENV_DIR=~/pyenv/unsloth-env

# Create directories if they don't exist
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$SRC_DIR"
mkdir -p "configs/training"
mkdir -p "configs/data"

# Check if .env file exists or create it
ENV_FILE="$SCRIPTS_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "AWS credentials file not found. Creating one..."
    
    # Prompt for AWS credentials
    read -p "Enter AWS Access Key ID: " aws_key
    read -p "Enter AWS Secret Access Key: " aws_secret
    read -p "Enter AWS Region (default: us-east-1): " aws_region
    aws_region=${aws_region:-us-east-1}
    
    # Create .env file
    mkdir -p "$SCRIPTS_DIR"
    cat > "$ENV_FILE" << EOF
AWS_ACCESS_KEY_ID=$aws_key
AWS_SECRET_ACCESS_KEY=$aws_secret
AWS_REGION=$aws_region
EOF
    
    echo "AWS credentials saved to $ENV_FILE"
fi

# Check if setup script exists, otherwise create it
SETUP_SCRIPT="$SCRIPTS_DIR/setup_env.sh"
if [ ! -f "$SETUP_SCRIPT" ]; then
    echo "Creating setup script..."
    cat > "$SETUP_SCRIPT" << 'EOF'
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
EOF
    
    chmod +x "$SETUP_SCRIPT"
    echo "Setup script created at $SETUP_SCRIPT"
fi

# Run the setup script if environment doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Setting up Python environment..."
    bash "$SETUP_SCRIPT"
else
    echo "Using existing Python environment at $VENV_DIR"
fi

# Check if training script exists
TRAIN_SCRIPT="$SRC_DIR/run_training.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

# Run the training in the virtual environment
echo "Starting training process..."
source "$VENV_DIR/bin/activate"
cd "$VENV_DIR"
python src/run_training.py --env-file scripts/.env

echo "Training completed!"
echo "Results available in the models/finetuned directory"