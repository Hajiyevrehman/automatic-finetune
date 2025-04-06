
# LLM Fine-Tuning Pipeline

A comprehensive, modular pipeline for preparing datasets and fine-tuning Large Language Models, with S3 integration for scalable cloud storage.

## Features

- **Complete Data Processing Pipeline**: From raw data to tokenized, split datasets ready for fine-tuning
- **Cloud Integration**: Built-in S3 support for storing and retrieving datasets
- **Modular Design**: Process datasets with a single command or step-by-step
- **Reproducible**: Uses DVC for tracking data versions and pipeline stages
- **Extensible**: Easy to add new dataset converters or processing steps
- **Training Automation**: Launch cloud GPU instances and run fine-tuning with minimal setup

## Project Structure

```
.
├── configs/                 # Configuration files
│   ├── data/                # Data processing configs
│   └── training/            # Training configs
├── scripts/                 # Utility scripts
│   ├── dataset_converters/  # Dataset-specific converters
│   ├── launch-a100-direct.py # Cloud GPU launcher
│   ├── setup_env.sh         # Environment setup script 
│   └── utils/               # Utility scripts
├── src/                     # Source code
│   ├── cli/                 # Command-line interfaces
│   ├── cloud/               # Cloud integration (AWS/S3)
│   │   ├── auth.py
│   │   └── storage.py
│   ├── data/                # Data processing pipeline
│   └── finetuning/          # Fine-tuning implementation
│       └── unsloth_trainer.py
├── tests/                   # Test suite
└── run_train.sh             # Main training script
```

## Getting Started

### Prerequisites

- Python 3.9+
- AWS account with S3 access
- Required Python packages (see Installation)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hajiyevrehman/automatic-finetune.git
   cd automatic-finetune
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_REGION=your-region
   ```

   Alternatively, create a `.env` file with these variables.

## Data Pipeline

### Configuration

Edit `configs/data/data_processing.yaml` to customize the pipeline:

```yaml
# Data directories
directories:
  raw: "data/raw"
  processed: "data/processed"
  validation: "data/validation"

# Model configuration
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_length: 2048

# S3 configuration
s3:
  default_bucket: "your-bucket-name"
  region: "us-east-1"
```

### Setting Up DVC

Initialize DVC and configure S3 storage:

```bash
python scripts/utils/setup_dvc.py --bucket your-bucket-name --dataset servicenow-qa --init
```

### Running the Pipeline

Execute the complete pipeline with DVC:

```bash
dvc repro
```

Or run individual stages:

```bash
dvc repro download
dvc repro convert
dvc repro preprocess
# etc.
```

### Using the CLI Directly

You can also use the CLI interface directly:

```bash
# Convert a dataset
python -m src.cli.data_cli convert --dataset servicenow-qa --output servicenow-qa_converted --config configs/data/data_processing.yaml

# Run the complete pipeline
python -m src.cli.data_cli run-pipeline --dataset servicenow-qa --config configs/data/data_processing.yaml
```

### Adding a New Dataset

1. Create a converter script in `scripts/dataset_converters/`:
   ```python
   # your_dataset_converter.py
   import json

   def main():
       # Read your dataset and convert to the required format
       # Save to data/raw/your-dataset.json

   if __name__ == "__main__":
       main()
   ```

2. Update the DVC pipeline:
   ```bash
   python scripts/utils/setup_dvc.py --bucket your-bucket-name --dataset your-dataset-name
   ```

## Training Pipeline

### Launching a Cloud Instance

To launch a cloud GPU instance for training:

```bash
python scripts/launch-a100-direct.py --gpu-count 1 --create-ssh
```

This will provision an A100 GPU instance and output connection details.

### Connecting to the Instance

Use SSH to connect to your instance:

```bash
ssh -i /path/to/your/key ubuntu@instance-ip-address -o IdentitiesOnly=yes
```

### Setting Up the Training Environment

Once connected to the instance:

```bash
# Clone the repository
git clone https://github.com/Hajiyevrehman/automatic-finetune.git
cd automatic-finetune

# Start the training process
./run_train.sh
```

When prompted, enter your AWS credentials to enable S3 access for dataset and model storage.

### Updating Configuration Files

To update the training configuration from your local machine:

```bash
# Extract instance details from saved JSON
IP=$(grep -o '"ip_address": *"[^"]*"' .lambda_instance.json | cut -d'"' -f4)
KEY_PATH=$(grep -o '"ssh_key_path": *"[^"]*"' .lambda_instance.json | cut -d'"' -f4)

# Transfer configuration file
scp -i "$KEY_PATH" -o IdentitiesOnly=yes configs/training/llm_finetuning.yaml ubuntu@$IP:~/automatic-finetune/configs/training/llm_finetuning.yaml
```

## Pipeline Stages

1. **Download**: Download or prepare the raw dataset
2. **Convert**: Convert to a standardized format with user/assistant messages
3. **Preprocess**: Clean and normalize the text data
4. **Validate**: Check data quality and format
5. **Tokenize**: Tokenize data using the target model's tokenizer
6. **Split**: Divide into train/validation/test sets
7. **Train**: Fine-tune the model using Unsloth optimization

## Testing

Run the test suite with:

```bash
python -m pytest
```

## Acknowledgments

- ServiceNow for the example dataset
- Qwen for the tokenizer used in this example
- Unsloth for the optimized training implementation
