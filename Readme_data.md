# LLM Fine-Tuning Data Pipeline

A comprehensive, modular data pipeline for preparing datasets for fine-tuning Large Language Models, with S3 integration for scalable cloud storage.

## Features

- **Complete Data Processing Pipeline**: From raw data to tokenized, split datasets ready for fine-tuning
- **Cloud Integration**: Built-in S3 support for storing and retrieving datasets
- **Modular Design**: Process datasets with a single command or step-by-step
- **Reproducible**: Uses DVC for tracking data versions and pipeline stages
- **Extensible**: Easy to add new dataset converters or processing steps

## Project Structure

```
.
├── configs/                 # Configuration files
│   ├── data/                # Data processing configs
│   └── training/            # Training configs
├── data/                    # Data directory
│   ├── processed/           # Processed and tokenized data
│   ├── raw/                 # Raw data files
│   └── validation/          # Validated data
├── dvc.yaml                 # DVC pipeline configuration
├── scripts/                 # Utility scripts
│   ├── dataset_converters/  # Dataset-specific converters
│   │   └── servicenow_converter.py
│   └── utils/               # Utility scripts
│       └── setup_dvc.py
├── src/                     # Source code
│   ├── cli/                 # Command-line interfaces
│   │   └── data_cli.py
│   ├── cloud/               # Cloud integration (AWS/S3)
│   │   ├── auth.py
│   │   └── storage.py
│   └── data/                # Data processing pipeline
│       └── pipeline.py
└── tests/                   # Test suite
    ├── test_cloud.py
    └── test_pipeline.py
```

## Getting Started

### Prerequisites

- Python 3.9+
- AWS account with S3 access
- Required Python packages (see Installation)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-finetuning-pipeline.git
   cd llm-finetuning-pipeline
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

## Usage

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

## Pipeline Stages

1. **Download**: Download or prepare the raw dataset
2. **Convert**: Convert to a standardized format with user/assistant messages
3. **Preprocess**: Clean and normalize the text data
4. **Validate**: Check data quality and format
5. **Tokenize**: Tokenize data using the target model's tokenizer
6. **Split**: Divide into train/validation/test sets

## Testing

Run the test suite with:

```bash
python -m pytest
```

## License


## Acknowledgments

- ServiceNow for the example dataset
- Qwen for the tokenizer used in this example
