# Cloud Training Guide

This guide explains how to set up and run the LLM fine-tuning pipeline on AWS GPU instances.

## Overview

The cloud training pipeline allows you to train large language models on AWS GPU instances. The system:

1. Provisions a GPU instance (spot or on-demand)
2. Attaches a persistent EBS volume for model storage
3. Deploys a Docker container with all dependencies
4. Runs the training process with optimizations for GPU
5. Automatically uploads model checkpoints to S3
6. Shuts down the instance when training completes or becomes idle

## Prerequisites

Before you begin, you'll need:

1. AWS CLI installed and configured with appropriate credentials
2. Terraform installed (version 1.0+)
3. Docker installed and configured
4. A Docker Hub account (or other container registry)
5. An SSH key pair for AWS EC2 access

## Setup Instructions

### 1. Configure AWS Credentials

Make sure your AWS credentials are set up:

```bash
aws configure
```

### 2. Update Configuration Files

1. Update the Docker image name in `terraform/compute/main.tf` to use your Docker Hub username
2. Update GitHub repository URL in the user data script if you're hosting your code elsewhere
3. Review and modify `configs/training/cloud_training_config.yaml` as needed

### 3. Build and Deploy

Run the deployment script:

```bash
python scripts/deploy_to_aws.py \
  --bucket YOUR_BUCKET_NAME \
  --instance-type g4dn.xlarge \
  --region us-east-1 \
  --key-name YOUR_SSH_KEY
```

Options:
- `--bucket`: S3 bucket name for data and model storage
- `--instance-type`: EC2 instance type (g4dn.xlarge, p3.2xlarge, etc.)
- `--region`: AWS region
- `--key-name`: EC2 SSH key pair name
- `--no-spot`: Use on-demand instance instead of spot
- `--no-docker`: Skip Docker image build/push

### 4. Monitor Training

1. SSH into the EC2 instance using the output command
2. View training logs:
   ```bash
   docker logs -f training
   ```
3. Access MLflow UI at http://INSTANCE_IP:5000

### 5. Retrieve Trained Models

Models are automatically saved to S3. You can download them with:

```bash
aws s3 cp s3://YOUR_BUCKET_NAME/models/qwen2.5-7b-lora/ ./models/ --recursive
```

### 6. Clean Up Resources

When you're done, clean up the AWS resources:

```bash
python scripts/deploy_to_aws.py --bucket YOUR_BUCKET_NAME --destroy
```

## Advanced Configuration

### GPU Instance Selection

Choose an instance type based on your model size and training requirements:

- **g4dn.xlarge**: 1 GPU, 4 vCPUs, 16 GB RAM - Good for 7B models with QLoRA
- **g4dn.2xlarge**: 1 GPU, 8 vCPUs, 32 GB RAM - Better for 7B models
- **p3.2xlarge**: 1 GPU (V100), 8 vCPUs, 61 GB RAM - Good for 7B-13B models
- **p3.8xlarge**: 4 GPUs (V100), 32 vCPUs, 244 GB RAM - Good for larger models

### Training Optimizations

The default configuration uses these optimizations:

1. **QLoRA**: 4-bit quantization to reduce memory usage
2. **DeepSpeed Zero-2**: Optimizer state sharding and CPU offloading
3. **Mixed Precision**: BF16 for faster computation
4. **Flash Attention**: Optimized attention implementation
5. **Gradient Accumulation**: To handle larger effective batch sizes

You can adjust these in the configuration files.

### Idle Detection

The training script includes automatic idle detection and shutdown:

- Monitors CPU, GPU, and memory usage
- Shuts down the instance after 30 minutes of inactivity
- Saves all progress to S3 before shutdown

This ensures you don't waste money on idle instances.

## Troubleshooting

### Docker Container Issues

If the container fails to start:

1. SSH into the instance
2. Check Docker logs: `docker logs training`
3. Try running the container manually:
   ```bash
   docker run --gpus all -it yourusername/llm-finetuning:latest /bin/bash
   ```

### Out of Memory Errors

If you encounter OOM errors:

1. Decrease `per_device_train_batch_size` or increase `gradient_accumulation_steps`
2. Enable optimizer offloading in DeepSpeed config
3. Try a larger instance type

### S3 Access Issues

If there are problems accessing S3:

1. Check IAM role permissions
2. Verify the S3 bucket exists and is accessible
3. Test S3 access from the instance: `aws s3 ls s3://YOUR_BUCKET_NAME/`

## Best Practices

1. **Use Spot Instances**: They're 70-90% cheaper than on-demand
2. **Save Checkpoints Frequently**: Set `save_steps` to a reasonable value
3. **Monitor Costs**: Set up AWS budget alerts
4. **Start Small**: Test with a small subset of data first
5. **Use Persistent Storage**: The attached EBS volume persists data between spot terminations
