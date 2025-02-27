#!/usr/bin/env python
"""
Script to set up MLflow for experiment tracking.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_mlflow():
    """Check if MLflow is installed."""
    try:
        import mlflow

        logger.info(f"MLflow version {mlflow.__version__} is installed")
        return True
    except ImportError:
        logger.error(
            "MLflow is not installed. Please install it with 'pip install mlflow'"
        )
        return False


def setup_mlflow_local(tracking_uri=None, artifacts_uri=None):
    """
    Set up MLflow locally.

    Args:
        tracking_uri: URI for MLflow tracking server
        artifacts_uri: URI for MLflow artifacts storage
    """
    import mlflow

    # Create directories if they don't exist
    os.makedirs("mlflow", exist_ok=True)

    # Set tracking URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_uri = "sqlite:///mlflow/mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)

    # Set artifacts URI
    if artifacts_uri:
        os.environ["MLFLOW_ARTIFACT_ROOT"] = artifacts_uri
    else:
        artifacts_uri = os.path.abspath("mlflow/artifacts")
        os.makedirs(artifacts_uri, exist_ok=True)
        os.environ["MLFLOW_ARTIFACT_ROOT"] = artifacts_uri

    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow artifact URI: {os.environ.get('MLFLOW_ARTIFACT_ROOT')}")

    return tracking_uri, artifacts_uri


def setup_mlflow_s3(bucket_name, s3_prefix="mlflow", aws_region=None):
    """
    Set up MLflow with S3 backend for artifacts.

    Args:
        bucket_name: S3 bucket name
        s3_prefix: Prefix for MLflow artifacts in S3 bucket
        aws_region: AWS region
    """
    import mlflow
    from src.cloud.auth import create_bucket_if_not_exists, get_aws_credentials

    # Ensure bucket exists
    _, _, default_region = get_aws_credentials()
    region = aws_region or default_region

    if not create_bucket_if_not_exists(bucket_name, region):
        logger.error(f"Failed to create or access bucket {bucket_name}")
        return None, None

    # Set up tracking locally but artifacts in S3
    tracking_uri = "sqlite:///mlflow/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    # Format the S3 URI for artifacts
    artifacts_uri = f"s3://{bucket_name}/{s3_prefix}"
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifacts_uri

    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"MLflow artifact URI: {artifacts_uri}")

    return tracking_uri, artifacts_uri


def start_mlflow_server(host="127.0.0.1", port=5000, tracking_uri=None):
    """
    Start the MLflow tracking server.

    Args:
        host: Host to bind to
        port: Port to bind to
        tracking_uri: URI for MLflow tracking server
    """
    if tracking_uri is None:
        tracking_uri = "sqlite:///mlflow/mlflow.db"

    cmd = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        tracking_uri,
    ]

    # Add artifact root if specified
    if "MLFLOW_ARTIFACT_ROOT" in os.environ:
        cmd.extend(["--default-artifact-root", os.environ["MLFLOW_ARTIFACT_ROOT"]])

    logger.info(f"Starting MLflow server with command: {' '.join(cmd)}")

    try:
        # Start the server process
        subprocess.Popen(cmd)
        logger.info(f"MLflow server started at http://{host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start MLflow server: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Set up MLflow for experiment tracking"
    )

    parser.add_argument("--bucket", type=str, help="S3 bucket name for artifacts")
    parser.add_argument("--region", type=str, help="AWS region for S3 bucket")
    parser.add_argument(
        "--s3-prefix", type=str, default="mlflow", help="S3 prefix for artifacts"
    )
    parser.add_argument(
        "--start-server", action="store_true", help="Start MLflow server"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for MLflow server"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow server")

    args = parser.parse_args()

    # Check MLflow installation
    if not check_mlflow():
        sys.exit(1)

    # Create mlflow directory if it doesn't exist
    os.makedirs("mlflow", exist_ok=True)

    # Set up MLflow
    if args.bucket:
        tracking_uri, artifacts_uri = setup_mlflow_s3(
            args.bucket, s3_prefix=args.s3_prefix, aws_region=args.region
        )
    else:
        tracking_uri, artifacts_uri = setup_mlflow_local()

    # Start server if requested
    if args.start_server:
        start_mlflow_server(args.host, args.port, tracking_uri)


if __name__ == "__main__":
    main()
