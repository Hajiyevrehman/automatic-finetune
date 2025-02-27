"""
AWS authentication module for cloud services.
Handles credential loading and session creation for S3 and other AWS services.
"""

import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()


def get_aws_credentials() -> tuple:
    """
    Get AWS credentials from environment variables.

    Returns:
        tuple: (aws_access_key_id, aws_secret_access_key, aws_region)

    Raises:
        ValueError: If required credentials are missing
    """
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get(
        "AWS_REGION", "us-east-1"
    )  # Default to us-east-1 if not specified

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )

    return aws_access_key_id, aws_secret_access_key, aws_region


def create_aws_session() -> boto3.Session:
    """
    Create an authenticated AWS session.

    Returns:
        boto3.Session: Authenticated AWS session
    """
    try:
        aws_access_key_id, aws_secret_access_key, aws_region = get_aws_credentials()

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        logger.info(f"AWS session created successfully in region {aws_region}")
        return session

    except Exception as e:
        logger.error(f"Failed to create AWS session: {str(e)}")
        raise


def get_s3_client() -> boto3.client:
    """
    Get an authenticated S3 client.

    Returns:
        boto3.client: Authenticated S3 client
    """
    session = create_aws_session()
    return session.client("s3")


def get_s3_resource() -> boto3.resource:
    """
    Get an authenticated S3 resource.

    Returns:
        boto3.resource: Authenticated S3 resource
    """
    session = create_aws_session()
    return session.resource("s3")


def validate_s3_connection(bucket_name: str) -> bool:
    """
    Validate S3 connection by checking if a bucket exists and is accessible.

    Args:
        bucket_name (str): Name of the S3 bucket to check

    Returns:
        bool: True if connection is valid and bucket is accessible, False otherwise
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
        return True

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            logger.error(f"Bucket {bucket_name} does not exist")
        elif error_code == "403":
            logger.error(f"Access denied to bucket {bucket_name}. Check permissions")
        else:
            logger.error(f"Error accessing bucket {bucket_name}: {str(e)}")
        return False

    except Exception as e:
        logger.error(f"Error validating S3 connection: {str(e)}")
        return False


def create_bucket_if_not_exists(bucket_name: str, region: Optional[str] = None) -> bool:
    """
    Create an S3 bucket if it doesn't already exist.

    Args:
        bucket_name (str): Name of the bucket to create
        region (str, optional): AWS region to create the bucket in.
                               Defaults to the region from credentials.

    Returns:
        bool: True if bucket exists or was created successfully, False otherwise
    """
    s3_client = get_s3_client()
    _, _, default_region = get_aws_credentials()
    region = region or default_region

    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} already exists")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code != "404":
                logger.error(f"Error checking bucket existence: {str(e)}")
                return False

        # Create the bucket
        if region == "us-east-1":
            # us-east-1 is the default region and requires different syntax
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {"LocationConstraint": region}
            s3_client.create_bucket(
                Bucket=bucket_name, CreateBucketConfiguration=location
            )

        logger.info(f"Bucket {bucket_name} created successfully in region {region}")
        return True

    except Exception as e:
        logger.error(f"Failed to create bucket {bucket_name}: {str(e)}")
        return False
