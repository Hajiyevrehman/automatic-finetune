"""
S3 storage utilities for data pipeline operations.
Provides functionality to upload, download, and manage files in S3 buckets.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError

from src.cloud.auth import get_s3_client, get_s3_resource

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class S3Storage:
    """
    S3 storage class for managing data in S3 buckets.
    """

    def __init__(self, bucket_name: str):
        """
        Initialize S3Storage with a bucket name.

        Args:
            bucket_name (str): Name of the S3 bucket to use
        """
        self.bucket_name = bucket_name
        self.s3_client = get_s3_client()
        self.s3_resource = get_s3_resource()
        self.bucket = self.s3_resource.Bucket(bucket_name)

    def upload_file(self, local_path: Union[str, Path], s3_key: str) -> bool:
        """
        Upload a file to S3.

        Args:
            local_path (str or Path): Local file path
            s3_key (str): Destination S3 key (path within bucket)

        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            local_path = str(local_path)  # Convert Path to string if needed
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {str(e)}")
            return False

    def download_file(self, s3_key: str, local_path: Union[str, Path]) -> bool:
        """
        Download a file from S3.

        Args:
            s3_key (str): S3 key (path within bucket)
            local_path (str or Path): Local destination path

        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            local_path = str(local_path)  # Convert Path to string if needed

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_key} from S3: {str(e)}")
            return False

    def list_objects(self, prefix: str = "", suffix: str = "") -> List[str]:
        """
        List objects in the S3 bucket, optionally filtering by prefix and suffix.

        Args:
            prefix (str): Prefix to filter objects by
            suffix (str): Suffix to filter objects by

        Returns:
            List[str]: List of matching object keys
        """
        try:
            result = []
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if not suffix or key.endswith(suffix):
                            result.append(key)

            return result

        except Exception as e:
            logger.error(f"Error listing objects in S3: {str(e)}")
            return []

    def object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in the S3 bucket.

        Args:
            s3_key (str): S3 key to check

        Returns:
            bool: True if the object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def upload_json(self, data: Any, s3_key: str) -> bool:
        """
        Upload JSON data to S3.

        Args:
            data: Data to serialize as JSON
            s3_key (str): S3 key for the JSON file

        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            json_str = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Body=json_str,
                Bucket=self.bucket_name,
                Key=s3_key,
                ContentType="application/json",
            )
            logger.info(f"Uploaded JSON data to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload JSON data to S3: {str(e)}")
            return False

    def download_json(self, s3_key: str) -> Optional[Any]:
        """
        Download and parse JSON data from S3.

        Args:
            s3_key (str): S3 key of the JSON file

        Returns:
            Any: Parsed JSON data, or None if download or parsing failed
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            json_str = response["Body"].read().decode("utf-8")
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to download or parse JSON from S3: {str(e)}")
            return None
