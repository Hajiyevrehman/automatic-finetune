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

    def upload_directory(
        self, local_dir: Union[str, Path], s3_prefix: str = ""
    ) -> bool:
        """
        Upload an entire local directory to S3.

        Args:
            local_dir (str or Path): Local directory path
            s3_prefix (str): S3 prefix to prepend to all uploaded files

        Returns:
            bool: True if all uploads successful, False if any failed
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            logger.error(f"{local_dir} is not a directory")
            return False

        all_successful = True
        for path in local_dir.rglob("*"):
            if path.is_file():
                relative_path = path.relative_to(local_dir)
                s3_key = (
                    f"{s3_prefix}/{relative_path}" if s3_prefix else str(relative_path)
                )

                # Normalize Windows paths if needed
                s3_key = s3_key.replace("\\", "/")

                if not self.upload_file(path, s3_key):
                    all_successful = False

        return all_successful

    def download_directory(self, s3_prefix: str, local_dir: Union[str, Path]) -> bool:
        """
        Download all files with a given prefix from S3 to a local directory.

        Args:
            s3_prefix (str): S3 prefix to filter objects by
            local_dir (str or Path): Local directory to download files to

        Returns:
            bool: True if all downloads successful, False if any failed
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Normalize prefix to not start with / but end with / if not empty
            if s3_prefix and not s3_prefix.endswith("/"):
                s3_prefix = f"{s3_prefix}/"

            all_successful = True
            for obj in self.bucket.objects.filter(Prefix=s3_prefix):
                # Skip "directory" objects (objects ending with /)
                if obj.key.endswith("/"):
                    continue

                # Determine the relative path from the prefix
                if s3_prefix:
                    relative_path = obj.key[len(s3_prefix) :]
                else:
                    relative_path = obj.key

                local_path = local_dir / relative_path

                # Create subdirectories if needed
                local_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    self.s3_client.download_file(
                        self.bucket_name, obj.key, str(local_path)
                    )
                    logger.info(
                        f"Downloaded s3://{self.bucket_name}/{obj.key} to {local_path}"
                    )
                except Exception as e:
                    logger.error(f"Failed to download {obj.key}: {str(e)}")
                    all_successful = False

            return all_successful

        except Exception as e:
            logger.error(f"Error downloading directory from S3: {str(e)}")
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

    def delete_object(self, s3_key: str) -> bool:
        """
        Delete an object from the S3 bucket.

        Args:
            s3_key (str): S3 key of the object to delete

        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {s3_key} from S3: {str(e)}")
            return False

    def delete_objects(self, s3_keys: List[str]) -> bool:
        """
        Delete multiple objects from the S3 bucket.

        Args:
            s3_keys (List[str]): List of S3 keys to delete

        Returns:
            bool: True if all deletions successful, False otherwise
        """
        if not s3_keys:
            return True

        try:
            # S3 delete_objects API requires a specific format
            objects = [{"Key": key} for key in s3_keys]

            self.s3_client.delete_objects(
                Bucket=self.bucket_name, Delete={"Objects": objects}
            )

            logger.info(
                f"Deleted {len(s3_keys)} objects from S3 bucket {self.bucket_name}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete objects from S3: {str(e)}")
            return False

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

    def get_object_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an S3 object.

        Args:
            s3_key (str): S3 key of the object

        Returns:
            Dict[str, Any]: Object metadata, or None if retrieval failed
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                "size": response.get("ContentLength", 0),
                "last_modified": response.get("LastModified"),
                "content_type": response.get("ContentType"),
                "metadata": response.get("Metadata", {}),
                "etag": response.get("ETag", "").strip('"'),
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {s3_key}: {str(e)}")
            return None
