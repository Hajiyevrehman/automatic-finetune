"""
Tests for the cloud functionality.

This module contains tests for the AWS authentication and S3 storage classes.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

from src.cloud.auth import (
    create_aws_session,
    create_bucket_if_not_exists,
    get_aws_credentials,
    get_s3_client,
    get_s3_resource,
    validate_s3_connection,
)
from src.cloud.storage import S3Storage


class TestAwsAuth(unittest.TestCase):
    """Test cases for AWS authentication functions."""

    @mock.patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_REGION": "us-test-1",
        },
    )
    def test_get_aws_credentials(self):
        """Test getting AWS credentials from environment variables."""
        key, secret, region = get_aws_credentials()
        self.assertEqual(key, "test-key")
        self.assertEqual(secret, "test-secret")
        self.assertEqual(region, "us-test-1")

    @mock.patch.dict(
        os.environ,
        {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"},
    )
    def test_get_aws_credentials_default_region(self):
        """Test getting AWS credentials with default region."""
        key, secret, region = get_aws_credentials()
        self.assertEqual(key, "test-key")
        self.assertEqual(secret, "test-secret")
        self.assertEqual(region, "us-east-1")

    @mock.patch.dict(
        os.environ, {"AWS_SECRET_ACCESS_KEY": "test-secret"}, clear=True
    )  # Add clear=True to remove all other env vars
    def test_get_aws_credentials_missing_key(self):
        """Test missing AWS access key."""
        with self.assertRaises(ValueError):
            get_aws_credentials()

    @mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "test-key"}, clear=True)
    def test_get_aws_credentials_missing_secret(self):
        """Test missing AWS secret key."""
        with self.assertRaises(ValueError):
            get_aws_credentials()

    @mock.patch("src.cloud.auth.boto3.Session")
    @mock.patch("src.cloud.auth.get_aws_credentials")
    def test_create_aws_session(self, mock_get_creds, mock_session):
        """Test creating an AWS session."""
        mock_get_creds.return_value = ("test-key", "test-secret", "us-test-1")
        mock_session_instance = mock.MagicMock()
        mock_session.return_value = mock_session_instance

        session = create_aws_session()

        mock_session.assert_called_once_with(
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-test-1",
        )
        self.assertEqual(session, mock_session_instance)

    @mock.patch("src.cloud.auth.create_aws_session")
    def test_get_s3_client(self, mock_create_session):
        """Test getting an S3 client."""
        mock_session = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_create_session.return_value = mock_session

        s3_client = get_s3_client()

        mock_create_session.assert_called_once()
        mock_session.client.assert_called_once_with("s3")
        self.assertEqual(s3_client, mock_s3_client)

    @mock.patch("src.cloud.auth.create_aws_session")
    def test_get_s3_resource(self, mock_create_session):
        """Test getting an S3 resource."""
        mock_session = mock.MagicMock()
        mock_s3_resource = mock.MagicMock()
        mock_session.resource.return_value = mock_s3_resource
        mock_create_session.return_value = mock_session

        s3_resource = get_s3_resource()

        mock_create_session.assert_called_once()
        mock_session.resource.assert_called_once_with("s3")
        self.assertEqual(s3_resource, mock_s3_resource)

    @mock.patch("src.cloud.auth.get_s3_client")
    def test_validate_s3_connection_success(self, mock_get_client):
        """Test validating S3 connection (success case)."""
        mock_client = mock.MagicMock()
        mock_get_client.return_value = mock_client

        result = validate_s3_connection("test-bucket")

        mock_get_client.assert_called_once()
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        self.assertTrue(result)

    @mock.patch("src.cloud.auth.get_s3_client")
    def test_validate_s3_connection_failure(self, mock_get_client):
        """Test validating S3 connection (failure case)."""
        mock_client = mock.MagicMock()
        mock_client.head_bucket.side_effect = Exception("Connection failed")
        mock_get_client.return_value = mock_client

        result = validate_s3_connection("test-bucket")

        mock_get_client.assert_called_once()
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        self.assertFalse(result)

    @mock.patch("src.cloud.auth.get_s3_client")
    @mock.patch("src.cloud.auth.get_aws_credentials")
    def test_create_bucket_if_not_exists_already_exists(
        self, mock_get_creds, mock_get_client
    ):
        """Test creating a bucket that already exists."""
        mock_get_creds.return_value = ("test-key", "test-secret", "us-test-1")
        mock_client = mock.MagicMock()
        mock_get_client.return_value = mock_client

        result = create_bucket_if_not_exists("test-bucket")

        mock_get_client.assert_called_once()
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_client.create_bucket.assert_not_called()
        self.assertTrue(result)

    @mock.patch("src.cloud.auth.get_s3_client")
    @mock.patch("src.cloud.auth.get_aws_credentials")
    def test_create_bucket_if_not_exists_new_bucket(
        self, mock_get_creds, mock_get_client
    ):
        """Test creating a new bucket."""
        from botocore.exceptions import ClientError

        mock_get_creds.return_value = ("test-key", "test-secret", "us-test-1")
        mock_client = mock.MagicMock()

        # First call to head_bucket raises ClientError with 404 code
        mock_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        mock_get_client.return_value = mock_client

        result = create_bucket_if_not_exists("test-bucket")

        mock_get_client.assert_called_once()
        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_client.create_bucket.assert_called_once()
        self.assertTrue(result)


class TestS3Storage(unittest.TestCase):
    """Test cases for S3Storage class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = mock.MagicMock()
        self.mock_resource = mock.MagicMock()
        self.mock_bucket = mock.MagicMock()
        self.mock_resource.Bucket.return_value = self.mock_bucket

        # Create a patcher for get_s3_client
        self.patcher_client = mock.patch("src.cloud.storage.get_s3_client")
        self.mock_get_client = self.patcher_client.start()
        self.mock_get_client.return_value = self.mock_client

        # Create a patcher for get_s3_resource
        self.patcher_resource = mock.patch("src.cloud.storage.get_s3_resource")
        self.mock_get_resource = self.patcher_resource.start()
        self.mock_get_resource.return_value = self.mock_resource

        # Create S3Storage instance
        self.s3_storage = S3Storage("test-bucket")

    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher_client.stop()
        self.patcher_resource.stop()

    def test_init(self):
        """Test initialization of S3Storage."""
        self.assertEqual(self.s3_storage.bucket_name, "test-bucket")
        self.assertEqual(self.s3_storage.s3_client, self.mock_client)
        self.assertEqual(self.s3_storage.s3_resource, self.mock_resource)
        self.assertEqual(self.s3_storage.bucket, self.mock_bucket)
        self.mock_resource.Bucket.assert_called_once_with("test-bucket")

    def test_upload_file_success(self):
        """Test uploading a file (success case)."""
        result = self.s3_storage.upload_file("local/path.txt", "s3/key.txt")

        self.mock_client.upload_file.assert_called_once_with(
            "local/path.txt", "test-bucket", "s3/key.txt"
        )
        self.assertTrue(result)

    def test_upload_file_failure(self):
        """Test uploading a file (failure case)."""
        self.mock_client.upload_file.side_effect = Exception("Upload failed")

        result = self.s3_storage.upload_file("local/path.txt", "s3/key.txt")

        self.mock_client.upload_file.assert_called_once_with(
            "local/path.txt", "test-bucket", "s3/key.txt"
        )
        self.assertFalse(result)

    def test_download_file_success(self):
        """Test downloading a file (success case)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, "download.txt")

            result = self.s3_storage.download_file("s3/key.txt", local_path)

            self.mock_client.download_file.assert_called_once_with(
                "test-bucket", "s3/key.txt", local_path
            )
            self.assertTrue(result)

    def test_download_file_failure(self):
        """Test downloading a file (failure case)."""
        self.mock_client.download_file.side_effect = Exception("Download failed")

        result = self.s3_storage.download_file("s3/key.txt", "local/path.txt")

        self.mock_client.download_file.assert_called_once_with(
            "test-bucket", "s3/key.txt", "local/path.txt"
        )
        self.assertFalse(result)

    def test_list_objects(self):
        """Test listing objects in the bucket."""
        # Mock the paginator
        mock_paginator = mock.MagicMock()
        mock_pages = mock.MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "test/file1.txt"},
                    {"Key": "test/file2.json"},
                    {"Key": "other/file3.txt"},
                ]
            }
        ]
        self.mock_client.get_paginator.return_value = mock_paginator

        # Test without filters
        result = self.s3_storage.list_objects()

        self.mock_client.get_paginator.assert_called_with("list_objects_v2")
        mock_paginator.paginate.assert_called_with(Bucket="test-bucket", Prefix="")
        self.assertEqual(
            result, ["test/file1.txt", "test/file2.json", "other/file3.txt"]
        )

        # Test with prefix filter
        result = self.s3_storage.list_objects(prefix="test/")

        mock_paginator.paginate.assert_called_with(Bucket="test-bucket", Prefix="test/")

        # Test with suffix filter
        result = self.s3_storage.list_objects(suffix=".txt")

        self.assertEqual(result, ["test/file1.txt", "other/file3.txt"])

    def test_object_exists_true(self):
        """Test checking if an object exists (exists case)."""
        result = self.s3_storage.object_exists("s3/key.txt")

        self.mock_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="s3/key.txt"
        )
        self.assertTrue(result)

    def test_object_exists_false(self):
        """Test checking if an object exists (doesn't exist case)."""
        from botocore.exceptions import ClientError

        self.mock_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )

        result = self.s3_storage.object_exists("s3/key.txt")

        self.mock_client.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="s3/key.txt"
        )
        self.assertFalse(result)

    def test_upload_json_success(self):
        """Test uploading JSON data (success case)."""
        data = {"key": "value"}

        result = self.s3_storage.upload_json(data, "s3/data.json")

        self.mock_client.put_object.assert_called_once()
        args, kwargs = self.mock_client.put_object.call_args
        self.assertEqual(kwargs["Bucket"], "test-bucket")
        self.assertEqual(kwargs["Key"], "s3/data.json")
        self.assertEqual(kwargs["ContentType"], "application/json")
        # Check that the body is valid JSON
        json.loads(kwargs["Body"])
        self.assertTrue(result)

    def test_upload_json_failure(self):
        """Test uploading JSON data (failure case)."""
        self.mock_client.put_object.side_effect = Exception("Upload failed")

        result = self.s3_storage.upload_json({"key": "value"}, "s3/data.json")

        self.assertFalse(result)

    def test_download_json_success(self):
        """Test downloading JSON data (success case)."""
        mock_response = {"Body": mock.MagicMock()}
        mock_response["Body"].read.return_value = json.dumps({"key": "value"}).encode(
            "utf-8"
        )
        self.mock_client.get_object.return_value = mock_response

        result = self.s3_storage.download_json("s3/data.json")

        self.mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="s3/data.json"
        )
        self.assertEqual(result, {"key": "value"})

    def test_download_json_failure(self):
        """Test downloading JSON data (failure case)."""
        self.mock_client.get_object.side_effect = Exception("Download failed")

        result = self.s3_storage.download_json("s3/data.json")

        self.mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="s3/data.json"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
