#!/usr/bin/env python3
"""
Test AWS authentication and S3 access.
"""

import os

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_aws_auth():
    """Test AWS authentication and S3 bucket creation."""

    print("Testing AWS authentication...")

    # Get credentials from environment
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    if not aws_access_key or not aws_secret_key:
        print("ERROR: AWS credentials not found in .env file")
        return False

    if not bucket_name:
        print("ERROR: S3_BUCKET_NAME not found in .env file")
        return False

    # Create session
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region,
        )

        # Get S3 client
        s3_client = session.client("s3")

        # List buckets to test authentication
        response = s3_client.list_buckets()
        print(f"Authentication successful! Found {len(response['Buckets'])} buckets")

        # Check if our bucket exists
        bucket_exists = False
        for bucket in response["Buckets"]:
            if bucket["Name"] == bucket_name:
                bucket_exists = True
                print(f"Bucket '{bucket_name}' already exists")
                break

        # Create bucket if it doesn't exist
        if not bucket_exists:
            print(f"Creating bucket '{bucket_name}'...")

            if aws_region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": aws_region},
                )
            print(f"Bucket '{bucket_name}' created successfully!")

        # Test upload
        print("Testing upload to bucket...")
        s3_client.put_object(
            Bucket=bucket_name, Key="test/hello.txt", Body="Hello, S3!"
        )
        print("Test upload successful!")

        return True

    except ClientError as e:
        print(f"ERROR: {e}")
        return False

    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_aws_auth()
    if success:
        print("\n✅ AWS authentication test passed!")
    else:
        print("\n❌ AWS authentication test failed!")
