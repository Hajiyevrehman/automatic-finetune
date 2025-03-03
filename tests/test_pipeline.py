"""
Tests for the data pipeline.

This module contains tests for the DataPipeline class and its methods.
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
import yaml

from src.data.pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Test cases for DataPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory and config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)

        # Create directory structure
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.validation_dir = self.data_root / "validation"

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)

        # Create test config file
        self.config = {
            "directories": {
                "raw": str(self.raw_dir),
                "processed": str(self.processed_dir),
                "validation": str(self.validation_dir),
            },
            "model": {"name": "test-model", "max_length": 512},
            "tokenizer": {"padding": "max_length", "truncation": True},
            "splitting": {"train_ratio": 0.8, "val_test_ratio": 0.5, "seed": 42},
        }

        self.config_path = self.data_root / "config.yaml"
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

        # Create test data
        self.test_qa_pairs = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
        ]

        # Write test data to file
        self.test_data_path = self.raw_dir / "test-dataset.json"
        with open(self.test_data_path, "w") as f:
            json.dump(self.test_qa_pairs, f)

        # Mock S3 integration
        self.patcher_bucket = mock.patch(
            "src.data.pipeline.create_bucket_if_not_exists"
        )
        self.mock_create_bucket = self.patcher_bucket.start()
        self.mock_create_bucket.return_value = True

        # Create DataPipeline instance with local config
        self.pipeline = DataPipeline(self.config_path)

    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher_bucket.stop()
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of DataPipeline."""
        self.assertEqual(self.pipeline.config_path, self.config_path)
        self.assertEqual(self.pipeline.model_name, "test-model")
        # Check that directories were created
        self.assertTrue(self.raw_dir.exists())
        self.assertTrue(self.processed_dir.exists())
        self.assertTrue(self.validation_dir.exists())

    def test_load_config_yaml(self):
        """Test loading YAML config."""
        config = self.pipeline._load_config()
        self.assertEqual(config, self.config)

    def test_load_config_json(self):
        """Test loading JSON config."""
        # Create a JSON config
        json_config_path = self.data_root / "config.json"
        with open(json_config_path, "w") as f:
            json.dump(self.config, f)

        # Create a pipeline with JSON config
        pipeline = DataPipeline(json_config_path)

        # Check config is loaded correctly
        self.assertEqual(pipeline.config, self.config)

    def test_load_config_not_found(self):
        """Test loading non-existent config."""
        non_existent_path = self.data_root / "non-existent.yaml"
        with self.assertRaises(FileNotFoundError):
            DataPipeline(non_existent_path)

    def test_get_path(self):
        """Test getting a path from configuration."""
        path = self.pipeline._get_path("raw", "test-file.json")
        expected_path = self.raw_dir / "test-file.json"
        self.assertEqual(path, expected_path)

    def test_convert_dataset(self):
        """Test converting a dataset."""
        result = self.pipeline.convert_dataset("test-dataset", "test-converted")

        # Check result
        self.assertTrue(result)

        # Check output file exists
        output_path = self.processed_dir / "test-converted.json"
        self.assertTrue(output_path.exists())

        # Check output content
        with open(output_path, "r") as f:
            converted_data = json.load(f)

        self.assertEqual(len(converted_data), len(self.test_qa_pairs))
        self.assertEqual(
            converted_data[0]["messages"][1]["content"],
            "What is the capital of France?",
        )
        self.assertEqual(converted_data[0]["messages"][2]["content"], "Paris")

    def test_preprocess_data(self):
        """Test preprocessing a dataset."""
        # First convert the dataset
        self.pipeline.convert_dataset("test-dataset", "test-converted")

        # Then preprocess it
        result = self.pipeline.preprocess_data("test-converted", "test-preprocessed")

        # Check result
        self.assertTrue(result)

        # Check output file exists
        output_path = self.processed_dir / "test-preprocessed.json"
        self.assertTrue(output_path.exists())

        # Check output content
        with open(output_path, "r") as f:
            preprocessed_data = json.load(f)

        self.assertEqual(len(preprocessed_data), len(self.test_qa_pairs))
        # Check normalization happened (if relevant)

    def test_validate_data(self):
        """Test validating a dataset."""
        # First convert and preprocess the dataset
        self.pipeline.convert_dataset("test-dataset", "test-converted")
        self.pipeline.preprocess_data("test-converted", "test-preprocessed")

        # Then validate it
        result = self.pipeline.validate_data("test-preprocessed", "test-validated")

        # Check result
        self.assertTrue(result)

        # Check output files exist
        output_path = self.validation_dir / "test-validated.json"
        stats_path = self.validation_dir / "test-validated_stats.json"
        self.assertTrue(output_path.exists())
        self.assertTrue(stats_path.exists())

        # Check output content
        with open(output_path, "r") as f:
            validated_data = json.load(f)

        with open(stats_path, "r") as f:
            validation_stats = json.load(f)

        self.assertEqual(len(validated_data), len(self.test_qa_pairs))
        self.assertEqual(validation_stats["total"], len(self.test_qa_pairs))
        self.assertEqual(validation_stats["valid"], len(self.test_qa_pairs))
        self.assertEqual(validation_stats["invalid"], 0)

    @mock.patch("transformers.AutoTokenizer")
    def test_tokenize_data(self, mock_tokenizer_class):
        """Test tokenizing a dataset."""
        # Set up mock tokenizer
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock tokenized result
        mock_tokenized_result = mock.MagicMock()
        mock_tokenized_result.tolist.return_value = [[1, 2, 3, 4, 5]]
        mock_tokenizer.apply_chat_template.return_value = mock_tokenized_result

        # First prepare the dataset
        self.pipeline.convert_dataset("test-dataset", "test-converted")
        self.pipeline.preprocess_data("test-converted", "test-preprocessed")
        self.pipeline.validate_data("test-preprocessed", "test-validated")

        # Then tokenize it
        result = self.pipeline.tokenize_data("test-validated", "test-tokenized")

        # Check result
        self.assertTrue(result)

        # Check output file exists
        output_path = self.processed_dir / "test-tokenized.pt"
        self.assertTrue(output_path.exists())

        # Check output content - this should now be a pickle file
        with open(output_path, "rb") as f:
            tokenized_data = pickle.load(f)

        self.assertEqual(len(tokenized_data), len(self.test_qa_pairs))
        self.assertEqual(tokenized_data[0]["input_ids"], [1, 2, 3, 4, 5])

        # Check tokenizer was called correctly
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            self.pipeline.model_name
        )
        self.assertEqual(
            mock_tokenizer.apply_chat_template.call_count, len(self.test_qa_pairs)
        )

    def test_split_dataset(self):
        """Test splitting a tokenized dataset."""
        # Create a mock tokenized dataset
        tokenized_data = [
            {"input_ids": [1, 2, 3], "original_messages": []},
            {"input_ids": [4, 5, 6], "original_messages": []},
            {"input_ids": [7, 8, 9], "original_messages": []},
            {"input_ids": [10, 11, 12], "original_messages": []},
            {"input_ids": [13, 14, 15], "original_messages": []},
        ]

        # Write tokenized data to file
        tokenized_path = self.processed_dir / "test-tokenized.pt"
        with open(tokenized_path, "wb") as f:
            pickle.dump(tokenized_data, f)

        # Split the dataset
        result = self.pipeline.split_dataset("test-tokenized", 0.6)

        # Check result
        self.assertTrue(result)

        # Check output files exist
        train_path = self.processed_dir / "test-tokenized_train.pt"
        val_path = self.processed_dir / "test-tokenized_val.pt"
        test_path = self.processed_dir / "test-tokenized_test.pt"
        self.assertTrue(train_path.exists())
        self.assertTrue(val_path.exists())
        self.assertTrue(test_path.exists())

        # Check output content
        with open(train_path, "rb") as f:
            train_data = pickle.load(f)
        with open(val_path, "rb") as f:
            val_data = pickle.load(f)
        with open(test_path, "rb") as f:
            test_data = pickle.load(f)

        self.assertEqual(len(train_data), 3)  # 60% of 5
        self.assertEqual(len(val_data), 1)  # 20% of 5
        self.assertEqual(len(test_data), 1)  # 20% of 5

    def test_run_pipeline(self):
        """Test running the complete pipeline."""
        # Mock tokenization to avoid needing actual transformers
        with mock.patch.object(
            self.pipeline, "tokenize_data", return_value=True
        ) as mock_tokenize:
            with mock.patch.object(
                self.pipeline, "split_dataset", return_value=True
            ) as mock_split:
                result = self.pipeline.run_pipeline("test-dataset")

                # Check result
                self.assertTrue(result)

                # Check all steps were called
                mock_tokenize.assert_called_once()
                mock_split.assert_called_once()

                # Check intermediate files exist
                converted_path = self.processed_dir / "test-dataset_converted.json"
                preprocessed_path = (
                    self.processed_dir / "test-dataset_preprocessed.json"
                )
                validated_path = self.validation_dir / "test-dataset_validated.json"
                self.assertTrue(converted_path.exists())
                self.assertTrue(preprocessed_path.exists())
                self.assertTrue(validated_path.exists())

    def test_run_pipeline_failure(self):
        """Test running the pipeline with a failure."""
        # Mock a step to fail
        with mock.patch.object(
            self.pipeline, "preprocess_data", return_value=False
        ) as mock_preprocess:
            result = self.pipeline.run_pipeline("test-dataset")

            # Check result
            self.assertFalse(result)

            # Check later steps weren't called
            # Use manual inspection of logs to verify this

    @mock.patch("src.data.pipeline.S3Storage")
    def test_s3_integration(self, mock_s3_storage_class):
        """Test integration with S3 storage."""
        # Create a config with S3 settings
        s3_config = self.config.copy()
        s3_config["s3"] = {"default_bucket": "test-bucket"}

        s3_config_path = self.data_root / "s3_config.yaml"
        with open(s3_config_path, "w") as f:
            yaml.dump(s3_config, f)

        # Mock S3Storage instance
        mock_s3_storage = mock.MagicMock()
        mock_s3_storage_class.return_value = mock_s3_storage
        mock_s3_storage.download_file.return_value = True
        mock_s3_storage.upload_file.return_value = True
        mock_s3_storage.object_exists.return_value = False

        # Create pipeline with S3 config
        pipeline = DataPipeline(s3_config_path)

        # Check S3 integration was set up
        self.assertEqual(pipeline.s3, mock_s3_storage)
        mock_s3_storage_class.assert_called_once_with("test-bucket")

        # Test that _download_if_needed and _upload_if_needed work
        # Create a test file
        test_file_path = self.raw_dir / "s3_test.json"
        with open(test_file_path, "w") as f:
            json.dump({"test": "data"}, f)

        # Test _upload_if_needed
        result = pipeline._upload_if_needed(test_file_path, "raw", "s3_test.json")
        self.assertTrue(result)
        mock_s3_storage.upload_file.assert_called_with(
            test_file_path, "data/raw/s3_test.json"
        )

        # Test _download_if_needed
        # Case 1: File exists locally
        result = pipeline._download_if_needed("raw", "s3_test.json")
        self.assertEqual(result, test_file_path)
        mock_s3_storage.download_file.assert_not_called()

        # Case 2: File doesn't exist locally but exists in S3
        nonexistent_file = "nonexistent.json"
        mock_s3_storage.download_file.return_value = True
        result = pipeline._download_if_needed("raw", nonexistent_file)
        expected_path = self.raw_dir / nonexistent_file
        self.assertEqual(result, expected_path)
        mock_s3_storage.download_file.assert_called_with(
            f"data/raw/{nonexistent_file}", expected_path
        )


if __name__ == "__main__":
    unittest.main()
