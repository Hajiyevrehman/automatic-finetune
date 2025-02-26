# src/data/preprocess.py
import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_text(text, lowercase=True, remove_special_chars=False):
    """
    Clean text by applying various normalization techniques.

    Args:
        text (str): Input text
        lowercase (bool): Whether to convert text to lowercase
        remove_special_chars (bool): Whether to remove special characters

    Returns:
        str: Cleaned text
    """
    if text is None:
        return ""

    # Convert to lowercase if specified
    if lowercase:
        text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove special characters if specified
    if remove_special_chars:
        text = re.sub(r"[^\w\s]", "", text)

    return text


def preprocess_qa_pair(qa_pair, config):
    """
    Preprocess a single Q&A pair.

    Args:
        qa_pair (dict): Input Q&A pair
        config (dict): Configuration for preprocessing

    Returns:
        dict: Preprocessed Q&A pair
    """
    preprocessing_config = config.get("preprocessing", {})

    # Extract normalization settings
    text_norm = preprocessing_config.get("text_normalization", {})
    lowercase = text_norm.get("lowercase", True)
    remove_special_chars = text_norm.get("remove_special_chars", False)

    # Clean the question and answer
    cleaned_question = clean_text(
        qa_pair.get("question", ""),
        lowercase=lowercase,
        remove_special_chars=remove_special_chars,
    )

    cleaned_answer = clean_text(
        qa_pair.get("answer", ""),
        lowercase=lowercase,
        remove_special_chars=False,  # Always preserve special chars in answers
    )

    # Create preprocessed Q&A pair
    processed_pair = {"question": cleaned_question, "answer": cleaned_answer}

    # Preserve metadata if present
    for key in qa_pair:
        if key not in ["question", "answer"]:
            processed_pair[key] = qa_pair[key]

    return processed_pair


def preprocess_dataset(input_path, output_path, config_path):
    """
    Preprocess a dataset of Q&A pairs.

    Args:
        input_path (str): Path to input JSON file
        output_path (str): Path to output JSON file
        config_path (str): Path to configuration file
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info(f"Preprocessing dataset from {input_path}")

    # Load the dataset
    with open(input_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Preprocess each Q&A pair
    processed_pairs = [preprocess_qa_pair(pair, config) for pair in qa_pairs]

    logger.info(f"Preprocessed {len(processed_pairs)} Q&A pairs")

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the preprocessed dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved preprocessed dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Q&A dataset")
    parser.add_argument("input", type=str, help="Path to input JSON file")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/data_processing.json",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
