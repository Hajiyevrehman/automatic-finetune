# src/data/validate.py
import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_qa_pair(
    qa_pair,
    min_question_length=5,
    max_question_length=1000,
    min_answer_length=5,
    max_answer_length=10000,
):
    """
    Validate a Q&A pair.

    Args:
        qa_pair (dict): Q&A pair to validate
        min_question_length (int): Minimum question length
        max_question_length (int): Maximum question length
        min_answer_length (int): Minimum answer length
        max_answer_length (int): Maximum answer length

    Returns:
        tuple: (is_valid, issues)
    """
    issues = []

    # Check if required fields are present
    if "question" not in qa_pair:
        issues.append("Missing 'question' field")

    if "answer" not in qa_pair:
        issues.append("Missing 'answer' field")

    # If any required field is missing, return early
    if issues:
        return False, issues

    # Check question length
    question_length = len(qa_pair["question"])
    if question_length < min_question_length:
        issues.append(f"Question too short ({question_length} chars)")

    if question_length > max_question_length:
        issues.append(f"Question too long ({question_length} chars)")

    # Check answer length
    answer_length = len(qa_pair["answer"])
    if answer_length < min_answer_length:
        issues.append(f"Answer too short ({answer_length} chars)")

    if answer_length > max_answer_length:
        issues.append(f"Answer too long ({answer_length} chars)")

    # Validate tokenized data if present
    if "tokenized" in qa_pair:
        tokenized = qa_pair["tokenized"]

        # Check if tokenized data has required fields
        if "input_ids" not in tokenized:
            issues.append("Missing 'input_ids' in tokenized data")

        if "attention_mask" not in tokenized:
            issues.append("Missing 'attention_mask' in tokenized data")

        # Check if input_ids and attention_mask have the same length
        if "input_ids" in tokenized and "attention_mask" in tokenized:
            if len(tokenized["input_ids"]) != len(tokenized["attention_mask"]):
                issues.append("Mismatch in lengths of 'input_ids' and 'attention_mask'")

    return len(issues) == 0, issues


def validate_dataset(input_path, output_path=None, verbose=True):
    """
    Validate a dataset of Q&A pairs.

    Args:
        input_path (str): Path to input JSON file
        output_path (str): Optional path to save valid data
        verbose (bool): Whether to print detailed information

    Returns:
        dict: Validation statistics
    """
    logger.info(f"Validating dataset from {input_path}")

    # Load the dataset
    with open(input_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Validate each Q&A pair
    valid_count = 0
    invalid_count = 0
    valid_pairs = []
    issue_counts = {}

    for i, pair in enumerate(tqdm(qa_pairs, desc="Validating")):
        is_valid, issues = validate_qa_pair(pair)

        if is_valid:
            valid_count += 1
            valid_pairs.append(pair)
        else:
            invalid_count += 1

            if verbose:
                logger.warning(f"Sample {i} is invalid: {issues}")

            # Count issues
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

    # Calculate statistics
    total_count = valid_count + invalid_count
    valid_percentage = (valid_count / total_count * 100) if total_count > 0 else 0

    stats = {
        "total_samples": total_count,
        "valid_samples": valid_count,
        "invalid_samples": invalid_count,
        "valid_percentage": valid_percentage,
        "issue_counts": issue_counts,
    }

    # Log statistics
    logger.info(
        f"Validation completed: {valid_count}/{total_count} ({valid_percentage:.2f}%) samples are valid"
    )

    if issue_counts:
        logger.info("Common issues:")
        for issue, count in sorted(
            issue_counts.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"- {issue}: {count} samples")

    # Save valid data if output path is provided
    if output_path and valid_pairs:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(valid_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(valid_pairs)} valid samples to {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate Q&A dataset")
    parser.add_argument("input", type=str, help="Path to input JSON file")
    parser.add_argument("--output", type=str, help="Path to save valid data")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    args = parser.parse_args()

    validate_dataset(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
