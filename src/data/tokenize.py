# src/data/tokenize.py
import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_tokenizer(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Load the tokenizer for the specified Qwen2.5 model.

    Args:
        model_name (str): Name or path of the model

    Returns:
        tokenizer: Loaded tokenizer
    """
    logger.info(f"Loading tokenizer for {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        raise


def tokenize_qa_pair(tokenizer, qa_pair, config):
    """
    Tokenize a Q&A pair using Qwen2.5 chat template.

    Args:
        tokenizer: Hugging Face tokenizer
        qa_pair (dict): Q&A pair to tokenize
        config (dict): Tokenization configuration

    Returns:
        dict: Tokenized Q&A pair
    """
    tokenization_config = config.get("tokenization", {})
    max_length = tokenization_config.get("max_length", 512)
    truncation = tokenization_config.get("truncation", True)
    padding = tokenization_config.get("padding", "max_length")

    question = qa_pair.get("question", "")
    answer = qa_pair.get("answer", "")

    # Format as Qwen2.5 chat template
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize
    tokenized = tokenizer(
        formatted_text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors="pt",
    )

    # Convert tensors to lists for JSON serialization
    tokenized_data = {
        "input_ids": tokenized["input_ids"].tolist()[0],
        "attention_mask": tokenized["attention_mask"].tolist()[0],
    }

    # Add original text for reference
    result = {
        "question": question,
        "answer": answer,
        "tokenized": tokenized_data,
        "formatted_text": formatted_text,
    }

    # Preserve metadata if present
    for key in qa_pair:
        if key not in ["question", "answer"]:
            result[key] = qa_pair[key]

    return result


def tokenize_dataset(
    input_path, output_path, config_path, model_name="Qwen/Qwen2.5-7B-Instruct"
):
    """
    Tokenize a dataset of Q&A pairs using Qwen2.5 tokenizer.

    Args:
        input_path (str): Path to input JSON file
        output_path (str): Path to output JSON file
        config_path (str): Path to configuration file
        model_name (str): Name or path of the model for tokenizer
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info(f"Tokenizing dataset from {input_path}")

    # Load the dataset
    with open(input_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)

    # Tokenize each Q&A pair
    tokenized_pairs = []
    for pair in tqdm(qa_pairs, desc="Tokenizing"):
        tokenized_pair = tokenize_qa_pair(tokenizer, pair, config)
        tokenized_pairs.append(tokenized_pair)

    logger.info(f"Tokenized {len(tokenized_pairs)} Q&A pairs")

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the tokenized dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenized_pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved tokenized dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize Q&A dataset")
    parser.add_argument("input", type=str, help="Path to input JSON file")
    parser.add_argument("output", type=str, help="Path to output JSON file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data/data_processing.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path for tokenizer",
    )

    args = parser.parse_args()

    tokenize_dataset(args.input, args.output, args.config, args.model)


if __name__ == "__main__":
    main()
