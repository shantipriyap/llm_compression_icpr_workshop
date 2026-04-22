"""
GPTQ 4-bit quantization script.
Quantizes a model using AutoGPTQ and saves it locally.

Owner: Sakshi and Mahi
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import yaml
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_calibration_data(tokenizer, config: dict, n_samples: int = 128, max_length: int = 2048):
    """Load and tokenize calibration data for GPTQ."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:n_samples * 4]

    calibration_data = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        if enc["input_ids"].shape[1] >= 32:  # skip very short sequences
            calibration_data.append(enc["input_ids"])
        if len(calibration_data) >= n_samples:
            break

    logger.info(f"Loaded {len(calibration_data)} calibration samples.")
    return calibration_data


def quantize_model(model_id: str, output_dir: str, compression_cfg: dict):
    """Apply GPTQ 4-bit quantization to a model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = compression_cfg["gptq_4bit"]
    quantize_config = BaseQuantizeConfig(
        bits=cfg["bits"],
        group_size=cfg["group_size"],
        desc_act=cfg["desc_act"],
        damp_percent=cfg["damp_percent"],
    )

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info("Loading calibration data ...")
    calibration_data = load_calibration_data(
        tokenizer, cfg, n_samples=cfg.get("calibration_samples", 128)
    )

    logger.info(f"Loading model for quantization: {model_id}")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config=quantize_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    logger.info("Starting GPTQ quantization ...")
    t0 = time.time()
    model.quantize(calibration_data)
    elapsed = time.time() - t0
    logger.info(f"Quantization complete in {elapsed:.1f}s")

    logger.info(f"Saving quantized model to {output_path}")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="GPTQ 4-bit quantization")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", required=True, help="Directory to save quantized model")
    parser.add_argument(
        "--config",
        default="config/compression.yaml",
        help="Path to compression config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        compression_cfg = yaml.safe_load(f)["compression"]

    saved_path = quantize_model(args.model, args.output_dir, compression_cfg)
    logger.info(f"GPTQ quantized model saved to: {saved_path}")


if __name__ == "__main__":
    main()
