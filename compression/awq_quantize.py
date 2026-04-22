"""
AWQ 4-bit quantization script.
Quantizes a model using AutoAWQ and saves it locally.

Owner: Subrat, Anurag
"""

import argparse
import logging
import time
from pathlib import Path

import yaml
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def quantize_model(model_id: str, output_dir: str, compression_cfg: dict):
    """Apply AWQ 4-bit quantization to a model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = compression_cfg["awq_4bit"]
    quant_config = {
        "zero_point": cfg["zero_point"],
        "q_group_size": cfg["group_size"],
        "w_bit": cfg["bits"],
        "version": cfg["version"],
    }

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info(f"Loading model for AWQ quantization: {model_id}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        safetensors=True,
    )

    logger.info("Starting AWQ quantization ...")
    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    elapsed = time.time() - t0
    logger.info(f"AWQ quantization complete in {elapsed:.1f}s")

    logger.info(f"Saving AWQ quantized model to {output_path}")
    model.save_quantized(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="AWQ 4-bit quantization")
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
    logger.info(f"AWQ quantized model saved to: {saved_path}")


if __name__ == "__main__":
    main()
