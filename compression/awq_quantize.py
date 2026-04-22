"""
AWQ 4-bit quantization script.
Quantizes a model using AutoAWQ and saves it locally.

Supports two calibration modes:
  --calib english   : standard English-only calibration (C4-style, default)
  --calib multilingual : 50/50 English + Indic calibration corpus
                         (Hindi, Odia, Bengali, Tamil, Telugu, Kannada,
                          Malayalam, Marathi, Gujarati) to protect
                         Indic-script weight patterns from quantisation.

Owner: Subrat, Anurag
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List

import yaml
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Calibration corpora ────────────────────────────────────────────────────────

_ENGLISH_CALIB = [
    "Democracy is a system of government by the whole population.",
    "Machine learning is a subset of artificial intelligence.",
    "The water cycle describes how water evaporates from surfaces.",
    "Large language models are trained on massive text corpora.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Neural networks are inspired by the structure of the human brain.",
    "The speed of light in vacuum is approximately 299,792 kilometres per second.",
    "Quantum computing leverages superposition and entanglement for computation.",
    "Climate change refers to long-term shifts in global temperatures.",
    "The human genome contains approximately 3 billion base pairs.",
    "Reinforcement learning trains agents through reward and penalty signals.",
    "Transformer architectures use self-attention for sequence modelling.",
    "Gradient descent minimises the loss function iteratively.",
    "The French Revolution began in 1789 and reshaped modern history.",
    "Protein folding determines the three-dimensional structure of proteins.",
    "Open-source software is distributed with freely available source code.",
]

# Same semantic content in Indic languages — protects Indic-specific weight patterns
_INDIC_CALIB = [
    # Hindi (Devanagari)
    "\u0932\u094b\u0915\u0924\u0902\u0924\u094d\u0930 \u092a\u0942\u0930\u0940 \u091c\u0928\u0938\u0902\u0916\u094d\u092f\u093e \u0926\u094d\u0935\u093e\u0930\u093e \u0938\u0930\u0915\u093e\u0930 \u0915\u0940 \u090f\u0915 \u092a\u094d\u0930\u0923\u093e\u0932\u0940 \u0939\u0948\u0964",
    "\u092e\u0936\u0940\u0928 \u0932\u0930\u094d\u0928\u093f\u0902\u0917 \u0915\u0943\u0924\u094d\u0930\u093f\u092e \u092c\u0941\u0926\u094d\u0927\u093f\u092e\u0924\u094d\u0924\u093e \u0915\u093e \u090f\u0915 \u0909\u092a\u0938\u092e\u0941\u091a\u094d\u091a\u092f \u0939\u0948\u0964",
    "\u092a\u094d\u0930\u0915\u093e\u0936 \u0938\u0902\u0936\u094d\u0932\u0947\u0937\u0923 \u0938\u0942\u0930\u094d\u092f \u0915\u0947 \u092a\u094d\u0930\u0915\u093e\u0936 \u0915\u094b \u0930\u093e\u0938\u093e\u092f\u0928\u093f\u0915 \u090a\u0930\u094d\u091c\u093e \u092e\u0947\u0902 \u092c\u0926\u0932\u0924\u093e \u0939\u0948\u0964",
    "\u0928\u094d\u092f\u0942\u0930\u0932 \u0928\u0947\u091f\u0935\u0930\u094d\u0915 \u092e\u093e\u0928\u0935 \u092e\u0938\u094d\u0924\u093f\u0937\u094d\u0915 \u0915\u0940 \u0938\u0902\u0930\u091a\u0928\u093e \u0938\u0947 \u092a\u094d\u0930\u0947\u0930\u093f\u0924 \u0939\u0948\u0902\u0964",
    # Odia
    "\u0b17\u0b23\u0b24\u0b28\u0b4d\u0b24\u0b4d\u0b30 \u0b39\u0b47\u0b09\u0b1b\u0b3f \u0b38\u0b2e\u0b17\u0b4d\u0b30 \u0b1c\u0b28\u0b17\u0b23 \u0b26\u0b4d\u0b71\u0b3e\u0b30\u0b3e \u0b38\u0b30\u0b15\u0b3e\u0b30 \u0b2c\u0b4d\u0b5f\u0b2c\u0b38\u0b4d\u0b25\u0b3e\u0964",
    "\u0b2e\u0b47\u0b38\u0b3f\u0b28 \u0b32\u0b30\u0b4d\u0b28\u0b3f\u0b02 \u0b39\u0b47\u0b09\u0b1b\u0b3f \u0b15\u0b43\u0b24\u0b4d\u0b30\u0b3f\u0b2e \u0b2c\u0b41\u0b26\u0b4d\u0b27\u0b3f\u0b2e\u0b24\u0b4d\u0b24\u0b3e\u0b30 \u0b0f\u0b15 \u0b09\u0b2a-\u0b17\u0b23\u0964",
    # Bengali
    "\u0997\u09a3\u09a4\u09a8\u09cd\u09a4\u09cd\u09b0 \u09b9\u09b2 \u09b8\u09ae\u0997\u09cd\u09b0 \u099c\u09a8\u0997\u09a3\u09c7\u09b0 \u09a6\u09cd\u09ac\u09be\u09b0\u09be \u09b8\u09b0\u0995\u09be\u09b0 \u09ac\u09cd\u09af\u09ac\u09b8\u09cd\u09a5\u09be\u0964",
    "\u09ae\u09c7\u09b6\u09bf\u09a8 \u09b2\u09be\u09b0\u09cd\u09a8\u09bf\u0982 \u0995\u09c3\u09a4\u09cd\u09b0\u09bf\u09ae \u09ac\u09c1\u09a6\u09cd\u09a7\u09bf\u09ae\u09a4\u09cd\u09a4\u09be\u09b0 \u098f\u0995\u099f\u09bf \u0989\u09aa\u09b8\u09c7\u099f\u0964",
    # Tamil
    "\u0b9c\u0ba9\u0ba8\u0bbe\u0baf\u0b95\u0bae\u0bcd \u0b8e\u0ba9\u0bcd\u0baa\u0ba4\u0bc1 \u0bae\u0b95\u0bcd\u0b95\u0bb3\u0bcd \u0b85\u0ba9\u0bc8\u0bb5\u0bb0\u0bc1\u0bae\u0bcd \u0b86\u0bb3\u0bc1\u0bae\u0bcd \u0b92\u0bb0\u0bc1 \u0b85\u0bb0\u0b9a\u0bc1 \u0bae\u0bc1\u0bb1\u0bc8\u0964",
    "\u0b87\u0baf\u0ba8\u0bcd\u0ba4\u0bbf\u0bb0 \u0b95\u0bb1\u0bcd\u0bb1\u0bb2\u0bcd \u0b9a\u0bc6\u0baf\u0bb1\u0bcd\u0b95\u0bc8 \u0ba8\u0bc1\u0ba3\u0bcd\u0ba3\u0bb1\u0bbf\u0bb5\u0bbf\u0ba9\u0bcd \u0b92\u0bb0\u0bc1 \u0b89\u0b9f\u0bcd\u0baa\u0bbf\u0bb0\u0bbf\u0bb5\u0bc1 \u0b86\u0b95\u0bc1\u0bae\u0bcd\u0964",
    # Telugu
    "\u0c2a\u0c4d\u0c30\u0c1c\u0c3e\u0c38\u0c4d\u0c35\u0c3e\u0c2e\u0c4d\u0c2f\u0c02 \u0c05\u0c28\u0c47\u0c26\u0c3f \u0c2e\u0c4a\u0c24\u0c4d\u0c24\u0c02 \u0c1c\u0c28\u0c3e\u0c2d\u0c3e\u0c1a\u0c47\u0c24 \u0c2a\u0c4d\u0c30\u0c2d\u0c41\u0c24\u0c4d\u0c35 \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c4d\u0c25\u0964",
    # Kannada
    "\u0caa\u0ccd\u0cb0\u0c9c\u0cbe\u0caa\u0ccd\u0cb0\u0cad\u0cc1\u0ca4\u0ccd\u0cb5 \u0c8e\u0c82\u0ca6\u0cb0\u0cc7 \u0c87\u0ca1\u0cc0 \u0c9c\u0ca8\u0cb8\u0c82\u0c96\u0ccd\u0caf\u0cc6\u0caf \u0c86\u0ca1\u0cb3\u0cbf\u0ca4 \u0cb5\u0ccd\u0caf\u0cb5\u0cb8\u0ccd\u0ca5\u0cc6\u0964",
    # Malayalam
    "\u0d1c\u0d28\u0d3e\u0d27\u0d3f\u0d2a\u0d24\u0d4d\u0d2f\u0d02 \u0d0e\u0d28\u0d4d\u0d24\u0d3e\u0d23\u0d4d \u0d2e\u0d41\u0d34\u0d41\u0d35\u0d7b \u0d1c\u0d28\u0d19\u0d4d\u0d19\u0d33\u0d41\u0d1f\u0d46 \u0d2d\u0d30\u0d23 \u0d38\u0d02\u0d35\u0d3f\u0d27\u0d3e\u0d28\u0d2e\u0d3e\u0d23\u0d4d\u0964",
    # Marathi
    "\u0932\u094b\u0915\u0936\u093e\u0939\u0940 \u092e\u094d\u0939\u0923\u091c\u0947 \u0938\u0902\u092a\u0942\u0930\u094d\u0923 \u0932\u094b\u0915\u0938\u0902\u0916\u094d\u092f\u0947\u0926\u094d\u0935\u093e\u0930\u0947 \u0938\u0930\u0915\u093e\u0930 \u091a\u093e\u0932\u0935\u0923\u094d\u092f\u093e\u091a\u0940 \u0935\u094d\u092f\u0935\u0938\u094d\u0925\u093e\u0964",
    # Gujarati
    "\u0aaa\u0abe\u0aa3\u0ac0\u0aa8\u0ac1\u0a82 \u0ab0\u0abe\u0ab8\u0abe\u0aaf\u0aa3\u0abf\u0a95 \u0ab8\u0ac2\u0aa4\u0acd\u0ab0 \u0ab6\u0ac1\u0a82 \u0a9b\u0ac7?",
]


def build_calib_data(mode: str) -> List[str]:
    """Return calibration corpus for the specified mode."""
    if mode == "multilingual":
        # 50 / 50 English + Indic, interleaved so the model sees both
        combined = []
        for i in range(max(len(_ENGLISH_CALIB), len(_INDIC_CALIB))):
            if i < len(_ENGLISH_CALIB):
                combined.append(_ENGLISH_CALIB[i])
            if i < len(_INDIC_CALIB):
                combined.append(_INDIC_CALIB[i])
        logger.info(f"Multilingual calibration corpus: {len(combined)} sentences "
                    f"({len(_ENGLISH_CALIB)} EN + {len(_INDIC_CALIB)} Indic)")
        return combined
    else:
        logger.info(f"English-only calibration corpus: {len(_ENGLISH_CALIB)} sentences")
        return _ENGLISH_CALIB


def quantize_model(model_id: str, output_dir: str, compression_cfg: dict,
                   calib_mode: str = "english"):
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

    calib_data = build_calib_data(calib_mode)

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    logger.info(f"Loading model for AWQ quantization: {model_id}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        safetensors=True,
    )

    logger.info(f"Starting AWQ quantization (calib_mode={calib_mode}) ...")
    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    elapsed = time.time() - t0
    logger.info(f"AWQ quantization complete in {elapsed:.1f}s")

    output_path_labeled = Path(str(output_path) + f"_{calib_mode}_calib")
    output_path_labeled.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving AWQ quantized model to {output_path_labeled}")
    model.save_quantized(str(output_path_labeled))
    tokenizer.save_pretrained(str(output_path_labeled))

    return str(output_path_labeled)


def main():
    parser = argparse.ArgumentParser(description="AWQ 4-bit quantization")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", required=True, help="Directory to save quantized model")
    parser.add_argument(
        "--config",
        default="config/compression.yaml",
        help="Path to compression config YAML",
    )
    parser.add_argument(
        "--calib",
        choices=["english", "multilingual"],
        default="english",
        help="Calibration corpus: 'english' (C4-style, default) or "
             "'multilingual' (50%% EN + 50%% Indic, protects Indic weight patterns)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        compression_cfg = yaml.safe_load(f)["compression"]

    saved_path = quantize_model(args.model, args.output_dir, compression_cfg,
                                calib_mode=args.calib)
    logger.info(f"AWQ quantized model saved to: {saved_path}")


if __name__ == "__main__":
    main()
