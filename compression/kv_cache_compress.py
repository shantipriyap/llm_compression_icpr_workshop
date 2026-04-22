"""
TurboQuant-style KV-cache compression wrapper.
Wraps a loaded HuggingFace model with per-layer KV-cache quantization.

Owner: Debasish
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class KVCacheCompressor:
    """
    Applies per-head asymmetric 4-bit KV-cache quantization (TurboQuant-style).
    The model weights are kept in bf16/fp16; only the KV cache tensors are
    quantized on-the-fly during generation.
    """

    def __init__(self, model, config: dict):
        self.model = model
        self.kv_bits = config.get("kv_bits", 4)
        self.group_size = config.get("kv_group_size", 64)
        self.residual_length = config.get("residual_length", 128)
        self._install_hooks()

    # ------------------------------------------------------------------ #
    # Quantization helpers                                                  #
    # ------------------------------------------------------------------ #

    def _quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Asymmetric per-group min-max quantization."""
        shape = x.shape
        # Reshape to groups
        x_flat = x.reshape(-1, self.group_size)
        x_min = x_flat.min(dim=-1, keepdim=True).values
        x_max = x_flat.max(dim=-1, keepdim=True).values
        scale = (x_max - x_min) / (2 ** self.kv_bits - 1)
        scale = scale.clamp(min=1e-8)
        x_q = ((x_flat - x_min) / scale).round().clamp(0, 2**self.kv_bits - 1)
        x_dq = x_q * scale + x_min
        return x_dq.reshape(shape)

    def _install_hooks(self):
        """
        Register forward hooks on all attention modules to quantize
        KV vectors after they are computed but before they are cached.
        """
        self._hooks = []
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if "Attention" in cls_name:
                hook = module.register_forward_hook(self._attention_hook)
                self._hooks.append(hook)
        logger.info(f"Installed KV-cache compression hooks on {len(self._hooks)} attention layers.")

    def _attention_hook(self, module, inputs, outputs):
        """
        Post-forward hook: quantize the key/value portions of the output.
        Handles HuggingFace models that return (attn_output, attn_weights, past_kv).
        """
        if not isinstance(outputs, tuple) or len(outputs) < 3:
            return outputs

        attn_output, attn_weights, past_kv = outputs[0], outputs[1], outputs[2]

        if past_kv is not None:
            key_cache, value_cache = past_kv
            # Only quantize tokens beyond the residual window
            if key_cache.shape[2] > self.residual_length:
                key_cache = self._quantize_tensor(key_cache)
                value_cache = self._quantize_tensor(value_cache)
            past_kv = (key_cache, value_cache)

        return (attn_output, attn_weights, past_kv) + outputs[3:]

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def load_model_with_kv_compression(model_id: str, config: dict, device_map: str = "auto"):
    """Load a model and wrap it with KV-cache compression."""
    kv_cfg = config["kv_cache_compress"]

    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()

    compressor = KVCacheCompressor(model, kv_cfg)
    logger.info(f"KV-cache compression enabled (bits={kv_cfg['kv_bits']}, group={kv_cfg['kv_group_size']})")

    return model, tokenizer, compressor


def main():
    parser = argparse.ArgumentParser(description="TurboQuant-style KV-cache compression demo")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", default="Tell me about Odia language.", help="Test prompt")
    parser.add_argument(
        "--config",
        default="config/compression.yaml",
        help="Path to compression config YAML",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        compression_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer, compressor = load_model_with_kv_compression(
        args.model, compression_cfg
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    compressor.remove_hooks()


if __name__ == "__main__":
    main()
