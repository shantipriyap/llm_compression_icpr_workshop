"""
Shared utilities for loading models, generating text, and measuring metrics.
"""

import os
import re
import time
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Compatibility shim: LossKwargs was renamed in transformers 5.x       #
# ------------------------------------------------------------------ #
try:
    from transformers.utils import LossKwargs  # noqa: F401
except ImportError:
    from dataclasses import dataclass
    import transformers.utils as _tu

    @dataclass
    class LossKwargs:
        pass

    _tu.LossKwargs = LossKwargs


# ------------------------------------------------------------------ #
# Model loading                                                        #
# ------------------------------------------------------------------ #

def load_model(model_id_or_path: str, compression_method: str, compression_cfg: dict):
    """
    Load a model with the specified compression method.

    Args:
        model_id_or_path: HuggingFace model ID or local checkpoint path.
        compression_method: One of 'baseline', 'gptq', 'awq', 'kv_compress', 'fp8'.
        compression_cfg: Compression config dict from compression.yaml.

    Returns:
        (model, tokenizer) tuple.
    """
    logger.info(f"Loading model '{model_id_or_path}' with method '{compression_method}'")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)

    if compression_method in ("baseline", "fp16", "bf16"):
        dtype = torch.float16 if compression_method == "fp16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

    elif compression_method == "gptq":
        # Use transformers-native GPTQ loading (no auto_gptq dependency)
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            device_map="auto",
        )

    elif compression_method == "awq":
        # Use transformers-native AWQ loading (no autoawq dependency for inference)
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            device_map="auto",
        )

    elif compression_method == "kv_compress":
        from compression.kv_cache_compress import KVCacheCompressor
        cfg = compression_cfg["kv_cache_compress"]
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        KVCacheCompressor(model, cfg)

    else:
        raise ValueError(f"Unknown compression method: {compression_method}")

    model.eval()
    return model, tokenizer


# ------------------------------------------------------------------ #
# Generation                                                           #
# ------------------------------------------------------------------ #

def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float, int]:
    """
    Generate a response and return (answer_text, latency_sec, tokens_generated).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)

        output_ids = model.generate(**gen_kwargs)

    latency = time.time() - t0
    tokens_generated = output_ids.shape[1] - input_len
    answer = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return answer.strip(), latency, tokens_generated


# ------------------------------------------------------------------ #
# Memory utilities                                                     #
# ------------------------------------------------------------------ #

def get_gpu_memory_mb() -> float:
    """Return current GPU memory usage in MB (first device)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1024**2
    return 0.0


# ------------------------------------------------------------------ #
# Number extraction                                                    #
# ------------------------------------------------------------------ #

def extract_number(text: str) -> Optional[float]:
    """Extract the last number from a text string (used for GSM8K)."""
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None
