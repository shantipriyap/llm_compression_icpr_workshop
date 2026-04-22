"""
GSM8K (Grade School Math) evaluation script.

Measures exact-match accuracy on arithmetic word problems.
Owner: Sakshi and Mahi (GPTQ), Subrat & Anurag (AWQ), Debasish (TurboQuant)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.utils import load_model, generate_answer, get_gpu_memory_mb, extract_number

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GSM8K_PROMPT = """\
Solve the following math problem step by step. At the end, write "The answer is: <number>".

Problem: {question}
Solution:"""


def evaluate_gsm8k(model, tokenizer, cfg: dict, max_samples: int = None) -> dict:
    logger.info("Loading GSM8K dataset ...")
    dataset = load_dataset(cfg["hf_dataset"], cfg["hf_subset"], split=cfg["split"])

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    total_latency = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc="GSM8K"):
        question = example["question"]
        answer_str = example["answer"]  # e.g. "#### 42"
        gold_match = extract_number(answer_str.split("####")[-1].strip())

        prompt = GSM8K_PROMPT.format(question=question)
        prediction, latency, n_tokens = generate_answer(
            model, tokenizer, prompt, max_new_tokens=300
        )

        pred_number = extract_number(prediction)
        if pred_number is not None and gold_match is not None:
            if abs(pred_number - gold_match) < 1e-3:
                correct += 1

        total += 1
        total_latency += latency
        total_tokens += n_tokens

    results = {
        "benchmark": "gsm8k",
        "num_samples": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "avg_latency_sec": total_latency / total if total > 0 else 0.0,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    logger.info(f"GSM8K accuracy: {results['accuracy']:.4f} ({correct}/{total})")
    return results


def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument(
        "--compression",
        default="baseline",
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8"],
    )
    parser.add_argument("--output", default="results/gsm8k_results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/benchmarks.yaml") as f:
        bench_cfg = yaml.safe_load(f)["benchmarks"]["gsm8k"]
    with open(f"{args.config_dir}/compression.yaml") as f:
        comp_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer = load_model(args.model, args.compression, comp_cfg)
    results = evaluate_gsm8k(model, tokenizer, bench_cfg, max_samples=args.max_samples)

    results["model"] = args.model
    results["compression"] = args.compression

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
