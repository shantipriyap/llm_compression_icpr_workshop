"""
BoolQ (Boolean Question Answering) evaluation script.

Measures Yes/No accuracy on the BoolQ validation set.
Owner: Sakshi and Mahi (GPTQ), Subrat & Anurag (AWQ), Debasish (TurboQuant)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.utils import load_model, generate_answer, get_gpu_memory_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BOOLQ_PROMPT = """\
Read the passage and answer the question with only "Yes" or "No".

Passage: {passage}

Question: {question}
Answer (Yes/No):"""


def _parse_bool(text: str) -> str:
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "unknown"


def evaluate_boolq(model, tokenizer, cfg: dict, max_samples: int = None) -> dict:
    logger.info("Loading BoolQ dataset ...")
    dataset = load_dataset(cfg["hf_dataset"], split=cfg["split"])

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    total_latency = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc="BoolQ"):
        passage = example["passage"]
        question = example["question"]
        gold_label = "yes" if example["answer"] else "no"

        prompt = BOOLQ_PROMPT.format(passage=passage[:1000], question=question)
        prediction, latency, n_tokens = generate_answer(
            model, tokenizer, prompt, max_new_tokens=8
        )

        pred_label = _parse_bool(prediction)
        if pred_label == gold_label:
            correct += 1

        total += 1
        total_latency += latency
        total_tokens += n_tokens

    results = {
        "benchmark": "boolq",
        "num_samples": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "avg_latency_sec": total_latency / total if total > 0 else 0.0,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    logger.info(f"BoolQ accuracy: {results['accuracy']:.4f} ({correct}/{total})")
    return results


def main():
    parser = argparse.ArgumentParser(description="BoolQ evaluation")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument(
        "--compression",
        default="baseline",
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8"],
    )
    parser.add_argument("--output", default="results/boolq_results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/benchmarks.yaml") as f:
        bench_cfg = yaml.safe_load(f)["benchmarks"]["boolq"]
    with open(f"{args.config_dir}/compression.yaml") as f:
        comp_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer = load_model(args.model, args.compression, comp_cfg)
    results = evaluate_boolq(model, tokenizer, bench_cfg, max_samples=args.max_samples)

    results["model"] = args.model
    results["compression"] = args.compression

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
