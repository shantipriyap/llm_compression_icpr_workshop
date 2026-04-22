"""
MGSM (Multilingual Grade School Math) evaluation script.

Uses the juletxara/mgsm dataset (Hindi subset by default).
Measures exact-match accuracy on arithmetic word problems in Hindi.
Tests whether LLM compression degrades multilingual math reasoning.
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
from benchmarks.utils import load_model, generate_answer, get_gpu_memory_mb, extract_number

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MGSM_PROMPT = """\
Solve the following math problem step by step. At the end, write "The answer is: <number>".

Problem: {question}
Solution:"""


def evaluate_mgsm(model, tokenizer, language: str = "hi", max_samples: int = None) -> dict:
    logger.info(f"Loading MGSM dataset (language={language}) ...")
    dataset = load_dataset("juletxara/mgsm", language, split="test")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    total_latency = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc=f"MGSM-{language.upper()}"):
        question = example["question"]

        # answer_number field holds the numeric gold answer
        gold_number = None
        if "answer_number" in example and example["answer_number"] is not None:
            try:
                gold_number = float(example["answer_number"])
            except (ValueError, TypeError):
                gold_number = extract_number(str(example["answer_number"]))
        elif "answer" in example:
            gold_number = extract_number(str(example["answer"]))

        prompt = MGSM_PROMPT.format(question=question)
        prediction, latency, n_tokens = generate_answer(
            model, tokenizer, prompt, max_new_tokens=300
        )

        pred_number = extract_number(prediction)
        if pred_number is not None and gold_number is not None:
            if abs(pred_number - gold_number) < 1e-3:
                correct += 1

        total += 1
        total_latency += latency
        total_tokens += n_tokens

    results = {
        "benchmark": f"mgsm_{language}",
        "language": language,
        "num_samples": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "avg_latency_sec": total_latency / total if total > 0 else 0.0,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    logger.info(f"MGSM-{language.upper()} accuracy: {results['accuracy']:.4f} ({correct}/{total})")
    return results


def main():
    parser = argparse.ArgumentParser(description="MGSM multilingual math evaluation")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument(
        "--compression",
        default="baseline",
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8"],
    )
    parser.add_argument("--language", default="hi", help="Language code (default: hi)")
    parser.add_argument("--output", default="results/mgsm_hi_results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/compression.yaml") as f:
        comp_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer = load_model(args.model, args.compression, comp_cfg)
    results = evaluate_mgsm(model, tokenizer, language=args.language, max_samples=args.max_samples)

    results["model"] = args.model
    results["compression"] = args.compression

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
