"""
IndicSentiment evaluation script (Hindi and Odia).

Uses ai4bharat/IndicSentiment — binary sentiment classification.
Metric: accuracy (Positive / Negative).
Tests whether compression degrades Indic language sentiment understanding.
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

SENTIMENT_PROMPT = """\
Classify the sentiment of the following text as "Positive" or "Negative". \
Reply with only one word.

Text: {text}
Sentiment:"""

# Map dataset integer labels → canonical strings
LABEL_MAP = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}


def _parse_sentiment(text: str) -> str:
    text = text.strip().lower()
    if text.startswith("pos"):
        return "positive"
    if text.startswith("neg"):
        return "negative"
    return "unknown"


def evaluate_indic_sentiment(
    model, tokenizer, language: str = "hi", max_samples: int = None
) -> dict:
    # IndicSentiment subset names use "translation-<lang>" format
    subset = f"translation-{language}"
    logger.info(f"Loading IndicSentiment dataset (subset={subset}) ...")
    try:
        dataset = load_dataset(
            "ai4bharat/IndicSentiment", subset, split="test"
        )
    except Exception:
        # Fallback: try without subset name (some versions differ)
        logger.warning(f"Subset '{subset}' not found, trying direct load...")
        dataset = load_dataset(
            "ai4bharat/IndicSentiment", split="test"
        )

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    total_latency = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc=f"IndicSentiment-{language.upper()}"):
        # Handle varying field names across dataset versions
        text = (
            example.get("sentence")
            or example.get("text")
            or example.get("TEXT")
            or ""
        )
        raw_label = example.get("LABEL") or example.get("label") or example.get("labels")
        gold = LABEL_MAP.get(raw_label, str(raw_label).lower())

        prompt = SENTIMENT_PROMPT.format(text=text[:800])
        prediction, latency, n_tokens = generate_answer(
            model, tokenizer, prompt, max_new_tokens=8
        )

        pred = _parse_sentiment(prediction)
        if pred == gold:
            correct += 1

        total += 1
        total_latency += latency
        total_tokens += n_tokens

    results = {
        "benchmark": f"indic_sentiment_{language}",
        "language": language,
        "num_samples": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "avg_latency_sec": total_latency / total if total > 0 else 0.0,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    logger.info(
        f"IndicSentiment-{language.upper()} accuracy: {results['accuracy']:.4f} "
        f"({correct}/{total})"
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="IndicSentiment evaluation (Hindi / Odia)"
    )
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument(
        "--compression",
        default="baseline",
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8"],
    )
    parser.add_argument(
        "--language", default="hi",
        help="Language code: 'hi' (Hindi) or 'or' (Odia). Default: hi"
    )
    parser.add_argument("--output", default="results/indic_sentiment_hi_results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/compression.yaml") as f:
        comp_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer = load_model(args.model, args.compression, comp_cfg)
    results = evaluate_indic_sentiment(
        model, tokenizer, language=args.language, max_samples=args.max_samples
    )

    results["model"] = args.model
    results["compression"] = args.compression

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
