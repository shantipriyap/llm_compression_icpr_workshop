"""
IndicQA evaluation script (Odia and Hindi).

Uses ai4bharat/IndicQA — SQuAD-style extractive QA for Indic languages.
Metrics: Exact Match (EM) and token-overlap F1.
Tests whether compression degrades low-resource language comprehension.
"""

import argparse
import json
import logging
import re
import string
import sys
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.utils import load_model, generate_answer, get_gpu_memory_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INDICQA_PROMPT = """\
Read the passage and answer the question using only words from the passage. \
Give a short, direct answer.

Passage: {context}

Question: {question}
Answer:"""


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize(prediction) == _normalize(ground_truth)


def _best_scores(prediction: str, gold_answers: list) -> tuple:
    """Return best EM and F1 across all gold answers."""
    if not gold_answers:
        return 0.0, 0.0
    em = max(int(_exact_match(prediction, a)) for a in gold_answers)
    f1 = max(_token_f1(prediction, a) for a in gold_answers)
    return float(em), f1


def evaluate_indicqa(model, tokenizer, language: str = "or", max_samples: int = None) -> dict:
    logger.info(f"Loading IndicQA dataset (language={language}) ...")
    dataset = load_dataset(
        "ai4bharat/IndicQA", language, split="test", trust_remote_code=True
    )

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_em = 0.0
    total_f1 = 0.0
    total = 0
    total_latency = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc=f"IndicQA-{language.upper()}"):
        context = (example.get("context") or example.get("passage") or "")[:1500]
        question = example.get("question", "")

        # answers field is SQuAD-style: {"text": [...], "answer_start": [...]}
        answers_field = example.get("answers", {})
        if isinstance(answers_field, dict):
            gold_answers = answers_field.get("text", [])
        elif isinstance(answers_field, list):
            gold_answers = answers_field
        else:
            gold_answers = []

        # Skip unanswerable examples
        if not gold_answers or (len(gold_answers) == 1 and gold_answers[0] == ""):
            continue

        prompt = INDICQA_PROMPT.format(context=context, question=question)
        prediction, latency, n_tokens = generate_answer(
            model, tokenizer, prompt, max_new_tokens=64
        )

        # Use only the first line of the prediction as the answer
        prediction = prediction.strip().split("\n")[0].strip()

        em, f1 = _best_scores(prediction, gold_answers)
        total_em += em
        total_f1 += f1
        total += 1
        total_latency += latency
        total_tokens += n_tokens

    results = {
        "benchmark": f"indicqa_{language}",
        "language": language,
        "num_samples": total,
        "exact_match": total_em / total if total > 0 else 0.0,
        "f1": total_f1 / total if total > 0 else 0.0,
        "avg_latency_sec": total_latency / total if total > 0 else 0.0,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    logger.info(
        f"IndicQA-{language.upper()} EM: {results['exact_match']:.4f}  F1: {results['f1']:.4f}  "
        f"({total} samples)"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="IndicQA evaluation (Odia / Hindi)")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument(
        "--compression",
        default="baseline",
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8"],
    )
    parser.add_argument(
        "--language", default="or",
        help="Language code: 'or' (Odia) or 'hi' (Hindi). Default: or"
    )
    parser.add_argument("--output", default="results/indicqa_or_results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/compression.yaml") as f:
        comp_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer = load_model(args.model, args.compression, comp_cfg)
    results = evaluate_indicqa(
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
