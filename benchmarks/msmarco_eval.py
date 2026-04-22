"""
MS MARCO passage-QA evaluation script.

Measures ROUGE-L and answer F1 on the v2.1 validation set.
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
import evaluate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.utils import load_model, generate_answer, get_gpu_memory_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MSMARCO_PROMPT = """\
Use the following passages to answer the question. Be concise.

Passages:
{passages}

Question: {question}
Answer:"""


def _format_passages(passages: dict) -> str:
    texts = passages.get("passage_text", [])
    return "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts[:3]))  # top-3 passages


def evaluate_msmarco(model, tokenizer, cfg: dict, max_samples: int = None) -> dict:
    rouge = evaluate.load("rouge")

    logger.info("Loading MS MARCO dataset ...")
    dataset = load_dataset(cfg["hf_dataset"], cfg["hf_subset"], split=cfg["split"])

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    predictions = []
    references = []
    total_latency = 0.0
    total_tokens = 0

    for example in tqdm(dataset, desc="MS MARCO"):
        question = example["query"]
        passages = example["passages"]
        answers = example.get("answers", [])
        if not answers:
            continue

        prompt = MSMARCO_PROMPT.format(
            passages=_format_passages(passages), question=question
        )
        prediction, latency, n_tokens = generate_answer(
            model, tokenizer, prompt, max_new_tokens=128
        )
        predictions.append(prediction)
        references.append(answers[0])
        total_latency += latency
        total_tokens += n_tokens

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    n = len(predictions)
    results = {
        "benchmark": "msmarco",
        "num_samples": n,
        "rouge_l": rouge_scores["rougeL"],
        "rouge_1": rouge_scores["rouge1"],
        "avg_latency_sec": total_latency / n if n > 0 else 0.0,
        "tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0.0,
        "gpu_memory_mb": get_gpu_memory_mb(),
    }
    logger.info(f"MS MARCO ROUGE-L: {results['rouge_l']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="MS MARCO evaluation")
    parser.add_argument("--model", required=True, help="Model ID or path")
    parser.add_argument(
        "--compression",
        default="baseline",
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8"],
    )
    parser.add_argument("--output", default="results/msmarco_results.json")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/benchmarks.yaml") as f:
        bench_cfg = yaml.safe_load(f)["benchmarks"]["msmarco"]
    with open(f"{args.config_dir}/compression.yaml") as f:
        comp_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer = load_model(args.model, args.compression, comp_cfg)
    results = evaluate_msmarco(model, tokenizer, bench_cfg, max_samples=args.max_samples)

    results["model"] = args.model
    results["compression"] = args.compression

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
