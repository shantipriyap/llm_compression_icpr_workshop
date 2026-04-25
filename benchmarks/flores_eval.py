"""
FLORES-200 chrF++ evaluation script for Indic languages.

Uses the facebook/flores dataset (devtest split).
Measures translation quality (chrF++) from English to a target Indic language
by prompting the model with a zero-shot translation instruction.
Tests whether LLM compression degrades cross-lingual generation quality.

Supported language codes (FLORES-200 BCP-47 format):
  hin_Deva (Hindi), ory_Orya (Odia), ben_Beng (Bengali),
  tam_Taml (Tamil), tel_Telu (Telugu), guj_Gujr (Gujarati),
  mar_Deva (Marathi), mal_Mlym (Malayalam), kan_Knda (Kannada)

Usage:
  python benchmarks/flores_eval.py \\
      --model Qwen/Qwen3-8B \\
      --compression baseline \\
      --language hin_Deva \\
      --output results/flores_hi_results.json \\
      --max-samples 200
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

# Map FLORES-200 lang code → human-readable name for prompts
LANG_NAMES = {
    "hin_Deva": "Hindi",
    "ory_Orya": "Odia",
    "ben_Beng": "Bengali",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "guj_Gujr": "Gujarati",
    "mar_Deva": "Marathi",
    "mal_Mlym": "Malayalam",
    "kan_Knda": "Kannada",
}

FLORES_PROMPT = """\
Translate the following English sentence into {target_lang}. \
Output only the translation, with no explanation or extra words.

English: {source}
{target_lang}:"""


def _compute_chrf(hypothesis: str, reference: str, beta: float = 2.0) -> float:
    """Compute chrF++ (character n-gram F-score, beta=2 weights recall)."""
    def _char_ngrams(text: str, n: int) -> dict:
        ngrams: dict = {}
        for i in range(len(text) - n + 1):
            ng = text[i:i + n]
            ngrams[ng] = ngrams.get(ng, 0) + 1
        return ngrams

    def _precision_recall(hyp_ng: dict, ref_ng: dict) -> tuple[float, float]:
        matched = sum(min(hyp_ng.get(k, 0), ref_ng.get(k, 0)) for k in hyp_ng)
        prec = matched / max(sum(hyp_ng.values()), 1)
        rec = matched / max(sum(ref_ng.values()), 1)
        return prec, rec

    hyp = hypothesis.strip()
    ref = reference.strip()
    if not hyp or not ref:
        return 0.0

    total_p, total_r, count = 0.0, 0.0, 0
    for n in range(1, 7):  # char 1-6-grams
        h_ng = _char_ngrams(hyp, n)
        r_ng = _char_ngrams(ref, n)
        if h_ng or r_ng:
            p, r = _precision_recall(h_ng, r_ng)
            total_p += p
            total_r += r
            count += 1

    if count == 0:
        return 0.0
    avg_p = total_p / count
    avg_r = total_r / count
    if avg_p + avg_r == 0.0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)


def evaluate_flores(model, tokenizer, language: str, max_samples: int = None) -> dict:
    lang_name = LANG_NAMES.get(language, language)
    logger.info(f"Loading FLORES-200 devtest split (eng_Latn → {language}) ...")

    try:
        # facebook/flores uses the target language as config name
        dataset = load_dataset("facebook/flores", language, split="devtest", trust_remote_code=False)
        eng_dataset = load_dataset("facebook/flores", "eng_Latn", split="devtest", trust_remote_code=False)
    except Exception as e:
        logger.error(f"Failed to load FLORES-200 for language '{language}': {e}")
        return {"error": str(e), "language": language}

    if max_samples:
        n = min(max_samples, len(dataset))
        dataset = dataset.select(range(n))
        eng_dataset = eng_dataset.select(range(n))

    chrf_scores = []
    total_latency = 0.0
    total_tokens = 0

    for src_ex, tgt_ex in tqdm(
        zip(eng_dataset, dataset), total=len(dataset), desc=f"FLORES-{language}"
    ):
        source_sentence = src_ex["sentence"]
        reference = tgt_ex["sentence"]

        prompt = FLORES_PROMPT.format(
            target_lang=lang_name,
            source=source_sentence,
        )

        hypothesis, latency, n_tokens = generate_answer(
            model, tokenizer, prompt, max_new_tokens=128, temperature=0.0
        )
        hypothesis = hypothesis.strip().split("\n")[0].strip()

        score = _compute_chrf(hypothesis, reference)
        chrf_scores.append(score)
        total_latency += latency
        total_tokens += n_tokens

    n = len(chrf_scores)
    avg_chrf = sum(chrf_scores) / n if n > 0 else 0.0
    avg_latency = total_latency / n if n > 0 else 0.0
    tps = total_tokens / total_latency if total_latency > 0 else 0.0

    return {
        "language": language,
        "language_name": lang_name,
        "num_samples": n,
        "chrf_plus_plus": round(avg_chrf * 100, 2),  # percent, 0–100
        "latency_per_sample_s": round(avg_latency, 3),
        "tokens_per_sec": round(tps, 1),
        "gpu_memory_mb": get_gpu_memory_mb(),
    }


def main():
    parser = argparse.ArgumentParser(description="FLORES-200 chrF++ evaluation")
    parser.add_argument("--model", required=True, help="HF model ID or path")
    parser.add_argument(
        "--compression",
        default="baseline",
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8", "int4_bnb", "int4_bnb_kv"],
    )
    parser.add_argument(
        "--language", default="hin_Deva",
        help="FLORES-200 language code, e.g. hin_Deva, ory_Orya, ben_Beng",
    )
    parser.add_argument("--output", default="results/flores_results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    with open(f"{args.config_dir}/compression.yaml") as f:
        comp_cfg = yaml.safe_load(f)["compression"]

    model, tokenizer = load_model(args.model, args.compression, comp_cfg)
    results = evaluate_flores(model, tokenizer, args.language, args.max_samples)

    results["model"] = args.model
    results["compression"] = args.compression

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results written to {args.output}")

    lang = results.get("language_name", args.language)
    chrf = results.get("chrf_plus_plus", "N/A")
    tps = results.get("tokens_per_sec", "N/A")
    print(f"\nFLORES-200 | {lang} | chrF++: {chrf:.2f} | {tps:.1f} tok/s")


if __name__ == "__main__":
    main()
