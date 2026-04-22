"""
Main orchestration script: runs all benchmarks for a given model+compression combo.

Usage:
    python evaluation/run_all.py \\
        --model Qwen/Qwen3-8B \\
        --compression gptq \\
        --quantized-model-path ./quantized/qwen3-8b-gptq \\
        --benchmarks gsm8k boolq msmarco \\
        --max-samples 200 \\
        --output-dir results/qwen3-8b-gptq
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_SCRIPTS = {
    # English benchmarks
    "gsm8k":   "benchmarks/gsm8k_eval.py",
    "msmarco": "benchmarks/msmarco_eval.py",
    "boolq":   "benchmarks/boolq_eval.py",
    # Multilingual benchmarks
    "mgsm_bn":             "benchmarks/mgsm_eval.py",
    "indicqa_or":          "benchmarks/indicqa_eval.py",
    "indicqa_hi":          "benchmarks/indicqa_eval.py",
    "indic_sentiment_hi":  "benchmarks/indic_sentiment_eval.py",
    "indic_sentiment_or":  "benchmarks/indic_sentiment_eval.py",
}

# Extra CLI flags injected per benchmark key
BENCHMARK_EXTRA_ARGS = {
    "mgsm_bn":             ["--language", "bn"],
    "indicqa_or":          ["--language", "or"],
    "indicqa_hi":          ["--language", "hi"],
    "indic_sentiment_hi":  ["--language", "hi"],
    "indic_sentiment_or":  ["--language", "or"],
}


def run_benchmark(benchmark: str, model_path: str, compression: str,
                  output_path: str, max_samples: int, config_dir: str):
    script = BENCHMARK_SCRIPTS[benchmark]
    cmd = [
        sys.executable, script,
        "--model", model_path,
        "--compression", compression,
        "--output", output_path,
        "--config-dir", config_dir,
    ]
    if max_samples:
        cmd += ["--max-samples", str(max_samples)]
    # Inject language flags for multilingual benchmarks
    cmd += BENCHMARK_EXTRA_ARGS.get(benchmark, [])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"Benchmark '{benchmark}' failed with exit code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks for a model+compression")
    parser.add_argument("--model-id", required=True, help="Original HF model ID (for labelling)")
    parser.add_argument("--model-path", required=True,
                        help="Path to (optionally quantized) model to evaluate")
    parser.add_argument(
        "--compression",
        required=True,
        choices=["baseline", "gptq", "awq", "kv_compress", "fp8", "int4_bnb", "int4_bnb_kv"],
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["gsm8k", "boolq", "msmarco"],
        choices=list(BENCHMARK_SCRIPTS.keys()) + ["all", "multilingual"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    # Expand shorthand benchmark groups
    ENGLISH_BENCHMARKS = ["gsm8k", "boolq", "msmarco"]
    MULTILINGUAL_BENCHMARKS = [
        "mgsm_bn", "indicqa_or", "indicqa_hi",
        "indic_sentiment_hi", "indic_sentiment_or",
    ]
    expanded = []
    for b in args.benchmarks:
        if b == "all":
            expanded = ENGLISH_BENCHMARKS + MULTILINGUAL_BENCHMARKS
            break
        elif b == "multilingual":
            expanded += MULTILINGUAL_BENCHMARKS
        else:
            expanded.append(b)
    args.benchmarks = expanded

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for bench in args.benchmarks:
        output_file = output_dir / f"{bench}_results.json"
        rc = run_benchmark(
            bench, args.model_path, args.compression,
            str(output_file), args.max_samples, args.config_dir
        )
        if output_file.exists():
            with open(output_file) as f:
                all_results[bench] = json.load(f)
        else:
            all_results[bench] = {"error": f"exit code {rc}"}

    # Write combined summary
    summary_path = output_dir / "summary.json"
    summary = {
        "model_id": args.model_id,
        "model_path": args.model_path,
        "compression": args.compression,
        "results": all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    # Print quick table
    print("\n=== Results Summary ===")
    print(f"Model : {args.model_id}")
    print(f"Method: {args.compression}")
    print(f"{'Benchmark':<12} {'Key Metric':<12} {'Latency(s)':<12} {'Tok/s':<10}")
    print("-" * 50)
    for bench, res in all_results.items():
        if "error" in res:
            print(f"{bench:<12} ERROR")
            continue
        metric = res.get("accuracy") or res.get("rouge_l") or "N/A"
        lat = res.get("avg_latency_sec", 0)
        tps = res.get("tokens_per_sec", 0)
        print(f"{bench:<12} {float(metric):<12.4f} {lat:<12.3f} {tps:<10.1f}")


if __name__ == "__main__":
    main()
