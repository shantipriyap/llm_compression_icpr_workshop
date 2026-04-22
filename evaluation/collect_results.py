"""
Collect and compare results across all model × compression combinations.

Usage:
    python evaluation/collect_results.py --results-dir results/
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_all_summaries(results_dir: Path) -> list[dict]:
    rows = []
    for summary_file in sorted(results_dir.glob("**/summary.json")):
        with open(summary_file) as f:
            summary = json.load(f)

        row = {
            "model": summary.get("model_id", "unknown"),
            "compression": summary.get("compression", "unknown"),
        }
        for bench, res in summary.get("results", {}).items():
            if "accuracy" in res:
                row[f"{bench}_accuracy"] = round(res["accuracy"], 4)
            if "rouge_l" in res:
                row[f"{bench}_rouge_l"] = round(res["rouge_l"], 4)
            if "tokens_per_sec" in res:
                row[f"{bench}_tps"] = round(res["tokens_per_sec"], 1)
            if "gpu_memory_mb" in res:
                row[f"{bench}_gpu_mb"] = round(res["gpu_memory_mb"], 1)
        rows.append(row)
    return rows


def compute_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns showing % degradation vs FP16/BF16 baseline."""
    metric_cols = [c for c in df.columns if c.endswith(("_accuracy", "_rouge_l"))]
    baseline = df[df["compression"] == "baseline"].copy()

    for _, base_row in baseline.iterrows():
        model = base_row["model"]
        mask = df["model"] == model
        for col in metric_cols:
            base_val = base_row.get(col)
            if base_val and base_val > 0:
                df.loc[mask, f"{col}_drop%"] = (
                    (df.loc[mask, col] - base_val) / base_val * 100
                ).round(2)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect and display all benchmark results")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--output-csv", default="results/all_results.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = load_all_summaries(results_dir)

    if not rows:
        print("No summary.json files found under", results_dir)
        return

    df = pd.DataFrame(rows)
    df = compute_degradation(df)

    print("\n=== Full Results Table ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
