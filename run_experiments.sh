#!/usr/bin/env bash
# Master experiment runner — runs baseline + all compression methods
# Launched inside tmux so it survives disconnection.
# Log: /root/icpr/results/experiment.log
set -uo pipefail

cd /root/icpr
source .venv/bin/activate

LOGFILE="/root/icpr/results/experiment.log"
mkdir -p /root/icpr/results

exec > >(tee -a "$LOGFILE") 2>&1
echo "========================================"
echo "ICPR 2026 LLM Compression Experiments"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

MAX_SAMPLES=${MAX_SAMPLES:-200}
export MAX_SAMPLES

# ── 1. BASELINE (BF16) ────────────────────────────────────────────
echo ""
echo ">>> [1/4] Running BF16 Baseline..."
bash scripts/run_baseline.sh

# ── 2. GPTQ 4-bit ─────────────────────────────────────────────────
echo ""
echo ">>> [2/4] Running GPTQ 4-bit..."
bash scripts/run_gptq.sh

# ── 3. AWQ 4-bit ──────────────────────────────────────────────────
echo ""
echo ">>> [3/4] Running AWQ 4-bit..."
bash scripts/run_awq.sh

# ── 4. KV-Cache Compression (TurboQuant-style) ────────────────────
echo ""
echo ">>> [4/4] Running KV-Cache Compression..."
bash scripts/run_kv_compress.sh

# ── Final aggregation ─────────────────────────────────────────────
echo ""
echo ">>> Aggregating English results..."
python3 evaluation/collect_results.py --results-dir results/

# ── 5. Multilingual Benchmarks ────────────────────────────────────
echo ""
echo ">>> [5/5] Running Multilingual Benchmarks (Hindi + Odia)..."

MULTILINGUAL_BENCHMARKS="mgsm_hi indicqa_or indicqa_hi indic_sentiment_hi indic_sentiment_or"

for MODEL_ID in "Qwen/Qwen3-8B" "microsoft/Phi-4-mini-instruct"; do
    MODEL_KEY=$(echo "$MODEL_ID" | tr '/' '_' | tr '[:upper:]' '[:lower:]' | sed 's/qwen\/qwen3-8b/qwen3_8b/; s/microsoft\/phi-4-mini-instruct/phi4_mini/')
    for COMPRESSION in baseline kv_compress; do
        echo ""
        echo "=== Multilingual: $MODEL_ID / $COMPRESSION ==="
        python3 evaluation/run_all.py \
            --model-id "$MODEL_ID" \
            --model-path "$MODEL_ID" \
            --compression "$COMPRESSION" \
            --benchmarks multilingual \
            --max-samples "$MAX_SAMPLES" \
            --output-dir "results/${MODEL_KEY}/multilingual_${COMPRESSION}"
    done
done

echo ""
echo ">>> Aggregating all results (English + Multilingual)..."
python3 evaluation/collect_results.py --results-dir results/

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "Results: /root/icpr/results/all_results.csv"
echo "========================================"
