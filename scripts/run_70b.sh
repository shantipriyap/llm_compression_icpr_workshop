#!/usr/bin/env bash
# ===========================================================================
# ICPR 2026 — Llama-3.1-70B-Instruct Experiment
# Compression: int4_bnb (NF4 double-quant ~35GB) + int4_bnb_kv (+ KV-compress)
# Full datasets: GSM8K=1319, BoolQ=3270, MSMARCO=1000, multilingual full splits
#
# Run AFTER full_run.sh completes:
#   tmux new -s icpr70b
#   bash scripts/run_70b.sh
# Monitor: tail -f results/llama3_70b/run_70b.log
# ===========================================================================
set -uo pipefail
cd /root/icpr
source .venv/bin/activate

export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL_ID="meta-llama/Llama-3.1-70B-Instruct"
MODEL_KEY="llama3_70b"

LOGFILE="results/${MODEL_KEY}/run_70b.log"
mkdir -p "results/${MODEL_KEY}"
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================================"
echo "  Llama-3.1-70B-Instruct Benchmark Run"
echo "  Started : $(date)"
echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Method  : int4_bnb (NF4 double-quant) + int4_bnb_kv"
echo "  Dataset : FULL (GSM8K=1319, BoolQ=3270, MSMARCO=1000)"
echo "========================================================"

ENGLISH_BENCHES="gsm8k boolq msmarco"
MULTILINGUAL_BENCHES="mgsm_bn indicqa_hi indicqa_or indic_sentiment_hi indic_sentiment_or"

for COMPRESSION in int4_bnb int4_bnb_kv; do
    echo ""
    echo "########################################################"
    echo "  $MODEL_KEY / $COMPRESSION"
    echo "  $(date)"
    echo "########################################################"

    echo ""
    echo "── English Benchmarks ──────────────────────────────────"
    python3 evaluation/run_all.py \
        --model-id    "$MODEL_ID" \
        --model-path  "$MODEL_ID" \
        --compression "$COMPRESSION" \
        --benchmarks  $ENGLISH_BENCHES \
        --output-dir  "results/${MODEL_KEY}/${COMPRESSION}" \
    && echo "  [OK] English done: $COMPRESSION" \
    || echo "  [WARN] English had errors: $COMPRESSION"

    echo ""
    echo "── Multilingual Benchmarks ─────────────────────────────"
    python3 evaluation/run_all.py \
        --model-id    "$MODEL_ID" \
        --model-path  "$MODEL_ID" \
        --compression "$COMPRESSION" \
        --benchmarks  $MULTILINGUAL_BENCHES \
        --output-dir  "results/${MODEL_KEY}/multilingual_${COMPRESSION}" \
    && echo "  [OK] Multilingual done: $COMPRESSION" \
    || echo "  [WARN] Multilingual had errors: $COMPRESSION"
done

echo ""
echo "── Aggregating results ─────────────────────────────────────"
python3 evaluation/collect_results.py --results-dir results/

echo ""
echo "========================================================"
echo "  LLAMA-70B EXPERIMENTS COMPLETE: $(date)"
echo "  Log: $LOGFILE"
echo "========================================================"
