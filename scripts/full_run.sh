#!/usr/bin/env bash
# ===========================================================================
# ICPR 2026 — Full Benchmark Run (no sample cap)
# 4 models × 2 compression (baseline + kv_compress) × 3 EN + 5 multilingual
# Estimated wall time: ~24–36 h on RTX PRO 6000 Blackwell (102 GB VRAM)
# Run inside tmux:  tmux new -s icpr && bash scripts/full_run.sh
# Monitor:          tmux attach -t icpr
#                   tail -f results/full_run.log
# ===========================================================================
set -uo pipefail
cd /root/icpr
source .venv/bin/activate

export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

LOGFILE="results/full_run.log"
mkdir -p results
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================================"
echo "  ICPR 2026 Full Benchmark Experiment"
echo "  Started : $(date)"
echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Datasets: FULL (GSM8K=1319, BoolQ=3270, MSMARCO=1000)"
echo "  Models  : Qwen3-8B | Phi-4-Mini | Llama-3.1-8B | Gemma-3-12B"
echo "  Methods : baseline | kv_compress"
echo "  Langs   : EN + Hindi + Odia + Bengali"
echo "========================================================"

# Full dataset sizes (no --max-samples flag → evaluation scripts use complete splits)
ENGLISH_BENCHES="gsm8k boolq msmarco"
MULTILINGUAL_BENCHES="mgsm_bn indicqa_hi indicqa_or indic_sentiment_hi indic_sentiment_or"

declare -A MODELS=(
    ["qwen3_8b"]="Qwen/Qwen3-8B"
    ["phi4_mini"]="microsoft/Phi-4-mini-instruct"
    ["llama3_8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["gemma3_12b"]="google/gemma-3-12b-it"
)

# Run one model at a time to avoid OOM during sequential loading
MODEL_ORDER="qwen3_8b phi4_mini llama3_8b gemma3_12b"

for MODEL_KEY in $MODEL_ORDER; do
    MODEL_ID="${MODELS[$MODEL_KEY]}"
    echo ""
    echo "########################################################"
    echo "  MODEL: $MODEL_ID  ($MODEL_KEY)"
    echo "  $(date)"
    echo "########################################################"

    for COMPRESSION in baseline kv_compress; do
        echo ""
        echo "──────────────────────────────────────────"
        echo "  $MODEL_KEY / $COMPRESSION / ENGLISH"
        echo "──────────────────────────────────────────"
        python3 evaluation/run_all.py \
            --model-id    "$MODEL_ID" \
            --model-path  "$MODEL_ID" \
            --compression "$COMPRESSION" \
            --benchmarks  $ENGLISH_BENCHES \
            --output-dir  "results/${MODEL_KEY}/${COMPRESSION}" \
        && echo "  [OK] English benchmarks done: $MODEL_KEY / $COMPRESSION" \
        || echo "  [WARN] English benchmarks had errors: $MODEL_KEY / $COMPRESSION"

        echo ""
        echo "──────────────────────────────────────────"
        echo "  $MODEL_KEY / $COMPRESSION / MULTILINGUAL"
        echo "──────────────────────────────────────────"
        python3 evaluation/run_all.py \
            --model-id    "$MODEL_ID" \
            --model-path  "$MODEL_ID" \
            --compression "$COMPRESSION" \
            --benchmarks  $MULTILINGUAL_BENCHES \
            --output-dir  "results/${MODEL_KEY}/multilingual_${COMPRESSION}" \
        && echo "  [OK] Multilingual benchmarks done: $MODEL_KEY / $COMPRESSION" \
        || echo "  [WARN] Multilingual benchmarks had errors: $MODEL_KEY / $COMPRESSION"
    done
done

echo ""
echo "========================================================"
echo "  Aggregating all results..."
echo "========================================================"
python3 evaluation/collect_results.py --results-dir results/

echo ""
echo "========================================================"
echo "  ALL EXPERIMENTS COMPLETE: $(date)"
echo "  Results: results/full_run.log"
echo "========================================================"
