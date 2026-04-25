#!/usr/bin/env bash
# ===========================================================================
# ICPR 2026 — Run ONLY missing experiments (skip already-completed ones)
#
# Already done (in paper):
#   Qwen3-8B   : BF16 baseline + KV-Quant (EN + IndicSentiment Hi/Or)
#   Phi-4-Mini : KV-Quant (EN + IndicSentiment Hi/Or)
#
# Missing / failed:
#   1. Phi-4-Mini BF16 baseline (EN benchmarks — previously errored)
#   2. Llama-3.1-8B  BF16 + KV-Quant (EN + Multilingual)
#   3. Gemma-3-12B   BF16 + KV-Quant (EN + Multilingual)
#   4. MGSM-Bengali + IndicQA Hi/Or for Qwen3-8B + Phi-4-Mini (prev. failed)
#   5. FLORES-200 (new): Qwen3-8B + Phi-4-Mini, BF16 + KV-Quant
#
# Run inside tmux:
#   tmux new -s icpr_missing
#   bash scripts/run_missing.sh
# Monitor: tail -f results/missing_run.log
# ===========================================================================
set -uo pipefail
cd /root/icpr
source .env
source .venv/bin/activate

LOGFILE="results/missing_run.log"
mkdir -p results
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================================"
echo "  ICPR 2026 — Missing Experiments Run"
echo "  Started : $(date)"
echo "  GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

EN_BENCHES="gsm8k boolq msmarco"
MULTI_BENCHES="mgsm_bn indicqa_hi indicqa_or indic_sentiment_hi indic_sentiment_or"
FLORES_BENCHES="flores_hi flores_or flores_bn flores_ta flores_te"

run_en() {
    local model_id=$1 model_key=$2 compression=$3
    echo ""
    echo "── EN: $model_key / $compression ──────────────────────────"
    python3 evaluation/run_all.py \
        --model-id    "$model_id" \
        --model-path  "$model_id" \
        --compression "$compression" \
        --benchmarks  $EN_BENCHES \
        --output-dir  "results/${model_key}/${compression}" \
    && echo "  [OK] $model_key / $compression / EN" \
    || echo "  [WARN] $model_key / $compression / EN had errors"
}

run_multi() {
    local model_id=$1 model_key=$2 compression=$3
    echo ""
    echo "── MULTILINGUAL: $model_key / $compression ─────────────────"
    python3 evaluation/run_all.py \
        --model-id    "$model_id" \
        --model-path  "$model_id" \
        --compression "$compression" \
        --benchmarks  $MULTI_BENCHES \
        --output-dir  "results/${model_key}/multilingual_${compression}" \
    && echo "  [OK] $model_key / $compression / MULTILINGUAL" \
    || echo "  [WARN] $model_key / $compression / MULTILINGUAL had errors"
}

run_flores() {
    local model_id=$1 model_key=$2 compression=$3
    echo ""
    echo "── FLORES-200: $model_key / $compression ────────────────────"
    python3 evaluation/run_all.py \
        --model-id    "$model_id" \
        --model-path  "$model_id" \
        --compression "$compression" \
        --benchmarks  $FLORES_BENCHES \
        --max-samples 200 \
        --output-dir  "results/${model_key}/flores_${compression}" \
    && echo "  [OK] $model_key / $compression / FLORES" \
    || echo "  [WARN] $model_key / $compression / FLORES had errors"
}

# ── 1. Phi-4-Mini BF16 baseline (English only — previously failed) ──────────
echo ""
echo "########################################################"
echo "  [1/5] Phi-4-Mini BF16 Baseline (English)"
echo "########################################################"
run_en "microsoft/Phi-4-mini-instruct" "phi4_mini" "baseline"

# ── 2. Llama-3.1-8B: BF16 + KV-Quant ───────────────────────────────────────
echo ""
echo "########################################################"
echo "  [2/5] Llama-3.1-8B"
echo "########################################################"
for COMP in baseline kv_compress; do
    run_en "meta-llama/Llama-3.1-8B-Instruct" "llama3_8b" "$COMP"
    run_multi "meta-llama/Llama-3.1-8B-Instruct" "llama3_8b" "$COMP"
done

# ── 3. Gemma-3-12B: BF16 + KV-Quant ────────────────────────────────────────
echo ""
echo "########################################################"
echo "  [3/5] Gemma-3-12B"
echo "########################################################"
for COMP in baseline kv_compress; do
    run_en "google/gemma-3-12b-it" "gemma3_12b" "$COMP"
    run_multi "google/gemma-3-12b-it" "gemma3_12b" "$COMP"
done

# ── 4. Fix MGSM-Bengali + IndicQA for Qwen3-8B + Phi-4-Mini ────────────────
echo ""
echo "########################################################"
echo "  [4/5] MGSM-Bengali + IndicQA fix (Qwen3-8B + Phi-4-Mini)"
echo "########################################################"
MGSM_INDICQA_BENCHES="mgsm_bn indicqa_hi indicqa_or"
for MODEL_ID in "Qwen/Qwen3-8B" "microsoft/Phi-4-mini-instruct"; do
    MODEL_KEY=$(echo "$MODEL_ID" | sed 's|Qwen/Qwen3-8B|qwen3_8b|; s|microsoft/Phi-4-mini-instruct|phi4_mini|')
    for COMP in baseline kv_compress; do
        echo ""
        echo "── MGSM+IndicQA: $MODEL_KEY / $COMP ──"
        python3 evaluation/run_all.py \
            --model-id    "$MODEL_ID" \
            --model-path  "$MODEL_ID" \
            --compression "$COMP" \
            --benchmarks  $MGSM_INDICQA_BENCHES \
            --output-dir  "results/${MODEL_KEY}/multilingual_${COMP}" \
        && echo "  [OK]" || echo "  [WARN] errors"
    done
done

# ── 5. FLORES-200: Qwen3-8B + Phi-4-Mini ────────────────────────────────────
echo ""
echo "########################################################"
echo "  [5/5] FLORES-200 (Qwen3-8B + Phi-4-Mini)"
echo "########################################################"
for MODEL_ID in "Qwen/Qwen3-8B" "microsoft/Phi-4-mini-instruct"; do
    MODEL_KEY=$(echo "$MODEL_ID" | sed 's|Qwen/Qwen3-8B|qwen3_8b|; s|microsoft/Phi-4-mini-instruct|phi4_mini|')
    for COMP in baseline kv_compress; do
        run_flores "$MODEL_ID" "$MODEL_KEY" "$COMP"
    done
done

# ── Aggregate all results ────────────────────────────────────────────────────
echo ""
echo ">>> Aggregating all results..."
python3 evaluation/collect_results.py --results-dir results/

echo ""
echo "========================================================"
echo "  MISSING EXPERIMENTS COMPLETE: $(date)"
echo "  Log: $LOGFILE"
echo "========================================================"
