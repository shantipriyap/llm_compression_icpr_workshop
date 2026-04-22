#!/usr/bin/env bash
# run_analysis.sh — Run all three novel diagnostic experiments on the server.
# Executes: token_fertility, cross_lingual_consistency, activation_analysis
#
# Usage (inside /root/icpr on the server):
#   source .venv/bin/activate
#   bash scripts/run_analysis.sh [model_id]
#
# Default model: Qwen/Qwen3-8B
# Override:      bash scripts/run_analysis.sh microsoft/Phi-4-mini-instruct

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-8B}"
LOGDIR="results/analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "======================================================"
echo "  Novel Diagnostic Analysis Suite"
echo "  Model  : $MODEL"
echo "  Results: $LOGDIR"
echo "======================================================"

# ── 1. Token Fertility (tokenizer-only, CPU, fast) ─────────────────────────
echo ""
echo "[1/3] Token Fertility Analysis (CPU, all 5 models)..."
python3 benchmarks/token_fertility.py \
    --models \
        google/gemma-3-1b-it \
        Qwen/Qwen3-1.7B \
        microsoft/Phi-4-mini-instruct \
        Qwen/Qwen3-8B \
        meta-llama/Llama-3.1-8B-Instruct \
    --languages en hi mr gu or bn ta te kn ml \
    --output "$LOGDIR/token_fertility.json" \
    2>&1 | tee "$LOGDIR/token_fertility.log"

echo "[1/3] Token Fertility DONE → $LOGDIR/token_fertility.json"

# ── 2. Activation Analysis (GPU, single model) ─────────────────────────────
echo ""
echo "[2/3] Activation Analysis (GPU) for $MODEL..."
python3 benchmarks/activation_analysis.py \
    --model "$MODEL" \
    --num_samples 16 \
    --languages en hi or bn ta te \
    --plot \
    --output "$LOGDIR/activation_stats.json" \
    2>&1 | tee "$LOGDIR/activation_analysis.log"

echo "[2/3] Activation Analysis DONE → $LOGDIR/activation_stats.json"

# Compressed variant
echo ""
echo "[2b/3] Activation Analysis (NF4 compressed) for $MODEL..."
python3 benchmarks/activation_analysis.py \
    --model "$MODEL" \
    --quantize_kv \
    --num_samples 16 \
    --languages en hi or bn ta te \
    --plot \
    --output "$LOGDIR/activation_stats_nf4.json" \
    2>&1 | tee "$LOGDIR/activation_analysis_nf4.log"

echo "[2b/3] Activation Analysis NF4 DONE → $LOGDIR/activation_stats_nf4.json"

# ── 3. Cross-Lingual Consistency (GPU, baseline + NF4) ─────────────────────
echo ""
echo "[3/3] Cross-Lingual Consistency (GPU) for $MODEL..."
python3 benchmarks/cross_lingual_consistency.py \
    --model "$MODEL" \
    --languages en hi mr gu or bn ta te kn ml \
    --max_new_tokens 60 \
    --output "$LOGDIR/consistency.json" \
    2>&1 | tee "$LOGDIR/consistency.log"

echo "[3/3] Cross-Lingual Consistency DONE → $LOGDIR/consistency.json"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  All analyses complete."
echo "  Results saved to: $LOGDIR/"
ls -lh "$LOGDIR/"
echo "======================================================"
