#!/usr/bin/env bash
# Run TurboQuant-style KV-cache compression benchmarks for all three models.
# Owner: Debasish
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

export HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN env var must be set before running}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

declare -A MODELS=(
    ["qwen3_8b"]="Qwen/Qwen3-8B"
    ["phi4_mini"]="microsoft/Phi-4-mini-instruct"
    ["llama3_8b"]="meta-llama/Llama-3.1-8B-Instruct"
    ["gemma3_12b"]="google/gemma-3-12b-it"
)

for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$MODEL_KEY]}"
    RESULT_DIR="./results/$MODEL_KEY/kv_compress"

    echo "============================================="
    echo "[TurboQuant KV-compress] Running benchmarks: $MODEL_ID"
    echo "============================================="
    python evaluation/run_all.py \
        --model-id "$MODEL_ID" \
        --model-path "$MODEL_ID" \
        --compression kv_compress \
        --benchmarks gsm8k boolq msmarco \
        --output-dir "$RESULT_DIR"

    echo "Done: $MODEL_KEY → $RESULT_DIR"
done

echo ""
echo "All KV-compress experiments complete. Collecting results..."
python evaluation/collect_results.py --results-dir ./results
