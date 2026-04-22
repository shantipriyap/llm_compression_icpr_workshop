#!/usr/bin/env bash
# Run AWQ 4-bit quantization + benchmarks for all three models.
# Owner: Subrat, Anurag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MAX_SAMPLES=${MAX_SAMPLES:-200}
QUANT_OUT="./quantized/awq"

declare -A MODELS=(
    ["qwen3_8b"]="Qwen/Qwen3-8B"
    ["phi4_mini"]="microsoft/Phi-4-mini-instruct"
    # ["gemma4_12b"]="google/gemma-4-12b"  # uncomment when available
)

for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$MODEL_KEY]}"
    QUANT_PATH="$QUANT_OUT/$MODEL_KEY"
    RESULT_DIR="./results/$MODEL_KEY/awq"

    echo "============================================="
    echo "[AWQ] Quantizing: $MODEL_ID"
    echo "============================================="
    python compression/awq_quantize.py \
        --model "$MODEL_ID" \
        --output-dir "$QUANT_PATH" \
        --config config/compression.yaml

    echo "============================================="
    echo "[AWQ] Running benchmarks: $MODEL_ID"
    echo "============================================="
    python evaluation/run_all.py \
        --model-id "$MODEL_ID" \
        --model-path "$QUANT_PATH" \
        --compression awq \
        --benchmarks gsm8k boolq msmarco \
        --max-samples "$MAX_SAMPLES" \
        --output-dir "$RESULT_DIR"

    echo "Done: $MODEL_KEY → $RESULT_DIR"
done

echo ""
echo "All AWQ experiments complete. Collecting results..."
python evaluation/collect_results.py --results-dir ./results
