#!/usr/bin/env bash
# Run GPTQ 4-bit quantization + benchmarks for all three models.
# Owner: Sakshi and Mahi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MAX_SAMPLES=${MAX_SAMPLES:-200}   # Override with: MAX_SAMPLES=1319 ./scripts/run_gptq.sh
QUANT_OUT="./quantized/gptq"

declare -A MODELS=(
    ["qwen3_8b"]="Qwen/Qwen3-8B"
    ["phi4_mini"]="microsoft/Phi-4-mini-instruct"
    # ["gemma4_12b"]="google/gemma-4-12b"  # uncomment when available
)

for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_ID="${MODELS[$MODEL_KEY]}"
    QUANT_PATH="$QUANT_OUT/$MODEL_KEY"
    RESULT_DIR="./results/$MODEL_KEY/gptq"

    echo "============================================="
    echo "[GPTQ] Quantizing: $MODEL_ID"
    echo "============================================="
    python compression/gptq_quantize.py \
        --model "$MODEL_ID" \
        --output-dir "$QUANT_PATH" \
        --config config/compression.yaml

    echo "============================================="
    echo "[GPTQ] Running benchmarks: $MODEL_ID"
    echo "============================================="
    python evaluation/run_all.py \
        --model-id "$MODEL_ID" \
        --model-path "$QUANT_PATH" \
        --compression gptq \
        --benchmarks gsm8k boolq msmarco \
        --max-samples "$MAX_SAMPLES" \
        --output-dir "$RESULT_DIR"

    echo "Done: $MODEL_KEY → $RESULT_DIR"
done

echo ""
echo "All GPTQ experiments complete. Collecting results..."
python evaluation/collect_results.py --results-dir ./results
