#!/usr/bin/env bash
# Re-run only the experiments that failed in the initial run:
#   - Phi-4-Mini: baseline, gptq, awq, kv_compress (all benchmarks)
#   - Qwen3-8B GPTQ: all benchmarks (using already-quantized model)
#   - Qwen3-8B AWQ: all benchmarks (using already-quantized model)
set -uo pipefail
cd /root/icpr
source .venv/bin/activate

LOGFILE="results/rerun.log"
mkdir -p results
exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "ICPR Re-run: Failed Experiments"
echo "Started: $(date)"
echo "========================================"

MAX_SAMPLES=${MAX_SAMPLES:-200}
export HF_TOKEN=${HF_TOKEN:-""}

# ── Phi-4-Mini: Baseline ───────────────────────────────────────────
echo ""
echo ">>> Phi-4-Mini BF16 Baseline"
python3 evaluation/run_all.py \
    --model-id "microsoft/Phi-4-mini-instruct" \
    --model-path "microsoft/Phi-4-mini-instruct" \
    --compression baseline \
    --benchmarks gsm8k boolq msmarco \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "results/phi4_mini/baseline"

# ── Qwen3-8B GPTQ benchmarks (model already quantized) ────────────
echo ""
echo ">>> Qwen3-8B GPTQ Benchmarks (re-using quantized model)"
python3 evaluation/run_all.py \
    --model-id "Qwen/Qwen3-8B" \
    --model-path "./quantized/gptq/qwen3_8b" \
    --compression gptq \
    --benchmarks gsm8k boolq msmarco \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "results/qwen3_8b/gptq"

# ── Qwen3-8B AWQ benchmarks (model already quantized) ─────────────
echo ""
echo ">>> Qwen3-8B AWQ Benchmarks (re-using quantized model)"
python3 evaluation/run_all.py \
    --model-id "Qwen/Qwen3-8B" \
    --model-path "./quantized/awq/qwen3_8b" \
    --compression awq \
    --benchmarks gsm8k boolq msmarco \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "results/qwen3_8b/awq"

# ── Phi-4-Mini: GPTQ ──────────────────────────────────────────────
echo ""
echo ">>> Phi-4-Mini GPTQ Quantize + Benchmarks"
python3 compression/gptq_quantize.py \
    --model "microsoft/Phi-4-mini-instruct" \
    --output-dir "./quantized/gptq/phi4_mini" \
    --config config/compression.yaml || echo "GPTQ quantization failed, skipping"

python3 evaluation/run_all.py \
    --model-id "microsoft/Phi-4-mini-instruct" \
    --model-path "./quantized/gptq/phi4_mini" \
    --compression gptq \
    --benchmarks gsm8k boolq msmarco \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "results/phi4_mini/gptq"

# ── Phi-4-Mini: AWQ ───────────────────────────────────────────────
echo ""
echo ">>> Phi-4-Mini AWQ Quantize + Benchmarks"
python3 compression/awq_quantize.py \
    --model "microsoft/Phi-4-mini-instruct" \
    --output-dir "./quantized/awq/phi4_mini" \
    --config config/compression.yaml || echo "AWQ quantization failed, skipping"

python3 evaluation/run_all.py \
    --model-id "microsoft/Phi-4-mini-instruct" \
    --model-path "./quantized/awq/phi4_mini" \
    --compression awq \
    --benchmarks gsm8k boolq msmarco \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "results/phi4_mini/awq"

# ── Phi-4-Mini: KV-compress ───────────────────────────────────────
echo ""
echo ">>> Phi-4-Mini KV-Cache Compression Benchmarks"
python3 evaluation/run_all.py \
    --model-id "microsoft/Phi-4-mini-instruct" \
    --model-path "microsoft/Phi-4-mini-instruct" \
    --compression kv_compress \
    --benchmarks gsm8k boolq msmarco \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "results/phi4_mini/kv_compress"

# ── Final aggregation ─────────────────────────────────────────────
echo ""
echo ">>> Aggregating all results..."
python3 evaluation/collect_results.py --results-dir results/

echo ""
echo "========================================"
echo "RE-RUN COMPLETE: $(date)"
echo "========================================"
