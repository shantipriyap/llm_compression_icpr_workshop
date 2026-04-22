#!/usr/bin/env bash
# Full environment setup + experiment launcher for ICPR 2026
set -euo pipefail
cd /root/icpr
source .venv/bin/activate

echo "=== Installing PyTorch (CUDA 12.8) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

echo "=== Installing core dependencies ==="
pip install "transformers>=4.45.0" "accelerate>=0.30.0" "datasets>=2.19.0" \
    "evaluate>=0.4.1" pyyaml tqdm pandas numpy rouge-score \
    sentencepiece protobuf bitsandbytes -q

echo "=== Installing auto-gptq ==="
pip install auto-gptq -q || echo "auto-gptq install failed, continuing"

echo "=== Installing autoawq ==="
pip install autoawq -q || echo "autoawq install failed, continuing"

echo "=== Verifying GPU ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== SETUP COMPLETE ==="
