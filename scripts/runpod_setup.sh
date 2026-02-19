#!/bin/bash
# RunPod B200 pod initialization script
# Usage: bash scripts/runpod_setup.sh

set -euo pipefail

echo "=== NameAI RunPod Setup ==="

apt-get update && apt-get install -y git vim tmux htop

# Install project
pip install -e ".[dev]"

# Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'BF16: {torch.cuda.is_bf16_supported()}')
"

mkdir -p data/raw data/processed checkpoints

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. python -m nameai.data.build_dataset"
echo "  2. python -m nameai.training.trainer"
echo "  3. nameai generate 'AI article generator'"
