#!/bin/bash
# One-time setup script for vLLM on NVIDIA GPUs
#
# Usage:
#   ./local_llm/vllm/setup.sh
#
# This creates (in project root):
#   - .venv-vllm/    (vLLM server environment)
#   - .venv-litellm/ (LiteLLM proxy environment)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_LLM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$LOCAL_LLM_DIR")"
cd "$PROJECT_DIR"

echo "=============================================="
echo "  vLLM Setup for SpatialAgent"
echo "=============================================="
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. This setup requires NVIDIA GPUs."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

echo "Step 1/3: Creating vLLM environment..."
if [ -d ".venv-vllm" ]; then
    echo "  .venv-vllm already exists, skipping..."
else
    uv venv .venv-vllm --python 3.12
    echo "  Installing vLLM (this may take a few minutes)..."
    uv pip install --python .venv-vllm/bin/python vllm==0.11.2 ninja
fi

echo ""
echo "Step 2/3: Creating LiteLLM environment..."
if [ -d ".venv-litellm" ]; then
    echo "  .venv-litellm already exists, skipping..."
else
    uv venv .venv-litellm --python 3.12
    echo "  Installing LiteLLM proxy..."
    uv pip install --python .venv-litellm/bin/python 'litellm[proxy]'
fi

echo ""
echo "Step 3/3: Verifying installation..."
.venv-vllm/bin/python -c "import vllm; print(f'  vLLM version: {vllm.__version__}')"
.venv-litellm/bin/python -c "import litellm; print(f'  LiteLLM version: {litellm.__version__}')"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Start servers:  ./local_llm/vllm/start.sh"
echo "  2. Check status:   ./local_llm/vllm/start.sh status"
echo "  3. Stop servers:   ./local_llm/vllm/start.sh stop"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""
