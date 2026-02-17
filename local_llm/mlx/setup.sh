#!/bin/bash
# One-time setup script for MLX on Apple Silicon
#
# Usage:
#   ./local_llm/mlx/setup.sh
#
# This creates (in project root):
#   - .venv-mlx/ (MLX server environment)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_LLM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$LOCAL_LLM_DIR")"
cd "$PROJECT_DIR"

echo "=============================================="
echo "  MLX Setup for SpatialAgent (Apple Silicon)"
echo "=============================================="
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check for Apple Silicon
if [[ "$(uname -m)" != "arm64" ]] || [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Warning: This setup is designed for Apple Silicon Macs."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

echo "Step 1/2: Creating MLX environment..."
if [ -d ".venv-mlx" ]; then
    echo "  .venv-mlx already exists, skipping..."
else
    uv venv .venv-mlx --python 3.12
    echo "  Installing MLX dependencies..."
    uv pip install --python .venv-mlx/bin/python -r "$LOCAL_LLM_DIR/mlx/requirements.txt"
    uv pip install --python .venv-mlx/bin/python 'litellm[proxy]'
fi

echo ""
echo "Step 2/2: Verifying installation..."
.venv-mlx/bin/python -c "import mlx; print(f'  MLX version: {mlx.__version__}')" 2>/dev/null || echo "  MLX not available (expected on non-Mac)"
.venv-mlx/bin/python -c "import litellm; print(f'  LiteLLM version: {litellm.__version__}')"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Start servers:  ./local_llm/mlx/start.sh"
echo "  2. Check status:   ./local_llm/mlx/start.sh status"
echo "  3. Stop servers:   ./local_llm/mlx/start.sh stop"
echo ""
