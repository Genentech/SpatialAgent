#!/bin/bash
# SpatialAgent Environment Setup Script
# Creates a conda environment with Python 3.11 and all required packages
#
# Usage:
#   ./setup_env.sh [env_name]
#
# Default environment name: spatial_agent

set -e  # Exit on error

ENV_NAME="${1:-spatial_agent}"
CONDA_BASE=$(conda info --base)

echo "==========================================="
echo "SpatialAgent Environment Setup"
echo "==========================================="
echo "Environment name: $ENV_NAME"
echo "Conda base: $CONDA_BASE"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "WARNING: Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Aborting."
        exit 1
    fi
fi

echo ""
echo "Step 1: Creating conda environment with Python 3.11..."
conda create -n "$ENV_NAME" python=3.11 -y

# Activate environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ""
echo "Step 2: Installing packages from requirements.txt..."

# Upgrade pip
pip install --upgrade pip

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install all packages from requirements.txt
pip install -r "$SCRIPT_DIR/resource/requirements.txt"

echo ""
echo "Step 3: Installing UTAG (Python 3.12 compatible fork)..."
pip install -e "$SCRIPT_DIR/resource/utag"

echo ""
echo "Step 4: Verifying installation..."

python -c "
import sys
print(f'Python: {sys.version}')
print()

modules = [
    ('scanpy', 'scanpy'),
    ('anndata', 'anndata'),
    ('squidpy', 'squidpy'),
    ('scvi', 'scvi-tools'),
    ('cell2location', 'cell2location'),
    ('SpaGCN', 'SpaGCN'),
    ('GraphST', 'GraphST'),
    ('ot', 'POT'),
    ('tangram', 'tangram-sc'),
    ('liana', 'liana'),
    ('langchain', 'langchain'),
    ('anthropic', 'anthropic'),
    ('torch', 'pytorch'),
    ('leidenalg', 'leidenalg'),
    ('bbknn', 'bbknn'),
    ('scvelo', 'scvelo'),
    ('cellrank', 'cellrank'),
    ('mofapy2', 'mofapy2'),
    ('Bio', 'biopython'),
    ('utag', 'utag'),
    ('cellphonedb', 'cellphonedb'),
]

print('Package verification:')
failed = []
for mod, name in modules:
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', 'OK')
        print(f'  {name}: {v}')
    except Exception as e:
        print(f'  {name}: FAILED')
        failed.append(name)

if failed:
    print(f'\\nWARNING: {len(failed)} package(s) failed to import: {failed}')
    sys.exit(1)
else:
    print('\\nAll packages verified successfully!')
"

echo ""
echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To register as Jupyter kernel:"
echo "  python -m ipykernel install --user --name $ENV_NAME --display-name \"Python ($ENV_NAME)\""
echo ""
