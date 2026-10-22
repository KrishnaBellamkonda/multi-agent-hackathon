#!/bin/bash
# Setup script for reproducibility environment

set -e

echo "=============================================="
echo "Setting up reproducibility environment"
echo "=============================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ "$python_version" < "3.8" ]]; then
    echo "Error: Python 3.8+ required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import numpy as np
import scipy
import pandas as pd
import sklearn
import torch
import gpytorch
import matplotlib
import seaborn
import yaml
import tqdm

print('All dependencies installed successfully!')
print(f'  numpy: {np.__version__}')
print(f'  scipy: {scipy.__version__}')
print(f'  pandas: {pd.__version__}')
print(f'  sklearn: {sklearn.__version__}')
print(f'  torch: {torch.__version__}')
print(f'  gpytorch: {gpytorch.__version__}')
"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the ablation study:"
echo "  python scripts/run_full_ablation.py"
echo ""
