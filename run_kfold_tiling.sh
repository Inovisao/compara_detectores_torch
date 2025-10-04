#!/bin/bash
#
# K-Fold Grid Tiling Pipeline Runner
#
# This script sets up the environment and runs the k-fold tiling pipeline.
#
# Usage:
#   ./run_kfold_tiling.sh
#
# Requirements:
#   - Conda environment 'create-dataset' (create with: conda env create -f create_dataset/environment.yml)
#   - Source images in ./dataset/all/train/
#   - Fold JSON files in ./dataset/all/filesJSON/
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=============================================================================="
echo "K-Fold Grid Tiling Pipeline"
echo "=============================================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found in PATH${NC}"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "^create-dataset "; then
    echo -e "${YELLOW}Warning: 'create-dataset' conda environment not found${NC}"
    echo "Creating environment from create_dataset/environment.yml..."
    conda env create -f "$SCRIPT_DIR/create_dataset/environment.yml"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create conda environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}Environment created successfully${NC}"
fi

# Check required directories
if [ ! -d "$SCRIPT_DIR/dataset/all/train" ]; then
    echo -e "${RED}Error: Source images directory not found: $SCRIPT_DIR/dataset/all/train${NC}"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/dataset/all/filesJSON" ]; then
    echo -e "${RED}Error: Fold JSON directory not found: $SCRIPT_DIR/dataset/all/filesJSON${NC}"
    exit 1
fi

# Check if fold JSON files exist
FOLD_COUNT=$(ls "$SCRIPT_DIR/dataset/all/filesJSON"/fold_*_train.json 2>/dev/null | wc -l)
if [ "$FOLD_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No fold JSON files found in $SCRIPT_DIR/dataset/all/filesJSON${NC}"
    exit 1
fi

echo "Found $FOLD_COUNT folds"
echo ""

# Activate conda environment and run the script
echo "Activating conda environment 'create-dataset'..."
eval "$(conda shell.bash hook)"
conda activate create-dataset

echo "Running k-fold tiling pipeline..."
echo ""

cd "$SCRIPT_DIR"
python3 utils/generate_kfold_tiles.py

EXIT_CODE=$?

# Deactivate conda environment
conda deactivate

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================================================================="
    echo "Pipeline completed successfully!"
    echo -e "==============================================================================${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}=============================================================================="
    echo "Pipeline failed with exit code $EXIT_CODE"
    echo -e "==============================================================================${NC}"
    exit $EXIT_CODE
fi
