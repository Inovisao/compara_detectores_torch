#!/bin/bash
#
# Lazy Training Script for Tiled Datasets
#
# This script automates the entire training process using tiled k-fold datasets.
# Just run: ./train_tiled.sh
#
# What it does:
# 1. Verifies tiled datasets exist
# 2. Activates the conda environment
# 3. Sets up environment variables
# 4. Runs training on all folds with all models
# 5. Generates results and predictions
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}=============================================================================="
echo "Lazy Training Script for Tiled Datasets"
echo -e "==============================================================================${NC}"
echo ""

# Step 1: Check if tiled datasets exist
echo -e "${YELLOW}[1/5] Checking tiled datasets...${NC}"
TILES_DIR="$SCRIPT_DIR/dataset/tiles/grid"

if [ ! -d "$TILES_DIR" ]; then
    echo -e "${RED}Error: Tiled datasets not found at $TILES_DIR${NC}"
    echo ""
    echo "Please run the tiling pipeline first:"
    echo "  ./run_kfold_tiling.sh"
    exit 1
fi

# Count folds
FOLD_COUNT=$(ls -d "$TILES_DIR"/fold_* 2>/dev/null | wc -l)
if [ "$FOLD_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No fold directories found in $TILES_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found $FOLD_COUNT folds in $TILES_DIR${NC}"

# Verify each fold has train/val/test
echo "  Verifying fold structure..."
for fold_dir in "$TILES_DIR"/fold_*; do
    fold_name=$(basename "$fold_dir")
    missing=""

    for split in train val test; do
        if [ ! -d "$fold_dir/$split" ]; then
            missing="$missing $split"
        elif [ ! -f "$fold_dir/$split/_annotations.coco.json" ]; then
            missing="$missing $split(no annotations)"
        fi
    done

    if [ -n "$missing" ]; then
        echo -e "${RED}  ✗ $fold_name: Missing$missing${NC}"
        exit 1
    else
        echo -e "${GREEN}  ✓ $fold_name: train/val/test OK${NC}"
    fi
done

echo ""

# Step 2: Check conda environment
echo -e "${YELLOW}[2/5] Setting up conda environment...${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found in PATH${NC}"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if detectores environment exists
if ! conda env list | grep -q "^detectores "; then
    echo -e "${YELLOW}Warning: 'detectores' conda environment not found${NC}"
    echo "Attempting to continue with current environment..."
else
    echo -e "${GREEN}✓ Found 'detectores' conda environment${NC}"
    echo "  Activating..."
    eval "$(conda shell.bash hook)"
    conda activate detectores
    echo -e "${GREEN}✓ Activated 'detectores' environment${NC}"
fi

echo ""

# Step 3: Set environment variables
echo -e "${YELLOW}[3/5] Configuring environment variables...${NC}"

export USE_TILED_DATASET=true
echo -e "${GREEN}✓ USE_TILED_DATASET=true${NC}"

# Check if user wants to override model selection
if [ -z "$MODELS_TO_RUN" ]; then
    export MODELS_TO_RUN="Faster,YOLOV5_TPH,YOLOV8"
    echo -e "${GREEN}✓ MODELS_TO_RUN=${MODELS_TO_RUN} (default)${NC}"
else
    echo -e "${GREEN}✓ MODELS_TO_RUN=${MODELS_TO_RUN} (from environment)${NC}"
fi

echo ""

# Step 4: Display training configuration
echo -e "${YELLOW}[4/5] Training Configuration${NC}"
echo "  Number of folds: $FOLD_COUNT"
echo "  Models to train: $MODELS_TO_RUN"
echo "  Dataset type: Tiled (grid 6x7)"
echo "  Working directory: $SCRIPT_DIR/src"
echo ""

# Ask for confirmation
read -p "$(echo -e ${BLUE}Continue with training? [y/N]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""

# Step 5: Run training
echo -e "${YELLOW}[5/5] Starting training...${NC}"
echo -e "${BLUE}=============================================================================="
echo "Training in progress..."
echo -e "==============================================================================${NC}"
echo ""

cd "$SCRIPT_DIR/src"

# Run main.py and capture exit code
if python3 main.py; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

cd "$SCRIPT_DIR"

echo ""
echo -e "${BLUE}=============================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Results saved to:"
    echo "  - Model checkpoints: $SCRIPT_DIR/src/model_checkpoints/"
    echo "  - Results CSV: $SCRIPT_DIR/results/results.csv"
    echo "  - Counting CSV: $SCRIPT_DIR/results/counting.csv"
    echo "  - Predictions: $SCRIPT_DIR/results/prediction/"
    echo ""
    echo "Next steps:"
    echo "  1. Review results: cat results/results.csv"
    echo "  2. View predictions: ls results/prediction/"
    echo "  3. Analyze by fold: grep 'fold_1' results/results.csv"
    exit 0
else
    echo -e "${RED}Training failed with exit code $EXIT_CODE${NC}"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Please check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  - Missing dependencies: conda install <package>"
    echo "  - CUDA/GPU errors: Check GPU availability"
    echo "  - Dataset errors: Verify tiled datasets with test_tiled_integration.py"
    exit $EXIT_CODE
fi
