#!/bin/bash
#
# Quick Test Script - Train a single model on a single fold
#
# Usage:
#   ./quick_test.sh [MODEL] [FOLD_NUM]
#
# Examples:
#   ./quick_test.sh YOLOV8 1
#   ./quick_test.sh Faster 2
#   ./quick_test.sh YOLOV5_TPH 3
#
# Available models: YOLOV8, Faster, YOLOV5_TPH
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
MODEL=${1:-YOLOV8}
FOLD_NUM=${2:-1}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}=============================================================================="
echo "Quick Test - Single Model Training"
echo -e "==============================================================================${NC}"
echo ""
echo "  Model: $MODEL"
echo "  Fold: fold_$FOLD_NUM"
echo ""

# Validate model
case "$MODEL" in
    YOLOV8|Faster|YOLOV5_TPH)
        ;;
    *)
        echo -e "${RED}Error: Invalid model '$MODEL'${NC}"
        echo "Available models: YOLOV8, Faster, YOLOV5_TPH"
        exit 1
        ;;
esac

# Check tiled dataset
FOLD_DIR="$SCRIPT_DIR/dataset/tiles/grid/fold_$FOLD_NUM"
if [ ! -d "$FOLD_DIR" ]; then
    echo -e "${RED}Error: Fold directory not found: $FOLD_DIR${NC}"
    echo "Available folds:"
    ls -d "$SCRIPT_DIR/dataset/tiles/grid"/fold_* 2>/dev/null || echo "  None found"
    exit 1
fi

echo -e "${GREEN}✓ Found fold_$FOLD_NUM${NC}"

# Verify splits
for split in train val test; do
    if [ ! -d "$FOLD_DIR/$split" ]; then
        echo -e "${RED}Error: Missing $split split in fold_$FOLD_NUM${NC}"
        exit 1
    fi
    img_count=$(ls "$FOLD_DIR/$split"/*.jpg 2>/dev/null | wc -l)
    echo "  $split: $img_count images"
done

echo ""

# Setup environment
export USE_TILED_DATASET=true
export MODELS_TO_RUN="$MODEL"

# Create a temporary main.py that only trains the specified fold
TEMP_MAIN="$SCRIPT_DIR/src/main_quick_test.py"

cat > "$TEMP_MAIN" <<'PYTHON_SCRIPT'
import os
import sys
import numpy as np

# Get model and fold from environment
MODEL = os.getenv('MODELS_TO_RUN', 'YOLOV8')
FOLD_NUM = int(os.getenv('FOLD_NUM', '1'))

print(f"\nQuick Test Mode: Training {MODEL} on fold_{FOLD_NUM}\n")

# Import from main.py
sys.path.insert(0, os.path.dirname(__file__))
from main import train_model, test_model, create_csv, generate_results
from main import GeraRult, save_imgs, GeraResultByClass, APENAS_TESTE

# Configuration
fold = f'fold_{FOLD_NUM}'
fold_dir = os.path.join('model_checkpoints', fold)
current_root = os.path.join('..', 'dataset', 'tiles', 'grid', fold)

if not os.path.exists(current_root):
    print(f"Error: Tiled dataset not found at {current_root}")
    sys.exit(1)

print(f"Dataset root: {current_root}")
print(f"Model checkpoint: {fold_dir}")
print("")

# Train the model
if not APENAS_TESTE:
    print(f"Training {MODEL} on {fold}...")
    model_path = train_model(MODEL, fold, fold_dir, current_root)
    if model_path is None:
        print("Training skipped (already exists with CONTINUE=True)")
        model_path = test_model(MODEL, fold_dir)
else:
    model_path = test_model(MODEL, fold_dir)

print(f"\nModel saved to: {model_path}")

# Generate results if enabled
if GeraRult:
    print("\nGenerating results...")
    create_csv(root=current_root, fold=fold, selected_model=MODEL, model_path=model_path, save_imgs=save_imgs)

if GeraResultByClass:
    print("\nGenerating per-class results...")
    generate_results(root=current_root, fold=fold, model=model_path, model_name=MODEL, save_imgs=save_imgs)

print("\n✓ Quick test completed!")
PYTHON_SCRIPT

echo -e "${YELLOW}Starting training...${NC}"
echo ""

# Activate conda if available
if conda env list 2>/dev/null | grep -q "^detectores "; then
    eval "$(conda shell.bash hook)"
    conda activate detectores
fi

cd "$SCRIPT_DIR/src"
export FOLD_NUM="$FOLD_NUM"

# Run the quick test
if python3 main_quick_test.py; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi

# Cleanup
rm -f main_quick_test.py

cd "$SCRIPT_DIR"

echo ""
echo -e "${BLUE}=============================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Quick test completed successfully!${NC}"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Results:"
    echo "  - Model checkpoint: src/model_checkpoints/fold_$FOLD_NUM/$MODEL/"
    echo "  - Results: results/results.csv"
    echo ""
    echo "To train all folds with all models, run:"
    echo "  ./train_tiled.sh"
    exit 0
else
    echo -e "${RED}Quick test failed with exit code $EXIT_CODE${NC}"
    echo -e "==============================================================================${NC}"
    exit $EXIT_CODE
fi
