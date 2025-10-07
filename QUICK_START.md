# Quick Start Guide - Tiled Dataset Training

This guide shows you how to train object detection models on tiled datasets with minimal effort.

## 🚀 TL;DR - Just Run This

```bash
# Full training (all models, all folds)
./train_tiled.sh

# Quick test (single model, single fold)
./quick_test.sh YOLOV8 1
```

That's it! Everything else is automated. ✨

---

## Prerequisites

Before running training, make sure you have:

1. ✅ Tiled datasets generated (run `./run_kfold_tiling.sh` if needed)
2. ✅ Conda environment `detectores` set up
3. ✅ All dependencies installed

## Training Options

### Option 1: Full Training (Recommended for Production)

Train all models on all folds:

```bash
./train_tiled.sh
```

**What it does:**
- ✅ Verifies tiled datasets exist
- ✅ Activates conda environment automatically
- ✅ Trains YOLOV8, Faster R-CNN, and YOLOV5_TPH
- ✅ Processes all 6 folds
- ✅ Generates results and predictions

**Customization:**

```bash
# Train only specific models
export MODELS_TO_RUN="YOLOV8,Faster"
./train_tiled.sh

# Train just one model
export MODELS_TO_RUN="YOLOV8"
./train_tiled.sh
```

### Option 2: Quick Test (Recommended for Development)

Train a single model on a single fold for quick testing:

```bash
# Syntax: ./quick_test.sh [MODEL] [FOLD_NUM]

# Examples:
./quick_test.sh YOLOV8 1          # Train YOLOV8 on fold 1
./quick_test.sh Faster 2          # Train Faster R-CNN on fold 2
./quick_test.sh YOLOV5_TPH 3      # Train YOLOV5_TPH on fold 3
```

**Available models:**
- `YOLOV8`
- `Faster` (Faster R-CNN)
- `YOLOV5_TPH`

**Benefits:**
- ⚡ Fast - only one fold/model
- 🔍 Good for testing changes
- 💾 Saves disk space during development

### Option 3: Manual Training (Advanced)

For full control:

```bash
cd src
export USE_TILED_DATASET=true
export MODELS_TO_RUN="YOLOV8"
python3 main.py
```

## Output Files

After training completes, you'll find:

```
compara_detectores_torch/
├── src/model_checkpoints/
│   ├── fold_1/
│   │   ├── YOLOV8/
│   │   │   └── train/weights/best.pt
│   │   ├── Faster/
│   │   │   └── best.pth
│   │   └── YOLOV5_TPH/
│   │       └── train/weights/best.pt
│   └── fold_2/ ... fold_6/
├── results/
│   ├── results.csv              # Main metrics (mAP, precision, recall, etc.)
│   ├── counting.csv             # Per-image counting results
│   └── prediction/              # Prediction visualizations
│       ├── YOLOV8/
│       ├── Faster/
│       └── YOLOV5_TPH/
```

## Analyzing Results

### View Overall Results

```bash
# All results
cat results/results.csv

# Filter by model
grep "YOLOV8" results/results.csv

# Filter by fold
grep "fold_1" results/results.csv

# Best mAP50 across all models
sort -t',' -k5 -nr results/results.csv | head -10
```

### View Per-Image Counting

```bash
cat results/counting.csv
```

### View Predictions

```bash
# List prediction images
ls results/prediction/YOLOV8/fold_1/

# Open with image viewer
xdg-open results/prediction/YOLOV8/fold_1/image_001.jpg
```

## Troubleshooting

### Issue: "Tiled datasets not found"

**Solution:**
```bash
./run_kfold_tiling.sh
```

### Issue: "detectores environment not found"

**Solution:**
```bash
# Create from environment file if available
conda env create -f environment.yml

# Or install manually
conda create -n detectores python=3.8
conda activate detectores
pip install torch torchvision ultralytics pycocotools
```

### Issue: "CUDA out of memory"

**Solution:**
- Reduce batch size in training scripts
- Use smaller model variant
- Train one model at a time

### Issue: Training is very slow

**Tips:**
- Use `quick_test.sh` for development
- Train on GPU (check with `nvidia-smi`)
- Use `export MODELS_TO_RUN="YOLOV8"` to train one model at a time

## Workflow Examples

### Development Workflow

```bash
# 1. Test with one fold/model first
./quick_test.sh YOLOV8 1

# 2. If successful, test all folds with that model
export MODELS_TO_RUN="YOLOV8"
./train_tiled.sh

# 3. If all good, train all models
./train_tiled.sh
```

### Production Workflow

```bash
# 1. Ensure tiled datasets are ready
ls dataset/tiles/grid/fold_*/train/*.jpg

# 2. Run full training
./train_tiled.sh

# 3. Analyze results
python3 analyze_results.py  # (if available)
```

### Quick Experimentation

```bash
# Test different models quickly
./quick_test.sh YOLOV8 1
./quick_test.sh Faster 1
./quick_test.sh YOLOV5_TPH 1

# Compare results
grep "fold_1" results/results.csv
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_TILED_DATASET` | `true` | Use tiled datasets (vs original) |
| `MODELS_TO_RUN` | `Faster,YOLOV5_TPH,YOLOV8` | Comma-separated model list |

### Main Script Options (src/main.py)

Edit these directly in `src/main.py` if needed:

```python
APENAS_TESTE = False      # True: only test, False: train + test
GeraRult = True          # Generate results CSV
save_imgs = True         # Save prediction images
GeraResultByClass = False # Generate per-class results
CONTINUE = False         # Continue from existing checkpoints
```

## Performance Tips

### Speed Up Training

1. **Use GPU**: Ensure CUDA is available
2. **Train one model at a time**: `export MODELS_TO_RUN="YOLOV8"`
3. **Use quick test during development**: `./quick_test.sh`
4. **Increase batch size** (if memory allows)

### Save Disk Space

1. **Don't save prediction images**: Set `save_imgs = False` in `src/main.py`
2. **Clean old checkpoints**: `rm -rf src/model_checkpoints/fold_*/`
3. **Train one fold at a time** using `quick_test.sh`

## Next Steps

- 📊 **Analyze results**: See which model performs best
- 🔧 **Tune hyperparameters**: Modify training scripts
- 🚀 **Deploy best model**: Use best.pt/best.pth for inference
- 📈 **Visualize metrics**: Plot mAP curves, precision-recall

## Getting Help

1. Check error messages in terminal output
2. Review `TILED_DATASET_USAGE.md` for detailed setup
3. Run integration test: `python3 test_tiled_integration.py`
4. Check dataset structure: `ls -R dataset/tiles/grid/fold_1/`

---

**Happy Training! 🎉**
