# Tiled Dataset Usage Guide

This repository now supports training detectors on both high-resolution and tiled datasets through the k-fold tiling pipeline.

## Overview

The tiled dataset integration allows you to train detection models (YOLOV8, Faster R-CNN, YOLOV5_TPH) on grid-tiled images, making it possible to train on high-resolution images using a single GPU.

## Quick Start

### 1. Generate Tiled Datasets (if not already done)

```bash
# Run the k-fold tiling pipeline
./run_kfold_tiling.sh
```

This will create tiled datasets in `./dataset/tiles/grid/fold_K/{train,val,test}/` for each of the 6 folds.

### 2. Train with Tiled Datasets

```bash
cd src

# Use tiled datasets (default)
export USE_TILED_DATASET=true
python3 main.py

# Or use original high-res datasets
export USE_TILED_DATASET=false
python3 main.py
```

## Configuration

### Environment Variable

- **`USE_TILED_DATASET`** (default: `true`)
  - `true`: Use tiled datasets from `dataset/tiles/grid/fold_K/`
  - `false`: Use original high-res datasets from `dataset/all/`

### Model Selection

You can specify which models to train using the `MODELS_TO_RUN` environment variable:

```bash
export MODELS_TO_RUN="YOLOV8,Faster,YOLOV5_TPH"
python3 main.py
```

## Dataset Structure

### Tiled Dataset Structure
```
dataset/tiles/grid/
├── fold_1/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   ├── metadata.json
│   │   ├── summary.json
│   │   └── *.jpg (tiled images)
│   ├── val/
│   │   └── ... (same structure)
│   └── test/
│       └── ... (same structure)
├── fold_2/
│   └── ...
└── fold_6/
    └── ...
```

### Original Dataset Structure
```
dataset/all/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg (high-res images)
└── filesJSON/
    ├── fold_1_train.json
    ├── fold_1_val.json
    ├── fold_1_test.json
    └── ... (for each fold)
```

## Testing the Integration

Run the integration test to verify everything is working:

```bash
python3 test_tiled_integration.py
```

This will test all three detectors (YOLOV8, Faster R-CNN, YOLOV5_TPH) with fold_1 tiled data.

## Tiling Configuration

The k-fold tiling pipeline uses the following configuration:

- **Grid**: 6 rows × 7 columns
- **Min object coverage**: 0.3
- **Empty tiles**: Kept only for test split (discarded for train/val)
- **Number of folds**: 6

You can modify these settings in `utils/generate_kfold_tiles.py`.

## Troubleshooting

### Issue: "Tiled dataset not found"

**Solution**: Run `./run_kfold_tiling.sh` to generate the tiled datasets first.

### Issue: Different image counts between detectors

This is expected:
- **YOLOV8**: Discards images without annotations in all splits
- **Faster R-CNN & YOLOV5_TPH**: Keep all images including empty ones (especially in test split)

### Issue: Training fails with "FileNotFoundError"

**Solution**: Make sure you're running from the `src/` directory:
```bash
cd src
python3 main.py
```

## Implementation Details

The integration modifies the following files:

- `src/main.py`: Added `USE_TILED_DATASET` flag and per-fold ROOT_DATA_DIR logic
- `src/Detectors/YOLOV8/GeraLabels.py`: Support for tiled dataset structure
- `src/Detectors/FasterRCNN/geradataset.py`: Support for tiled dataset structure
- `src/Detectors/YOLOV5_TPH/GeraLabels.py`: Support for tiled dataset structure

All detector run scripts now accept a `root_data_dir` parameter for flexible dataset paths.

## Next Steps

After successful training, you can:

1. Review results in `../results/results.csv` and `../results/counting.csv`
2. Check model checkpoints in `model_checkpoints/fold_K/MODEL_NAME/`
3. View prediction images in `../results/prediction/`

## Notes

- The tiled dataset mode is now the **default** (`USE_TILED_DATASET=true`)
- All 6 folds are automatically detected from the `dataset/tiles/grid/` directory
- Backward compatibility with the original dataset structure is maintained
