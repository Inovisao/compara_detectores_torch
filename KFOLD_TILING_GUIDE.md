# K-Fold Grid Tiling Pipeline Guide

This guide explains how to use the k-fold grid tiling pipeline to generate tiled datasets for training object detection models with large images.

## Overview

The pipeline processes 6 folds × 3 splits (train/val/test) using grid-based tiling:
- **Grid**: 6 rows × 7 columns
- **Source images**: `./dataset/all/train/`
- **Fold definitions**: `./dataset/all/filesJSON/fold_K_SPLIT.json`
- **Output**: `./dataset/tiles/grid/fold_K/SPLIT/`

### Key Features

1. **Grid-based tiling**: Each image is divided into a 6×7 grid
2. **Empty tile filtering**:
   - **Train/Val**: Tiles without annotations are discarded (saves disk space and training time)
   - **Test**: All tiles are kept (ensures complete coverage for evaluation)
3. **Metadata tracking**: Each split includes metadata.json with processing parameters
4. **Summary reports**: Detailed statistics (image count, annotation counts per category, etc.)
5. **Validation**: Automatic validation ensures all tiles and annotations are properly generated

## Prerequisites

### 1. Install Conda

If you don't have conda installed, install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 2. Create Conda Environment

```bash
conda env create -f create_dataset/environment.yml
```

This creates an environment named `create-dataset` with Python 3.9 and required dependencies (Pillow, numpy).

### 3. Verify Data Structure

Ensure you have the following directory structure:

```
./dataset/all/
├── train/                    # Source images (all original images)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── filesJSON/                # Fold split definitions
    ├── fold_1_train.json
    ├── fold_1_val.json
    ├── fold_1_test.json
    ├── fold_2_train.json
    ├── ...
    └── fold_6_test.json
```

## Running the Pipeline

### Option 1: Using the Bash Wrapper (Recommended)

The easiest way to run the pipeline:

```bash
./run_kfold_tiling.sh
```

This script will:
1. Check for conda installation
2. Create the conda environment if it doesn't exist
3. Validate required directories
4. Activate the environment
5. Run the tiling pipeline
6. Deactivate the environment

### Option 2: Manual Execution

If you prefer to run manually:

```bash
# Activate conda environment
conda activate create-dataset

# Run the tiling pipeline
python3 utils/generate_kfold_tiles.py

# Deactivate when done
conda deactivate
```

## Pipeline Behavior

### Processing Order

The pipeline processes folds sequentially:
1. Fold 1: train → val → test
2. Fold 2: train → val → test
3. ...
4. Fold 6: train → val → test

For each split, it:
1. Loads the fold JSON file (e.g., `fold_1_train.json`)
2. Reads source images from `./dataset/all/train/`
3. Generates tiles using a 6×7 grid
4. Filters annotations based on min_object_coverage (0.3)
5. **For train/val**: Discards tiles with zero annotations
6. **For test**: Keeps all tiles (including empty ones)
7. Saves tiled images and COCO annotations
8. Validates output
9. Generates metadata and summary reports

### Output Structure

```
./dataset/tiles/grid/
├── fold_1/
│   ├── train/
│   │   ├── image1_tile_0_0.jpg
│   │   ├── image1_tile_100_200.jpg
│   │   ├── ...
│   │   ├── _annotations.coco.json
│   │   ├── metadata.json
│   │   ├── summary.json
│   │   └── summary.txt
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── fold_2/
│   └── ...
└── fold_6/
    └── ...
```

### Metadata Files

Each split directory contains:

#### `metadata.json`
Processing parameters and provenance:
```json
{
  "fold": 1,
  "split": "train",
  "timestamp": "2025-10-03T18:35:00",
  "git_commit": "386a3a6",
  "source_annotations": "/path/to/fold_1_train.json",
  "source_images_dir": "/path/to/dataset/all/train",
  "tiling_config": {
    "grid_mode": true,
    "grid_rows": 6,
    "grid_cols": 7,
    "min_object_coverage": 0.3,
    "keep_empty_tiles": false
  }
}
```

#### `summary.json` / `summary.txt`
Statistics about the generated dataset:
```
Summary Report - Fold 1 - TRAIN
============================================================

Generated: 2025-10-03 18:35:00

Number of images: 12345
Number of annotations: 67890
Images without annotations: 0
Number of categories: 5

Annotations per category:
  - class1: 12345
  - class2: 23456
  - class3: 11111
  - class4: 10000
  - class5: 10978
```

## Configuration

To modify tiling parameters, edit `utils/generate_kfold_tiles.py`:

```python
# Configuration section (lines 37-44)
NUM_FOLDS = 6
SPLITS = ["train", "val", "test"]
GRID_ROWS = 6          # Change grid rows
GRID_COLS = 7          # Change grid columns
MIN_OBJECT_COVERAGE = 0.3  # Minimum % of object visible to keep annotation
```

## Troubleshooting

### "conda not found"
- Install Miniconda or Anaconda
- Add conda to your PATH

### "Annotations file not found"
- Verify fold JSON files exist in `./dataset/all/filesJSON/`
- Check file naming: `fold_K_SPLIT.json` (K=1-6, SPLIT=train/val/test)

### "Source images directory not found"
- Verify images are in `./dataset/all/train/`
- Check that image paths in fold JSON files match actual filenames

### "Validation failed"
- Check disk space
- Verify create_dataset module is working correctly
- Review error messages in the output

### Memory issues
- Process one fold at a time by modifying the script
- Close other applications to free up memory

## Expected Runtime

Processing time depends on:
- Number of images per fold
- Image sizes
- Disk I/O speed
- CPU performance

Typical runtime: 10-60 minutes for all 6 folds (varies widely)

## Next Steps

After successful tiling:

1. **Review outputs**: Check a few tiles visually to ensure quality
2. **Verify statistics**: Review summary.txt files to ensure reasonable distributions
3. **Update training config**: Point your detector training scripts to the new tiled datasets
4. **Test one fold first**: Before training all folds, test with fold 1 to validate the pipeline

### Training Integration Example

Update your training data configuration to use the tiled datasets:

```yaml
# For fold 1 training
train: ./dataset/tiles/grid/fold_1/train/_annotations.coco.json
val: ./dataset/tiles/grid/fold_1/val/_annotations.coco.json
test: ./dataset/tiles/grid/fold_1/test/_annotations.coco.json

# For fold 2 training
train: ./dataset/tiles/grid/fold_2/train/_annotations.coco.json
val: ./dataset/tiles/grid/fold_2/val/_annotations.coco.json
test: ./dataset/tiles/grid/fold_2/test/_annotations.coco.json
```

## Advanced Usage

### Process a Single Fold

To process only one fold, modify `utils/generate_kfold_tiles.py`:

```python
# Change line 284
for fold in [1]:  # Process only fold 1
    for split in SPLITS:
        # ...
```

### Custom Grid Size

To use a different grid size:

```python
# Change lines 40-41
GRID_ROWS = 8  # Your desired rows
GRID_COLS = 9  # Your desired columns
```

### Add Tile Resizing

To resize tiles after grid splitting, edit `create_dataset/src/config/settings.py`:

```python
# In TilingConfig
resize_output: Optional[Tuple[int, int]] = (1024, 1024)  # Resize to 1024×1024
```

## Implementation Details

### Changes Made to create_dataset

The following modifications were made to support k-fold tiling:

1. **Added `keep_empty_tiles` flag** (`create_dataset/src/config/settings.py:13`)
   - Controls whether to save tiles without annotations
   - Default: False (discard empty tiles)

2. **Updated processor** (`create_dataset/src/services/dataset/processor.py:26-33`)
   - Added `annotations_path`, `images_dir`, `split_name` parameters
   - Support for arbitrary COCO annotation files
   - Flexible output directory naming

3. **Empty tile filtering** (`create_dataset/src/services/dataset/processor.py:102-104`)
   - Skip saving tiles with zero annotations when `keep_empty_tiles=False`

4. **Updated grid_cols default** (`create_dataset/src/config/settings.py:17`)
   - Changed from 3 to 7 columns as per requirements

### Architecture

```
run_kfold_tiling.sh
    └─> utils/generate_kfold_tiles.py
            ├─> Reads fold JSON files
            ├─> Configures TilingConfig per split
            └─> Calls DatasetProcessor.process_dataset()
                    ├─> TilingEngine.generate_tiles()
                    ├─> AnnotationManager.transform_annotations()
                    ├─> Saves tiles and annotations
                    ├─> Validates output
                    └─> Returns success/failure
```

## Support

For issues or questions:
1. Check the summary reports in the output directories
2. Review error messages in the console output
3. Verify your directory structure matches the expected format
4. Check the eng_cl.md file for additional context

## Acknowledgments

This pipeline was developed to support efficient training of object detection models on high-resolution images using grid-based tiling and k-fold cross-validation.

---

Generated: 2025-10-03
