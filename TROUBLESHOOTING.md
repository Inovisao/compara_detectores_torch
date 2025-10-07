# Troubleshooting Guide

This guide documents common issues and their solutions when training detection models with tiled datasets.

---

## Issue 1: NumPy 2.x Incompatibility ✅ FIXED

### Error Message
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.

Traceback (most recent call last):
  File "src/ResultsDetections.py", line 6, in <module>
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
  ...
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

### Root Cause
- NumPy 2.0.2 was installed
- torchmetrics and torchvision were compiled with NumPy 1.x
- Binary incompatibility between NumPy versions

### Solution
```bash
conda activate detectores
pip install "numpy<2.0"
```

This downgrades NumPy to 1.26.4, which is compatible with all compiled modules.

### Verification
```bash
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
# Should output: NumPy version: 1.26.4
```

**Status**: ✅ **PERMANENTLY FIXED**

---

## Issue 2: Hardcoded Dataset Paths ✅ FIXED

### Error Message
```
FileNotFoundError: [Errno 2] No such file or directory:
'../dataset/all/Faster/train/_annotations.coco.json'
```

### Root Cause
- Detector config files had hardcoded paths: `ROOT_DATA_DIR = os.path.join('..', 'dataset','all')`
- When using tiled datasets, the correct path is: `../dataset/tiles/grid/fold_K/`
- The hardcoded paths prevented dynamic path resolution

### Solution Applied (Code Changes)

The following files were modified to support environment variables:

**1. Faster R-CNN**
- `src/Detectors/FasterRCNN/config.py`:
  ```python
  ROOT_DATA_DIR = os.getenv('FASTER_ROOT_DATA_DIR', os.path.join('..','dataset','all'))
  ```
- `src/Detectors/FasterRCNN/runFaster.py`:
  ```python
  os.environ['FASTER_ROOT_DATA_DIR'] = ROOT_DATA_DIR
  ```

**2. YOLOV8**
- `src/Detectors/YOLOV8/config.py`:
  ```python
  data_yaml = os.getenv('YOLOV8_DATA_YAML', '../dataset/all/data.yaml')
  ```
- `src/Detectors/YOLOV8/RunYOLOV8.py`:
  ```python
  os.environ['YOLOV8_DATA_YAML'] = os.path.join(ROOT_DATA_DIR, 'data.yaml')
  ```

**3. YOLOV5_TPH**
- `src/Detectors/YOLOV5_TPH/config.py`:
  ```python
  DATA_YAML_STR = os.getenv("TPH_DATA_YAML")
  if DATA_YAML_STR:
      DATA_YAML = Path(DATA_YAML_STR)
  else:
      DATA_YAML = PROJECT_ROOT / "dataset" / "all" / "data_yolov5_tph.yaml"
  ```
- `src/Detectors/YOLOV5_TPH/RunYOLOV5TPH.py`:
  ```python
  os.environ['TPH_DATA_YAML'] = os.path.join(ROOT_DATA_DIR, 'data_yolov5_tph.yaml')
  ```

### How It Works Now

1. `main.py` sets `current_root` based on `USE_TILED_DATASET`:
   - If `USE_TILED_DATASET=true`: `current_root = ../dataset/tiles/grid/fold_K/`
   - If `USE_TILED_DATASET=false`: `current_root = ../dataset/all/`

2. Each `Run*.py` file receives `ROOT_DATA_DIR` as parameter

3. Each `Run*.py` sets the appropriate environment variable before training

4. Each `config.py` reads the environment variable (or uses default)

**Status**: ✅ **PERMANENTLY FIXED**

---

## Issue 3: "Tiled datasets not found"

### Error Message
```
Error: Tiled datasets not found at ./dataset/tiles/grid
```

### Root Cause
- Tiled datasets haven't been generated yet
- The k-fold tiling pipeline hasn't been run

### Solution
```bash
./run_kfold_tiling.sh
```

This will:
1. Verify source images in `./dataset/all/train/`
2. Read fold annotations from `./dataset/all/filesJSON/`
3. Generate tiled datasets in `./dataset/tiles/grid/fold_1..6/`
4. Create train/val/test splits for each fold
5. Generate metadata and summary reports

### Verification
```bash
ls dataset/tiles/grid/fold_1/train/*.jpg | wc -l
# Should show ~3400 images
```

---

## Issue 4: CUDA Out of Memory

### Error Message
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

### Solutions

**Option 1: Reduce Batch Size**

For Faster R-CNN (`src/Detectors/FasterRCNN/config.py`):
```python
BATCH_SIZE = 2  # Reduce from 4
```

For YOLOV8:
```bash
export YOLOV8_BATCH=16  # Reduce from 64
./train_tiled.sh
```

For YOLOV5_TPH:
```bash
export TPH_BATCH=4  # Reduce from 8
./train_tiled.sh
```

**Option 2: Train One Model at a Time**
```bash
export MODELS_TO_RUN="YOLOV8"
./train_tiled.sh
```

**Option 3: Use Quick Test for Development**
```bash
./quick_test.sh YOLOV8 1  # Only one fold
```

---

## Issue 5: Training is Very Slow

### Potential Causes & Solutions

**1. Check GPU Usage**
```bash
nvidia-smi
# Should show GPU utilization > 80%
```

If GPU is not being used:
- Check CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**2. Reduce Dataset Size for Testing**
```bash
./quick_test.sh YOLOV8 1  # Test with one fold
```

**3. Check Batch Size**
Larger batch sizes = faster training (if GPU memory allows)

**4. Use Faster Model Variants**
- YOLOV8: Use `yolov8n.pt` (nano) instead of `yolov8s.pt` (small)

---

## Issue 6: Permission Denied on Scripts

### Error Message
```
bash: ./train_tiled.sh: Permission denied
```

### Solution
```bash
chmod +x train_tiled.sh quick_test.sh run_kfold_tiling.sh
```

---

## Issue 7: Conda Environment Not Found

### Error Message
```
CondaError: environment 'detectores' not found
```

### Solution
Check available environments:
```bash
conda env list
```

If `detectores` doesn't exist, create it:
```bash
conda create --name detectores python=3.9
conda activate detectores
# Install dependencies (see INSTALLATION_STATUS.md)
```

---

## Issue 8: Missing tph-yolov5 Repository

### Error Message
```
FileNotFoundError: Repository tph-yolov5 not found
```

### Solution
```bash
cd src/Detectors/YOLOV5_TPH
git clone https://github.com/cv516Buaa/tph-yolov5.git
```

---

## Quick Diagnostic Commands

### Check Installation Status
```bash
# NumPy version (should be <2.0)
python -c "import numpy; print(numpy.__version__)"

# Check if tiled datasets exist
ls dataset/tiles/grid/fold_1/train/*.jpg | head -5

# Check conda environment
conda env list | grep detectores

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Test Integration
```bash
# Quick integration test (all 3 detectors)
python3 test_tiled_integration.py

# Test single model
./quick_test.sh YOLOV8 1
```

### View Training Logs
```bash
# Recent training attempts
ls -lt src/model_checkpoints/fold_*/*/

# View results
cat results/results.csv
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check error messages carefully** - they often indicate the exact problem
2. **Verify environment**: `conda activate detectores`
3. **Run integration test**: `python3 test_tiled_integration.py`
4. **Check git log**: Recent commits may have fixes
5. **Review documentation**:
   - `INSTALLATION_STATUS.md` - Installation details
   - `QUICK_START.md` - Usage guide
   - `TILED_DATASET_USAGE.md` - Dataset configuration

---

## Summary of Fixes Applied

| Issue | Status | Solution |
|-------|--------|----------|
| NumPy 2.x incompatibility | ✅ Fixed | Downgraded to NumPy 1.26.4 |
| Hardcoded dataset paths | ✅ Fixed | Added environment variable support |
| Missing tiled datasets | ✅ Doc | Run `./run_kfold_tiling.sh` |
| Missing tph-yolov5 | ✅ Fixed | Cloned repository |
| Missing tensorboard | ✅ Fixed | Installed via pip |

**All critical issues resolved!** The system is ready for training.
