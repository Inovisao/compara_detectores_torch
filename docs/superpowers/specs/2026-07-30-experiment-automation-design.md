# Experiment Automation Design Spec

**Date:** 2026-07-30  
**Purpose:** Automate hyperparameter sweep experiments for object detection models (paper experimentation)

## Overview

Create an automation layer that orchestrates k-fold cross-validation experiments across multiple detection models (YOLOV8, FasterRCNN, DETR, SwinDetector) with systematic hyperparameter variation. The automation script will:

- Read experiment configuration from YAML
- Generate all hyperparameter combinations
- Execute training via wrapper scripts (zero modifications to legacy code)
- Track progress for resume capability
- Output flat results table for analysis

**Key constraint:** No modifications to existing code in `src/`, `utils/`, or `dataset/`.

## Architecture

### Folder Structure

```
compara_detectores_torch/
├── automation/                    # NEW - automation scripts only
│   ├── config/
│   │   └── experiment.yaml        # Experiment configuration
│   ├── wrappers/
│   │   ├── yolov8_wrapper.py      # YOLOV8 training wrapper
│   │   ├── fasterrcnn_wrapper.py  # FasterRCNN training wrapper
│   │   ├── detr_wrapper.py        # DETR training wrapper
│   │   └── swin_detector_wrapper.py  # Swin standalone detector
│   ├── run_experiments.py         # Main orchestrator
│   └── state/
│       ├── progress.json          # Tracks completed/failed experiments
│       └── experiment_plan.json   # Generated experiment list (dry-run)
├── src/                           # UNCHANGED - legacy code
├── utils/                         # UNCHANGED
├── dataset/                       # UNCHANGED
└── results/
    └── experiments.csv            # Flat results table
```

### Components

1. **`run_experiments.py`** - Main orchestrator
   - Reads YAML config
   - Generates experiment grid (Cartesian product of all hyperparameters)
   - Iterates through experiments sequentially
   - Checks `progress.json` for resume
   - Calls appropriate wrapper
   - Collects metrics and writes to `experiments.csv`
   - Updates `progress.json` after each experiment

2. **`wrappers/`** - Model-specific training wrappers
   - Each wrapper accepts experiment config + fold + seed
   - Calls existing utilities (GeraLabels, geradataset, etc.) without modification
   - Invokes training with experiment hyperparameters
   - Returns best model path + metrics

3. **`experiment.yaml`** - Configuration file
   - Defines models, hyperparameter grids, folds, seeds, augmentations
   - Supports model-specific augmentation options

4. **`progress.json`** - State tracking
   - Records completed experiments (for resume)
   - Records failed experiments (for retry)
   - Stores best model paths and metrics

5. **`experiments.csv`** - Results table
   - Flat table with all hyperparameters and metrics as columns
   - One row per experiment (model × hyperparams × fold × seed)

## Configuration (YAML)

```yaml
experiment_name: "javalis_detection_v1"
seed: 42

dataset:
  coco_json: "../anotacoes_andre_08.06.2026/Anotacao_Javalis_Coco_Json/annotations.coco.json"
  images_dir: "../anotacoes_andre_08.06.2026/Anotacao_Javalis_Coco_Json/images"
  output_dir: "../dataset/all"

folds:
  n_folds: 5
  val_percentage: 0.3

models:
  YOLOV8:
    architectures: [yolov8s, yolov8m]
    learning_rates: [0.001, 0.0001, 0.00001]
    optimizers: [AdamW, SGD]
    batch_sizes: [32, 64]
    weight_decays: [0.0005]
    schedulers: [cosine]
    epochs: [100]
    patience: [50]
    augmentations:
      mosaic: [0.0, 1.0]
      flipud: [0.0, 0.5]
      degrees: [0.0, 10.0]
  
  Faster:
    architectures: 
      - resnet50
      - resnet101
      - swin_tiny      # Swin as FasterRCNN backbone
      - swin_small     # Swin as FasterRCNN backbone
    learning_rates: [0.001, 0.0001]
    optimizers: [SGD, Adam]
    batch_sizes: [8, 16]
    weight_decays: [0.0005, 0.001]
    schedulers: [step]
    epochs: [30]
    patience: [5]
    augmentations:
      horizontal_flip: [true, false]
  
  Detr:
    architectures: [detr_resnet50, detr_resnet101]
    learning_rates: [0.0001, 0.00001]
    optimizers: [AdamW]
    batch_sizes: [4, 8]
    weight_decays: [0.0001]
    schedulers: [multi_step]
    epochs: [50]
    patience: [10]
    augmentations:
      mosaic: [0.0]
  
  SwinDetector:  # Standalone Swin detector
    architectures: [swin_tiny, swin_small]
    learning_rates: [0.0001, 0.00001]
    optimizers: [AdamW]
    batch_sizes: [4, 8]
    weight_decays: [0.0001]
    schedulers: [cosine]
    epochs: [50]
    patience: [10]
    augmentations:
      horizontal_flip: [true, false]

execution:
  dry_run: false
  continue_on_error: true
  retry_failed: false
  log_level: INFO
```

## Resume & Dry-Run

### Progress Tracking (`progress.json`)

```json
{
  "experiment_name": "javalis_detection_v1",
  "total_experiments": 4320,
  "completed": [
    {
      "id": "YOLOV8_yolov8s_lr0.001_AdamW_bs32_cosine_mosaic1.0_flipud0.5_deg10.0_fold1_seed42",
      "model": "YOLOV8",
      "fold": 1,
      "seed": 42,
      "status": "completed",
      "best_model_path": "model_checkpoints/fold_1/YOLOV8/train/weights/best.pt",
      "metrics": {
        "mAP": 0.85,
        "mAP50": 0.92,
        "mAP75": 0.78
      },
      "timestamp": "2026-07-30T10:23:45"
    }
  ],
  "failed": [
    {
      "id": "Faster_resnet50_lr0.001_SGD_bs8_step_fold3_seed42",
      "model": "Faster",
      "fold": 3,
      "seed": 42,
      "status": "failed",
      "error": "CUDA OOM",
      "timestamp": "2026-07-30T11:05:12"
    }
  ]
}
```

### Resume Logic

- On start, load `progress.json` if exists
- Skip experiments already in `completed`
- If `retry_failed: true`, retry experiments in `failed`
- After each experiment, atomically update `progress.json`

### Dry-Run Mode (`dry_run: true`)

- Generate full experiment list
- Print summary: total experiments, per-model breakdown, estimated time
- Validate config (paths, YAML syntax, model names)
- Write experiment plan to `state/experiment_plan.json`
- Do NOT execute any training

## Results Table (`experiments.csv`)

Flat table with one row per experiment:

| Column | Type | Example | Source |
|--------|------|---------|--------|
| experiment_id | string | `YOLOV8_yolov8s_lr0.001_...` | generated |
| model | string | `YOLOV8` | config |
| architecture | string | `yolov8s` | config |
| architecture_type | string | `resnet` / `swin_backbone` / `swin_standalone` | wrapper |
| learning_rate | float | `0.001` | config |
| optimizer | string | `AdamW` | config |
| batch_size | int | `32` | config |
| weight_decay | float | `0.0005` | config |
| scheduler | string | `cosine` | config |
| epochs | int | `100` | config |
| patience | int | `50` | config |
| augmentation_mosaic | float | `1.0` | config |
| augmentation_flipud | float | `0.5` | config |
| augmentation_degrees | float | `10.0` | config |
| augmentation_horizontal_flip | bool | `true` | config |
| fold | int | `1` | config |
| seed | int | `42` | config |
| mAP | float | `0.85` | existing |
| mAP50 | float | `0.92` | existing |
| mAP75 | float | `0.78` | existing |
| mAP50_95 | float | `0.71` | new (COCO-style) |
| MAE | float | `2.3` | existing |
| RMSE | float | `3.1` | existing |
| r | float | `0.94` | existing |
| precision | float | `0.88` | existing |
| recall | float | `0.82` | existing |
| f1_score | float | `0.85` | existing |
| train_loss_final | float | `0.45` | new |
| training_time_s | float | `1234.5` | new |
| num_parameters | int | `11200000` | new |
| best_model_path | string | `model_checkpoints/...` | existing |
| status | string | `completed` | orchestrator |
| timestamp | string | `2026-07-30T10:23:45` | orchestrator |

**Notes:**
- Model-specific augmentation columns get `N/A` for models that don't use them
- CSV is append-only (each completed experiment adds one row)
- Can be loaded directly into Pandas/R for analysis

## Wrapper Implementation

### YOLOV8 Wrapper

**Strategy:** Bypass `config.py`, use ultralytics API directly.

```python
# wrappers/yolov8_wrapper.py
from ultralytics import YOLO
from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from Detectors.YOLOV8.TrocaSettings import Settings

def train_yolov8(experiment_config, fold, seed):
    Settings()  # existing utility
    CriarLabelsYOLOV8(fold)  # existing utility
    
    model = YOLO(f"{experiment_config['architecture']}.pt")
    model.train(
        data='../dataset/all/data.yaml',
        lr0=experiment_config['learning_rate'],
        optimizer=experiment_config['optimizer'],
        batch=experiment_config['batch_size'],
        epochs=experiment_config['epochs'],
        patience=experiment_config['patience'],
        weight_decay=experiment_config['weight_decay'],
        cos_lr=(experiment_config['scheduler'] == 'cosine'),
        mosaic=experiment_config.get('augmentation_mosaic', 0.0),
        flipud=experiment_config.get('augmentation_flipud', 0.0),
        degrees=experiment_config.get('augmentation_degrees', 0.0),
        seed=seed,
        project='YOLOV8',
        exist_ok=True,
        plots=True
    )
    
    best_model_path = 'YOLOV8/train/weights/best.pt'
    return best_model_path
```

### FasterRCNN Wrapper

**Strategy:** Call training functions directly, override config values.

```python
# wrappers/fasterrcnn_wrapper.py
import sys
import os
from pathlib import Path
from Detectors.FasterRCNN.geradataset import geredata

def train_fasterrcnn(experiment_config, fold, seed):
    geredata(fold)  # existing utility
    
    # Determine architecture type
    arch = experiment_config['architecture']
    if arch.startswith('swin_'):
        architecture_type = 'swin_backbone'
    else:
        architecture_type = 'resnet'
    
    # Import and run training with overridden config
    # Implementation: either mock config or create temp config
    # ... (detailed implementation in plan)
    
    best_model_path = 'Faster/best.pth'
    return best_model_path, architecture_type
```

### DETR Wrapper

**Strategy:** Use existing CLI args support.

```python
# wrappers/detr_wrapper.py
import subprocess
import sys
from Detectors.Detr.GeraDobras import convert_coco_to_voc

def train_detr(experiment_config, fold, seed):
    convert_coco_to_voc(fold)  # existing utility
    
    cmd = [
        sys.executable, 'Detectors/Detr/train_detector.py',
        '--epochs', str(experiment_config['epochs']),
        '--batch', str(experiment_config['batch_size']),
        '--learning-rate', str(experiment_config['learning_rate']),
        '--lr-backbone', str(experiment_config['learning_rate'] * 0.1),
        '--weight-decay', str(experiment_config['weight_decay']),
        '--model', experiment_config['architecture'],
        '--seed', str(seed),
        '--device', 'cuda'
    ]
    
    subprocess.run(cmd, check=True)
    
    best_model_path = 'Detr/training/best_model.pth'
    return best_model_path
```

### SwinDetector Wrapper (Standalone)

**Strategy:** New detector using Swin backbone + custom detection head.

```python
# wrappers/swin_detector_wrapper.py
import timm
import torch
from torch.utils.data import DataLoader

def train_swin_detector(experiment_config, fold, seed):
    # Load Swin backbone from timm
    backbone = timm.create_model(
        experiment_config['architecture'],  # 'swin_tiny' or 'swin_small'
        pretrained=True,
        features_only=True
    )
    
    # Custom detection head (to be implemented)
    # ... (detailed implementation in plan)
    
    best_model_path = 'SwinDetector/best_model.pth'
    return best_model_path
```

## Swin Transformer Integration

### Option 1: Swin as FasterRCNN Backbone

- Replace ResNet-50/101 FPN backbone with Swin
- Keep FasterRCNN detection head (RPN + ROI heads)
- Use `timm` library to load Swin
- Architecture names: `swin_tiny`, `swin_small`

### Option 2: SwinDetector (Standalone)

- Swin as feature extractor
- Custom detection head (lightweight FC layers or detector head)
- Similar to DETR approach but with Swin backbone
- Architecture names: `swin_tiny`, `swin_small`

### Dependencies

```bash
pip install timm  # for Swin Transformer models
```

### Results Table

Add `architecture_type` column to distinguish:
- `resnet` - standard ResNet backbone
- `swin_backbone` - Swin as FasterRCNN backbone
- `swin_standalone` - SwinDetector standalone

## Execution Flow

1. **Load config:** Read `experiment.yaml`
2. **Generate experiments:** Cartesian product of all hyperparameters × folds × seeds
3. **Load progress:** Read `progress.json` (if exists)
4. **Iterate:** For each experiment:
   - Check if already completed (skip if yes)
   - Check if failed and `retry_failed: false` (skip if yes)
   - Call appropriate wrapper
   - Collect metrics from `ResultsDetections.py`
   - Write row to `experiments.csv`
   - Update `progress.json`
5. **Finish:** Print summary

## Error Handling

- **`continue_on_error: true`** - Log failure, mark as failed in `progress.json`, continue to next experiment
- **`continue_on_error: false`** - Stop immediately on first failure
- All errors logged with timestamp and experiment ID

## Metrics Collection

After training, call existing `ResultsDetections.py` functions:

```python
from ResultsDetections import create_csv
from ResultsDetectionsbyclass import generate_results

# After training completes
metrics = create_csv(
    root=ROOT_DATA_DIR,
    fold=fold,
    selected_model=model_name,
    model_path=best_model_path,
    save_imgs=False
)
# metrics = (mAP, mAP50, mAP75, MAE, RMSE, precision, recall, fscore, r)
```

Additional metrics to capture:
- `mAP50_95` - COCO-style average across IoU thresholds
- `train_loss_final` - Final training loss
- `training_time_s` - Wall-clock time for training
- `num_parameters` - Model parameter count

## Usage

```bash
# Dry run (validate config, show experiment plan)
cd automation
python run_experiments.py --dry-run

# Run experiments
python run_experiments.py

# Resume after interruption (automatic if progress.json exists)
python run_experiments.py

# Retry failed experiments
python run_experiments.py --retry-failed
```

## Future Extensions

- Parallel execution (multiple GPUs)
- Early stopping based on validation metrics
- Hyperparameter optimization (Bayesian optimization)
- Visualization dashboard
- Export results to LaTeX tables for paper
