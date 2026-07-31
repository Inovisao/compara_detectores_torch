# Config-Driven Refactor Design

**Date:** 2026-07-31
**Status:** Draft

## Motivation

The current codebase has two conflicting orchestration systems (legacy `src/main.py` and newer `automation/`), duplicated code across detector directories, stub implementations (Swin, RT-DETR), and heavy dependencies (MMDetection). Adding a new architecture requires touching 5+ files across both systems. The goal is a single, config-driven system where a new architecture is one file.

## Design Decisions (from brainstorming)

| Decision | Choice |
|----------|--------|
| MMDetection | **Drop**. Heavy, inference-only, outdated version. |
| Config system | **YAML + CLI overrides** (Typer). Reproducibility + quick iteration. |
| Adding architectures | **Single file per model family** implementing a `Detector` protocol. Registration is one line. |
| Train/eval coupling | **Decoupled.** `python cli.py train` and `python cli.py eval` are separate steps. `sweep` chains them. |
| Dependency strategy | **Pragmatic hybrid.** Keep Ultralytics (YOLO). Use torchvision for others. Optional `timm` for Swin. |
| Folds | **Config parameter.** `n_folds: 5, val_ratio: 0.2`. Generated at runtime from COCO JSON. No external scripts. |
| Config documentation | **Comment-generated defaults.** `python cli.py config --dump` produces a YAML with all hparams commented out, types documented, values set to factory defaults. |

## Project Structure

```
compara_detectores_torch/
├── pyproject.toml              # Single source of deps (replaces Bibliotecas.yml + requirements_automation.txt)
├── configs/
│   ├── default.yaml            # Baseline defaults (dataset, folds, all detector defaults commented)
│   └── sweeps/
│       └── baseline.yaml       # Grid search definitions
├── dataset/                    # Kept as-is (COCO JSON + images)
│   └── all/
├── detectors/                  # One file per model family
│   ├── __init__.py             # DETECTOR_REGISTRY dict
│   ├── base.py                 # Detector ABC
│   ├── yolov8.py
│   ├── faster_rcnn.py
│   └── detr.py
├── engine/
│   ├── trainer.py              # Generic training loop (torchvision-compatible)
│   ├── evaluator.py            # COCO eval + custom metrics
│   └── sweeper.py              # Grid generation → train → eval
├── data/
│   ├── dataset.py              # COCODataset (reads COCO JSON)
│   └── transforms.py           # Albumentations → torch tensor
├── cli.py                      # Typer CLI: train / eval / sweep / config
├── utils/
│   ├── logging.py
│   ├── folds.py                # Runtime fold splitting
│   └── results.py              # CSV collation + aggregation
└── results/                    # Output directory
    └── {experiment_name}/
        ├── fold_{N}/{architecture}/
        │   ├── best.pt
        │   ├── logs.csv
        │   └── metrics.json
        ├── summary.csv
        └── summary_aggregated.csv
```

## Detector Interface

```python
class Detector(ABC):
    @abstractmethod
    def train(self, train_loader, val_loader, config: dict, output_dir: Path) -> Path:
        """Train. Return path to best checkpoint.

        config dict contains:
            architecture: str    # e.g. "yolov8s"
            num_classes: int
            + all keys from the detector's default_hparams() overridden by YAML/CLI
        """

    @abstractmethod
    def predict(self, images: list) -> list:
        """Return list of [{boxes, scores, labels}]."""

    @abstractmethod
    def load(self, path: Path):
        """Load weights from checkpoint."""

    @classmethod
    def architectures(cls) -> list[str]:
        """Supported variants (e.g. ['yolov8n', 'yolov8s'])."""

    @classmethod
    def default_hparams(cls) -> dict:
        """Per-family hyperparameter defaults."""
```

Key design points:
- `train()` is opaque. YOLO delegates to `model.train()` internally; torchvision models use `engine/trainer.py`; DETR uses its own matcher/criterion loop.
- `architectures()` and `default_hparams()` are class methods — the CLI discovers them without instantiation.
- Registration is one line in `detectors/__init__.py`. Dropping a new `.py` file + one registry entry adds full support for `train`, `eval`, `sweep`, config dump, and CSV output.

## Config System

Three-layer merge: **CLI > YAML file > `default.yaml` > `default_hparams()`**.

### default.yaml (shipped, provides schema + docs)

```yaml
dataset:
  coco_json: dataset/all/train/_annotations.coco.json
  images_dir: dataset/all/train

folds:
  n_folds: 5
  val_ratio: 0.2
  seed: 42

output_dir: results
seed: 42
device: cuda

detectors:
  yolov8:
    architectures: [yolov8s]
    hparams:
      lr: 0.001
      epochs: 100
      batch_size: 16
      optimizer: AdamW
      weight_decay: 0.0005
      momentum: 0.937
      scheduler: cosine
      warmup_epochs: 3
      patience: 10
      imgsz: 640
      workers: 8
    # All hparams above are the factory defaults. Uncomment and edit to override.

  faster_rcnn:
    architectures: [resnet50]
    hparams: {lr: 0.0001, epochs: 30, batch_size: 8, optimizer: SGD, ...}

  detr:
    architectures: [detr_resnet50]
    hparams: {lr: 0.0001, epochs: 150, batch_size: 8, optimizer: AdamW, ...}
```

### Sweep config (override defaults, lists = grid)

```yaml
experiment: baseline_v2
detectors:
  yolov8:
    architectures: [yolov8n, yolov8s, yolov8m]
    hparams: {lr: [0.0001, 0.001, 0.01], batch_size: 32}
  faster_rcnn:
    architectures: [resnet50, resnet101]
    hparams: {lr: [0.0001, 0.001], batch_size: [8, 16]}
  detr: ~  # skip
```

### CLI

```bash
python cli.py train --detector yolov8 --arch yolov8n --lr 0.0005 --epochs 50
python cli.py eval --detector yolov8 --weights results/my_run/fold_0/yolov8n/best.pt
python cli.py sweep --config configs/sweeps/baseline.yaml
python cli.py config --dump > my_config.yaml  # generate commented config
python cli.py aggregate results/baseline_v2/   # mean ± std across folds
```

## Train & Eval Pipeline

```
config.yaml ──► sweeper ──► grid ──► for each experiment:
                                          ├── train() → best.pt + logs.csv
                                          └── eval()  → metrics.json + CSV row
```

**Train phase:**
1. Load config → instantiate `Detector` subclass
2. COCO JSON → split by fold → `DataLoader` pairs
3. `detector.train(train_loader, val_loader, hparams, output_dir)` → `best.pt`

**Eval phase:**
1. `detector.load(best.pt)`
2. `detector.predict()` on test set
3. COCO eval (mAP, mAP50, mAP75, mAP per class)
4. Custom counting metrics (MAE, RMSE, precision, recall, F1, Pearson r)
5. Write `metrics.json`, append CSV row

**Output per experiment:**
```
results/baseline_v2/
├── fold_0/{architecture}/best.pt, logs.csv, metrics.json
├── fold_1/...
├── summary.csv                  # all individual runs
└── summary_aggregated.csv       # mean ± std across folds
```

## Adding a New Architecture

Example: adding RetinaNet.

**One file** (`detectors/retinanet.py`, ~60 lines):
```python
class RetinaNetDetector(Detector):
    @classmethod
    def architectures(cls): return ["retinanet_resnet50", "retinanet_resnet101"]

    @classmethod
    def default_hparams(cls):
        return {"lr": 0.0001, "epochs": 30, "batch_size": 8, ...}

    def _build_model(self, arch, num_classes): ...

    def train(self, train_loader, val_loader, config, output_dir):
        model = self._build_model(...)
        return train_torchvision_model(model, train_loader, val_loader, config, output_dir)

    def predict(self, images): ...
    def load(self, path): ...
```

**One registry line** in `detectors/__init__.py`:
```python
DETECTOR_REGISTRY = {"retinanet": RetinaNetDetector, ...}
```

Everything else (CLI, config dump, sweep, metrics, CSV) works without changes.

## Dependencies

**Core (required):**
- `torch >= 2.1`
- `torchvision >= 0.16`
- `ultralytics >= 8.2` (YOLOv8)
- `PyYAML >= 6.0`
- `typer >= 0.9`
- `pycocotools >= 2.0`
- `torchmetrics >= 1.6`
- `albumentations >= 1.4`
- `scikit-learn >= 1.6` (fold splitting)
- `pandas >= 2.2` (CSV)
- `tqdm`

**Optional:**
- `timm >= 0.9` (Swin Transformer backbones)

**Dropped:**
- MMDetection, MMEngine, MMCV, MMCV-full
- `vision-transformers`
- `supervision`
- `torchinfo`
- `matplotlib`, `seaborn` (keep in dev extras)
- Duplicate DETR utility code from `facebookresearch/detr`

Dependencies declared in `pyproject.toml` with optional groups (`[project.optional-dependencies]`):

```toml
[project.optional-dependencies]
swin = ["timm>=0.9"]
dev = ["matplotlib", "seaborn", "pytest"]
```

## Migration Path

1. Create the new structure alongside the existing code (no breaking changes)
2. Port detectors one at a time: YOLOv8 → Faster R-CNN → DETR
3. Verify against current results CSV for mAP parity
4. Delete legacy `src/` and `automation/` directories
5. Update `README.md` with new CLI usage

## Scope Boundaries

**In scope:**
- New project structure, Detector ABC, config system, CLI, train/eval/sweep
- Port YOLOv8, Faster R-CNN, DETR to new interface
- Runtime fold generation
- Metrics parity with current `ResultsDetections.py`

**Out of scope:**
- SwinDetector, RT-DETR (add after the refactor using the new pattern)
- MMDetection support (dropped)
- Data augmentation tiling scripts (keep as-is in `utils/`)
- R visualization scripts (keep as-is in `utils/`)
- Migration of existing training results
