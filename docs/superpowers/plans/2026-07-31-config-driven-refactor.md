# Config-Driven Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the benchmarking framework into a config-driven architecture where adding a new detector requires one file + one registry line, with YAML+CLI configuration, decoupled train/eval, and reduced dependencies.

**Architecture:** A `Detector` ABC with per-family implementations (yolov8, faster_rcnn, detr). YAML config with CLI overrides (Typer). Generic torchvision training loop in `engine/trainer.py`. Runtime fold splitting from COCO JSON. COCO eval + custom counting metrics in `engine/evaluator.py`.

**Tech Stack:** PyTorch 2.1+, torchvision 0.16+, Ultralytics 8.2+, Typer, PyYAML, pycocotools, torchmetrics, albumentations, scikit-learn, pandas

## Global Constraints

- Drop MMDetection, vision-transformers, supervision, torchinfo
- Keep existing `dataset/all/` directory structure (COCO JSON + images)
- No breakage of existing legacy code — new structure lives alongside it
- Python 3.9+ (matches current conda env)
- Folds generated at runtime, no pre-computed JSON files
- `pyproject.toml` is single source of deps
- All detector hparams documented as YAML comments via `cli.py config --dump`

## File Map

| File | Create/Modify | Responsibility |
|------|--------------|----------------|
| `pyproject.toml` | Create | Dependencies, package metadata |
| `detectors/__init__.py` | Create | DETECTOR_REGISTRY dict |
| `detectors/base.py` | Create | Detector ABC |
| `detectors/faster_rcnn.py` | Create | FasterRCNNDetector (torchvision) |
| `detectors/yolov8.py` | Create | YOLOV8Detector (ultralytics) |
| `detectors/detr.py` | Create | DETRDetector (torchvision) |
| `data/__init__.py` | Create | Empty |
| `data/dataset.py` | Create | COCODataset (reads COCO JSON) |
| `data/transforms.py` | Create | Albumentations → torch tensor transforms |
| `engine/__init__.py` | Create | Empty |
| `engine/trainer.py` | Create | Generic torchvision training loop |
| `engine/evaluator.py` | Create | COCO eval + custom metrics |
| `engine/sweeper.py` | Create | Grid generation → train+eval |
| `utils/__init__.py` | Create | Empty |
| `utils/logging.py` | Create | setup_logging() |
| `utils/folds.py` | Create | split_folds() — runtime CV split |
| `utils/config.py` | Create | YAML loading, merge, config dump |
| `utils/results.py` | Create | CSV collation + fold aggregation |
| `cli.py` | Create | Typer CLI (train, eval, sweep, config, aggregate) |
| `configs/default.yaml` | Create | Shipped defaults, all hparams commented |
| `configs/sweeps/baseline.yaml` | Create | Example grid sweep |

---

### Task 1: Scaffold project structure

**Files:**
- Create: `pyproject.toml`
- Create: `detectors/__init__.py` (empty), `engine/__init__.py` (empty), `data/__init__.py` (empty), `utils/__init__.py` (empty)
- Create: `configs/sweeps/.gitkeep`

**Interfaces:**
- Produces: `pyproject.toml` with all core + optional deps, package directories ready for imports

**Steps:**

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p detectors engine data utils configs/sweeps
touch detectors/__init__.py engine/__init__.py data/__init__.py utils/__init__.py
touch configs/sweeps/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "compara-detectores"
version = "0.2.0"
description = "Config-driven benchmarking framework for object detectors"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.1",
    "torchvision>=0.16",
    "ultralytics>=8.2",
    "PyYAML>=6.0",
    "typer>=0.9",
    "pycocotools>=2.0",
    "torchmetrics>=1.6",
    "albumentations>=1.4",
    "scikit-learn>=1.6",
    "pandas>=2.2",
    "tqdm",
]

[project.optional-dependencies]
swin = ["timm>=0.9"]
dev = ["matplotlib", "seaborn", "pytest"]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Verify structure**

```bash
python -c "import detectors; import engine; import data; import utils; print('All packages importable')"
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml detectors/__init__.py engine/__init__.py data/__init__.py utils/__init__.py configs/
git commit -m "feat: scaffold project structure with pyproject.toml deps"
```

---

### Task 2: Detector ABC, registry, and logging

**Files:**
- Create: `detectors/base.py`
- Modify: `detectors/__init__.py`
- Create: `utils/logging.py`

**Interfaces:**
- Produces:
  - `Detector` ABC with `train()`, `predict()`, `load()`, `architectures()`, `default_hparams()`
  - `DETECTOR_REGISTRY: dict[str, type[Detector]]` in `detectors/__init__.py`
  - `setup_logging(level: str) -> None` in `utils/logging.py`

**Steps:**

- [ ] **Step 1: Write `detectors/base.py`**

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Detector(ABC):
    @abstractmethod
    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        config: dict,
        output_dir: Path,
    ) -> Path:
        """Run training. Return path to best checkpoint.

        config dict contains:
            architecture: str    # e.g. "yolov8s"
            num_classes: int
            + all keys from default_hparams() overridden by YAML/CLI
        """

    @abstractmethod
    def predict(self, images: list) -> list:
        """images: list of torch.Tensor [C,H,W]. Returns list of [{boxes, scores, labels}] where:
        - boxes: Tensor [N, 4] in xyxy format
        - scores: Tensor [N]
        - labels: Tensor [N] int64 class ids
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model weights from checkpoint path."""

    @classmethod
    @abstractmethod
    def architectures(cls) -> list[str]:
        """Supported architecture variant names (e.g. ['yolov8n', 'yolov8s'])."""

    @classmethod
    @abstractmethod
    def default_hparams(cls) -> dict:
        """Per-family hyperparameter defaults with type-annotated descriptions.

        Each value should be a dict with 'default' and 'help' keys.
        Example: {'lr': {'default': 0.001, 'help': '(float) initial learning rate'}, ...}
        """
```

- [ ] **Step 2: Verify ABC cannot be instantiated**

```bash
python -c "
from detectors.base import Detector
try:
    d = Detector()
    print('FAIL: should raise TypeError')
except TypeError as e:
    print('OK:', e)
"
```

- [ ] **Step 3: Write `detectors/__init__.py`**

```python
from detectors.base import Detector

DETECTOR_REGISTRY: dict[str, type[Detector]] = {}
```

- [ ] **Step 4: Write `utils/logging.py`**

```python
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
```

- [ ] **Step 5: Commit**

```bash
git add detectors/base.py detectors/__init__.py utils/logging.py
git commit -m "feat: Detector ABC, registry, and logging utility"
```

---

### Task 3: Data pipeline — COCODataset, transforms, fold splitting

**Files:**
- Create: `data/dataset.py`
- Create: `data/transforms.py`
- Create: `utils/folds.py`

**Interfaces:**
- Produces:
  - `class COCODataset(torch.utils.data.Dataset)` — loads COCO JSON, returns (image_tensor, target_dict)
  - `def get_train_transforms(img_size: int) -> A.Compose` — Albumentations train pipeline
  - `def get_val_transforms(img_size: int) -> A.Compose` — Albumentations val pipeline
  - `def split_folds(images: list, annotations: list, n_folds: int, val_ratio: float, seed: int) -> list[dict]` — returns list of `{'train': [ids], 'val': [ids], 'test': [ids]}`

**Steps:**

- [ ] **Step 1: Write `utils/folds.py`**

```python
"""Runtime cross-validation fold splitting from COCO images/annotations."""

import random
from sklearn.model_selection import train_test_split


def split_folds(
    images: list[dict],
    annotations: list[dict],
    n_folds: int,
    val_ratio: float,
    seed: int = 42,
) -> list[dict]:
    """Split COCO images into n_folds of (train, val, test) image_id sets.

    Returns list of dicts with 'train', 'val', 'test' keys, each a list of image_ids.
    Uses sequential splitting after shuffle for deterministic, non-overlapping test sets.
    """
    random.seed(seed)
    image_ids = [img["id"] for img in images]
    random.shuffle(image_ids)

    folds = []
    fold_size = len(image_ids) // n_folds
    remainder = len(image_ids) % n_folds

    start = 0
    for i in range(n_folds):
        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        test_ids = image_ids[start:end]

        remaining_ids = [x for x in image_ids if x not in test_ids]
        train_ids, val_ids = train_test_split(
            remaining_ids, test_size=val_ratio, random_state=seed
        )
        folds.append({"train": train_ids, "val": val_ids, "test": test_ids})
        start = end

    return folds
```

- [ ] **Step 2: Write `data/transforms.py`**

```python
"""Albumentations-based transform pipelines returning torch tensors."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )
```

- [ ] **Step 3: Write `data/dataset.py`**

```python
"""COCO-format PyTorch Dataset."""

import json
import cv2
import torch
from pathlib import Path
from typing import Optional


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_json: str,
        images_dir: str,
        image_ids: Optional[list[int]] = None,
        transforms=None,
    ):
        with open(coco_json, "r") as f:
            coco = json.load(f)

        self.images_dir = Path(images_dir)
        self.transforms = transforms

        self.images = {img["id"]: img for img in coco["images"]}
        if image_ids is not None:
            self.images = {k: v for k, v in self.images.items() if k in image_ids}

        self._anns_by_image: dict[int, list] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.images:
                self._anns_by_image.setdefault(img_id, []).append(ann)

        self.image_ids = sorted(self.images.keys())
        self.num_classes = len(coco["categories"])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        image = cv2.imread(str(self.images_dir / img_info["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self._anns_by_image.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes, labels=labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        return image, target
```

- [ ] **Step 4: Verify dataset loads**

```bash
python -c "
from data.dataset import COCODataset
ds = COCODataset('dataset/all/train/_annotations.coco.json', 'dataset/all/train')
print(f'Loaded {len(ds)} images, {ds.num_classes} classes')
img, target = ds[0]
print(f'Image shape: {img.shape}, Boxes: {target[\"boxes\"].shape}')
"
```

- [ ] **Step 5: Commit**

```bash
git add data/dataset.py data/transforms.py utils/folds.py
git commit -m "feat: COCODataset, transforms, and runtime fold splitting"
```

---

### Task 4: Config system — YAML loading, merge, and dump

**Files:**
- Create: `utils/config.py`

**Interfaces:**
- Produces:
  - `def load_config(yaml_path: str | None, cli_overrides: dict | None) -> dict` — merged config dict (CLI > YAML > defaults)
  - `def dump_commented_config(output_path: str) -> None` — writes commented YAML with all registerd detector hparams
  - Merge order: CLI > YAML file > internal `_DEFAULT_CONFIG` dict

**Steps:**

- [ ] **Step 1: Write `utils/config.py`**

```python
"""YAML config loading, three-layer merge, and commented config dump."""

import yaml
from pathlib import Path
from typing import Any, Optional

from detectors import DETECTOR_REGISTRY

_DEFAULT_CONFIG = {
    "experiment": "default",
    "dataset": {
        "coco_json": "dataset/all/train/_annotations.coco.json",
        "images_dir": "dataset/all/train",
    },
    "folds": {"n_folds": 5, "val_ratio": 0.2, "seed": 42},
    "output_dir": "results",
    "seed": 42,
    "device": "cuda",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists replace, dicts merge."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> dict:
    """Load and merge configuration: CLI > YAML file > defaults."""
    config = _DEFAULT_CONFIG.copy()

    if yaml_path:
        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, yaml_config)

    if cli_overrides:
        config = _deep_merge(config, cli_overrides)

    return config


def _build_detector_defaults() -> dict[str, Any]:
    """Build per-detector defaults from registered Detector.default_hparams()."""
    detector_defaults = {}
    for name, det_cls in DETECTOR_REGISTRY.items():
        hparams = {}
        for key, info in det_cls.default_hparams().items():
            hparams[key] = info["default"]
        detector_defaults[name] = {
            "architectures": det_cls.architectures(),
            "hparams": hparams,
        }
    return detector_defaults


def dump_commented_config(output_path: str) -> None:
    """Generate a commented YAML config with all registered detectors and their hparams."""
    detector_defaults = _build_detector_defaults()

    lines = [
        "# =============================================================================",
        "# Config-Driven Detector Benchmark — Generated Configuration",
        "# =============================================================================",
        "# All values shown are factory defaults from each detector's implementation.",
        "# Uncomment and edit any parameter to override.",
        "# =============================================================================",
        "",
        "experiment: default",
        "",
        "dataset:",
        f"  coco_json: {_DEFAULT_CONFIG['dataset']['coco_json']}",
        f"  images_dir: {_DEFAULT_CONFIG['dataset']['images_dir']}",
        "",
        "folds:",
        f"  n_folds: {_DEFAULT_CONFIG['folds']['n_folds']}",
        f"  val_ratio: {_DEFAULT_CONFIG['folds']['val_ratio']}",
        f"  seed: {_DEFAULT_CONFIG['folds']['seed']}",
        "",
        f"output_dir: {_DEFAULT_CONFIG['output_dir']}",
        f"seed: {_DEFAULT_CONFIG['seed']}",
        f"device: {_DEFAULT_CONFIG['device']}",
        "",
        "# " + "-" * 68,
        "# DETECTOR HYPERPARAMETERS",
        "# Uncomment any parameter to override the default.",
        "# " + "-" * 68,
        "",
        "detectors:",
    ]

    for name, det_cls in DETECTOR_REGISTRY.items():
        hparams = det_cls.default_hparams()
        arches = det_cls.architectures()
        lines.append(f"  {name}:")
        lines.append(f"    architectures: {arches}")
        lines.append(f"    # hparams: {'-' * 42}")
        for key, info in hparams.items():
            default_val = info["default"]
            help_str = info["help"]
            if isinstance(default_val, str):
                default_str = repr(default_val)
            else:
                default_str = str(default_val)
            pad = " " * max(1, 20 - len(key) - len(default_str))
            lines.append(f"    #   {key}: {default_str}{pad}# {help_str}")
        lines.append("")

    lines.extend([
        "# " + "-" * 68,
        "# SWEEP MODE",
        "# Replace scalar values with lists to create a grid sweep.",
        "# Example:",
        "#   yolov8:",
        "#     architectures: [yolov8n, yolov8s, yolov8m]",
        "#     hparams:",
        "#       lr: [0.0001, 0.001, 0.01]",
        "#       batch_size: [16, 32]",
        "# " + "-" * 68,
    ])

    content = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(content)
```

- [ ] **Step 2: Verify config loading (no detectors registered yet, but defaults work)**

```bash
python -c "
from utils.config import load_config
cfg = load_config()
print(cfg['dataset']['coco_json'])
print(cfg['folds'])
"
```

- [ ] **Step 3: Commit**

```bash
git add utils/config.py
git commit -m "feat: config system with YAML loading and commented dump"
```

---

### Task 5: Training engine — generic torchvision training loop

**Files:**
- Create: `engine/trainer.py`

**Interfaces:**
- Produces: `def train_torchvision_model(model: nn.Module, train_loader, val_loader, config: dict, output_dir: Path) -> Path` — trains a torchvision detection model, returns path to `output_dir/best.pth`

**Steps:**

- [ ] **Step 1: Write `engine/trainer.py`**

```python
"""Generic training loop for torchvision detection models."""

import csv
import logging
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = config.get("optimizer", "SGD").lower()
    lr = config.get("lr", 0.001)
    wd = config.get("weight_decay", 0.0005)
    momentum = config.get("momentum", 0.9)

    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_name = config.get("scheduler", "none").lower()
    epochs = config.get("epochs", 100)

    if sched_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        step_size = config.get("step_size", 10)
        gamma = config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.1
        )
    return None


def _train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(data_loader, desc=f"Train epoch {epoch + 1}", leave=False)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Filter images with no boxes (torchvision models require at least one box)
        valid = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
        if len(valid) == 0:
            continue
        images = [images[i] for i in valid]
        targets = [targets[i] for i in valid]

        loss_dict = model(images, targets)
        losses = sum(v for v in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{losses.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        valid = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
        if len(valid) == 0:
            continue
        images = [images[i] for i in valid]
        targets = [targets[i] for i in valid]

        loss_dict = model(images, targets)
        losses = sum(v for v in loss_dict.values())
        total_loss += losses.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_torchvision_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    output_dir: Path,
) -> Path:
    """Train a torchvision detection model. Returns path to best checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.get("device", "cuda"))
    epochs = config.get("epochs", 30)
    patience = config.get("patience", 5)

    model.to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    best_loss = float("inf")
    best_path = output_dir / "best.pth"
    patience_counter = 0
    log_path = output_dir / "logs.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss = _validate(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1}/{epochs} — train_loss: {train_loss:.4f}, "
            f"val_loss: {val_loss:.4f}, lr: {current_lr:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, current_lr])

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
                best_path,
            )
        else:
            patience_counter += 1

        last_path = output_dir / "last.pth"
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
            last_path,
        )

        if patience > 0 and patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"Training complete. Best model: {best_path}")
    return best_path
```

- [ ] **Step 2: Verify import**

```bash
python -c "from engine.trainer import train_torchvision_model; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add engine/trainer.py
git commit -m "feat: generic torchvision training loop with optimizer/scheduler/early-stopping"
```

---

### Task 6: CLI scaffold + train command

**Files:**
- Create: `cli.py`

**Interfaces:**
- Typer app with `train` subcommand: `python cli.py train --detector X --arch Y -c config.yaml`
- Consumes: `utils.config.load_config`, `detectors.DETECTOR_REGISTRY`, `utils.folds.split_folds`, `utils.logging.setup_logging`, `data.dataset.COCODataset`, `data.transforms.get_train_transforms`, `data.transforms.get_val_transforms`

**Steps:**

- [ ] **Step 1: Write `cli.py`**

```python
#!/usr/bin/env python
"""Config-Driven Detector Benchmark CLI."""

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from torch.utils.data import DataLoader

from utils.logging import setup_logging
from utils.config import load_config
from utils.folds import split_folds
from data.dataset import COCODataset
from data.transforms import get_train_transforms, get_val_transforms
from detectors import DETECTOR_REGISTRY

app = typer.Typer()
logger = logging.getLogger(__name__)


def _collate_fn(batch):
    return tuple(zip(*batch))


@app.command()
def train(
    detector: str = typer.Option(..., "--detector", "-d", help="Detector family name (e.g. yolov8, faster_rcnn)"),
    arch: str = typer.Option(..., "--arch", "-a", help="Architecture variant (e.g. yolov8s, resnet50)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    fold: int = typer.Option(0, "--fold", "-f", help="Fold index (0-based, 0 = run all folds)"),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate override"),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Epochs override"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Batch size override"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed override"),
):
    """Train a single detector architecture."""
    setup_logging("INFO")

    cli_overrides = {"seed": seed} if seed is not None else {}
    if lr is not None or epochs is not None or batch_size is not None:
        cli_overrides["detectors"] = {detector: {"hparams": {}}}
        if lr is not None:
            cli_overrides["detectors"][detector]["hparams"]["lr"] = lr
        if epochs is not None:
            cli_overrides["detectors"][detector]["hparams"]["epochs"] = epochs
        if batch_size is not None:
            cli_overrides["detectors"][detector]["hparams"]["batch_size"] = batch_size

    config = load_config(str(config_path) if config_path else None, cli_overrides)

    det_cls = DETECTOR_REGISTRY.get(detector)
    if det_cls is None:
        typer.echo(f"Unknown detector '{detector}'. Available: {list(DETECTOR_REGISTRY.keys())}", err=True)
        raise typer.Exit(1)
    if arch not in det_cls.architectures():
        typer.echo(f"Unknown architecture '{arch}' for {detector}. Available: {det_cls.architectures()}", err=True)
        raise typer.Exit(1)

    det_hparams = det_cls.default_hparams()
    run_config = {key: info["default"] for key, info in det_hparams.items()}

    if "detectors" in config and detector in config["detectors"]:
        det_config = config["detectors"][detector]
        if "hparams" in det_config:
            run_config.update(det_config["hparams"])

    with open(config["dataset"]["coco_json"], "r") as f:
        coco = json.load(f)

    folds = split_folds(
        coco["images"],
        coco["annotations"],
        config["folds"]["n_folds"],
        config["folds"]["val_ratio"],
        config["seed"],
    )

    run_config["num_classes"] = len(coco["categories"])
    run_config["architecture"] = arch
    run_config["device"] = config.get("device", "cuda")

    images_dir = config["dataset"]["images_dir"]
    coco_json = config["dataset"]["coco_json"]
    img_size = run_config.get("imgsz", 640)

    fold_indices = range(len(folds)) if fold == 0 else [fold - 1]

    for fi in fold_indices:
        f = folds[fi]
        output_dir = (
            Path(config["output_dir"])
            / config.get("experiment", "default")
            / f"fold_{fi + 1}"
            / arch
        )
        logger.info(f"Fold {fi + 1}/{len(folds)} — output: {output_dir}")

        train_ds = COCODataset(coco_json, images_dir, f["train"], get_train_transforms(img_size))
        val_ds = COCODataset(coco_json, images_dir, f["val"], get_val_transforms(img_size))
        train_loader = DataLoader(train_ds, batch_size=run_config["batch_size"], shuffle=True, collate_fn=_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=run_config["batch_size"], shuffle=False, collate_fn=_collate_fn)

        det_instance = det_cls()
        best_path = det_instance.train(train_loader, val_loader, run_config, output_dir)
        logger.info(f"Fold {fi + 1} complete — best model: {best_path}")

    typer.echo("Training complete.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify CLI starts**

```bash
python cli.py --help
python cli.py train --help
```

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat: CLI scaffold with train subcommand"
```

---

### Task 7: Faster R-CNN detector — first end-to-end path

**Files:**
- Create: `detectors/faster_rcnn.py`
- Modify: `detectors/__init__.py`

**Interfaces:**
- Produces: `class FasterRCNNDetector(Detector)` — torchvision Faster R-CNN with ResNet-50/101 backbones
- Consumes: `engine.trainer.train_torchvision_model`

**Steps:**

- [ ] **Step 1: Write `detectors/faster_rcnn.py`**

```python
"""Faster R-CNN detector using torchvision."""

from pathlib import Path

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detectors.base import Detector
from engine.trainer import train_torchvision_model


class FasterRCNNDetector(Detector):
    def __init__(self):
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def architectures(cls) -> list[str]:
        return ["resnet50", "resnet101"]

    @classmethod
    def default_hparams(cls) -> dict:
        return {
            "lr": {"default": 0.0001, "help": "(float) initial learning rate"},
            "epochs": {"default": 30, "help": "(int)"},
            "batch_size": {"default": 8, "help": "(int)"},
            "optimizer": {"default": "SGD", "help": "SGD | AdamW | Adam"},
            "weight_decay": {"default": 0.0005, "help": "(float)"},
            "momentum": {"default": 0.9, "help": "(float) SGD momentum"},
            "scheduler": {"default": "step", "help": "step | cosine | plateau | none"},
            "step_size": {"default": 10, "help": "(int) for step scheduler"},
            "gamma": {"default": 0.1, "help": "(float) LR decay factor"},
            "patience": {"default": 5, "help": "(int) early stopping, 0 = disabled"},
        }

    def _build_model(self, arch: str, num_classes: int) -> torch.nn.Module:
        if arch == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        elif arch == "resnet101":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                "resnet101", weights=None
            )
            model.backbone = backbone
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train(self, train_loader, val_loader, config: dict, output_dir: Path) -> Path:
        model = self._build_model(config["architecture"], config["num_classes"])
        self.model = model
        return train_torchvision_model(model, train_loader, val_loader, config, output_dir)

    def predict(self, images: list) -> list:
        self.model.eval()
        self.model.to(self.device)
        images = [img.to(self.device) for img in images]
        with torch.no_grad():
            outputs = self.model(images)
        results = []
        for output in outputs:
            results.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
        return results

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
```

- [ ] **Step 2: Register in `detectors/__init__.py`**

Edit `detectors/__init__.py` to:
```python
from detectors.base import Detector
from detectors.faster_rcnn import FasterRCNNDetector

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "faster_rcnn": FasterRCNNDetector,
}
```

- [ ] **Step 3: Quick smoke test (1 epoch)**

```bash
python cli.py train --detector faster_rcnn --arch resnet50 --epochs 1 --batch-size 2 2>&1 | head -20
```

- [ ] **Step 4: Commit**

```bash
git add detectors/faster_rcnn.py detectors/__init__.py
git commit -m "feat: Faster R-CNN detector (torchvision)"
```

---

### Task 8: Evaluation engine — COCO mAP + custom metrics

**Files:**
- Create: `engine/evaluator.py`

**Interfaces:**
- Produces:
  - `def compute_coco_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict`
  - `def compute_counting_metrics(pred_counts: Tensor, gt_counts: Tensor) -> dict`
  - `def classify_predictions(pred_per_image, gt_per_image, iou_threshold, classes) -> tuple[list, list, int, int]`
  - `def compute_classification_metrics(pred_labels, gt_labels, num_classes) -> dict`
  - `def evaluate_detector(detector_instance, test_loader, classes, iou_threshold=0.2) -> dict`
  - `def save_metrics(metrics: dict, output_path: Path) -> None`
  - `def append_csv_row(csv_path: Path, row: dict) -> None`

**Steps:**

- [ ] **Step 1: Write `engine/evaluator.py`**

```python
"""Evaluation: COCO mAP + custom counting/classification metrics."""

import csv
import json
import logging
from pathlib import Path

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score

logger = logging.getLogger(__name__)


def _calculate_iou(box1_xywh, box2_xywh):
    x1, y1, w1, h1 = box1_xywh
    x2, y2, w2, h2 = box2_xywh
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    ix = max(0, min(x1_max, x2_max) - max(x1, x2))
    iy = max(0, min(y1_max, y2_max) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def compute_coco_metrics(predictions: list[dict], ground_truths: list[dict]) -> dict:
    metric = MeanAveragePrecision()
    metric.update(predictions, ground_truths)
    result = metric.compute()
    return {
        "mAP": result["map"].item(),
        "mAP50": result["map_50"].item(),
        "mAP75": result["map_75"].item(),
    }


def compute_counting_metrics(pred_counts, gt_counts) -> dict:
    return {
        "MAE": MeanAbsoluteError()(pred_counts, gt_counts).item(),
        "RMSE": MeanSquaredError(squared=False)(pred_counts, gt_counts).item(),
        "pearson_r": PearsonCorrCoef()(pred_counts.float(), gt_counts.float()).item(),
    }


def classify_predictions(
    predictions_per_image: dict[str, list],
    ground_truth_per_image: dict[str, list],
    iou_threshold: float,
    classes: dict,
) -> tuple[list[int], list[int], int, int]:
    """Match predictions to ground truth per image. Returns (pred_labels, gt_labels, total_tp, total_fp)."""
    pred_labels = []
    gt_labels = []
    total_tp = 0
    total_fp = 0

    for img_name in predictions_per_image:
        gt_bboxes = ground_truth_per_image.get(img_name, [])
        pred_bboxes = predictions_per_image[img_name]
        matched_gt = set()

        for pred in pred_bboxes:
            best_iou = 0.0
            best_gt_idx = None
            for i, gt in enumerate(gt_bboxes):
                iou = _calculate_iou(pred[:4], gt[:4])
                if iou >= iou_threshold and iou > best_iou and i not in matched_gt:
                    best_iou = iou
                    best_gt_idx = i
            if best_gt_idx is not None:
                matched_gt.add(best_gt_idx)
                gt_class = gt_bboxes[best_gt_idx][4]
                pred_class = int(pred[4])
                pred_labels.append(pred_class)
                gt_labels.append(gt_class)
                if gt_class == pred_class:
                    total_tp += 1
                else:
                    total_fp += 1
            else:
                pred_labels.append(int(pred[4]))
                gt_labels.append(0)
                total_fp += 1

        for i, gt in enumerate(gt_bboxes):
            if i not in matched_gt:
                gt_labels.append(gt[4])
                pred_labels.append(0)

    return pred_labels, gt_labels, total_tp, total_fp


def compute_classification_metrics(pred_labels, gt_labels, num_classes) -> dict:
    preds = torch.tensor(pred_labels)
    targets = torch.tensor(gt_labels)
    return {
        "precision": BinaryPrecision()(preds, targets).item(),
        "recall": BinaryRecall()(preds, targets).item(),
        "f1": BinaryF1Score()(preds, targets).item(),
    }


def evaluate_detector(detector_instance, test_loader, classes, iou_threshold=0.2) -> dict:
    """Run full evaluation: predict all images, compute all metrics."""
    predictions_map = []
    ground_truths_map = []
    pred_per_image = {}
    gt_per_image = {}
    pred_counts = []
    gt_counts = []

    for images, targets in test_loader:
        outputs = detector_instance.predict(list(images))

        for img_idx, target in enumerate(targets):
            img_id = target["image_id"].item()
            gt_boxes = target["boxes"].tolist()
            gt_labels = target["labels"].tolist()

            output = outputs[img_idx]
            pred_boxes = output["boxes"].tolist()
            pred_scores = output["scores"].tolist()
            pred_labels_out = output["labels"].tolist()

            pred_boxes_xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in pred_boxes]
            gt_boxes_xyxy = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in gt_boxes]

            pred_per_image[str(img_id)] = [
                pred_boxes_xywh[i] + [pred_labels_out[i], pred_scores[i]]
                for i in range(len(pred_boxes))
            ]
            gt_per_image[str(img_id)] = [
                gt_boxes_xyxy[i] + [gt_labels[i]] for i in range(len(gt_boxes))
            ]
            pred_counts.append(len(pred_boxes))
            gt_counts.append(len(gt_boxes))

            predictions_map.append({
                "boxes": output["boxes"],
                "scores": output["scores"],
                "labels": output["labels"],
            })
            ground_truths_map.append({
                "boxes": target["boxes"],
                "labels": target["labels"],
            })

    coco = compute_coco_metrics(predictions_map, ground_truths_map)
    counting = compute_counting_metrics(
        torch.tensor(pred_counts), torch.tensor(gt_counts)
    )
    pred_lbl, gt_lbl, tp, fp = classify_predictions(
        pred_per_image, gt_per_image, iou_threshold, classes
    )
    class_metrics = compute_classification_metrics(pred_lbl, gt_lbl, len(classes))

    return {
        **coco, **counting, **class_metrics,
        "TP": tp, "FP": fp, "total_preds": len(pred_lbl),
    }


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics, indent=2))


def append_csv_row(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
```

- [ ] **Step 2: Verify import**

```bash
python -c "from engine.evaluator import evaluate_detector; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add engine/evaluator.py
git commit -m "feat: evaluation engine with COCO mAP and custom metrics"
```

---

### Task 9: CLI eval command

**Files:**
- Modify: `cli.py`

**Interfaces:**
- Adds `eval` subcommand: `python cli.py eval -d faster_rcnn -w results/run/fold_1/resnet50/best.pth -c configs/default.yaml`

**Steps:**

- [ ] **Step 1: Add `eval` command and imports to `cli.py`**

Add to top imports in `cli.py`:
```python
from engine.evaluator import evaluate_detector, save_metrics, append_csv_row
```

Add before `if __name__ == "__main__":`:
```python
@app.command()
def eval(
    detector: str = typer.Option(..., "--detector", "-d"),
    weights: Path = typer.Option(..., "--weights", "-w", help="Path to best.pth checkpoint"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    fold: int = typer.Option(1, "--fold", "-f", help="Fold index (1-based)"),
    iou: float = typer.Option(0.2, "--iou", help="IoU threshold for classification matching"),
):
    """Evaluate a trained detector checkpoint on its test fold."""
    import json
    setup_logging("INFO")

    config = load_config(str(config_path) if config_path else None)

    det_cls = DETECTOR_REGISTRY.get(detector)
    if det_cls is None:
        typer.echo(f"Unknown detector '{detector}'. Available: {list(DETECTOR_REGISTRY.keys())}", err=True)
        raise typer.Exit(1)

    with open(config["dataset"]["coco_json"], "r") as f:
        coco = json.load(f)

    folds = split_folds(
        coco["images"], coco["annotations"],
        config["folds"]["n_folds"], config["folds"]["val_ratio"], config["seed"],
    )
    classes = {cat["id"]: cat["name"] for cat in coco["categories"]}

    fi = fold - 1
    if fi < 0 or fi >= len(folds):
        typer.echo(f"Fold {fold} out of range (1-{len(folds)})", err=True)
        raise typer.Exit(1)

    f = folds[fi]
    images_dir = config["dataset"]["images_dir"]
    coco_json = config["dataset"]["coco_json"]
    output_dir = Path(config["output_dir"]) / config.get("experiment", "default")
    fold_dir = output_dir / f"fold_{fold}"

    if not weights.exists():
        arch = weights.parent.name if weights.parent.name != "." else "unknown"
        weights = fold_dir / arch / "best.pth"

    logger.info(f"Fold {fold}/{len(folds)} — weights: {weights}")

    test_ds = COCODataset(coco_json, images_dir, f["test"], get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn)

    det_instance = det_cls()
    det_instance.load(weights)
    metrics = evaluate_detector(det_instance, test_loader, classes, iou)

    fold_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, fold_dir / "metrics.json")
    append_csv_row(output_dir / "summary.csv", {
        "ml": detector, "fold": f"fold_{fold}", **metrics,
    })
    typer.echo(f"Fold {fold}: mAP={metrics['mAP']:.4f} mAP50={metrics['mAP50']:.4f} MAE={metrics['MAE']:.2f}")
```

- [ ] **Step 2: Verify eval command appears**

```bash
python cli.py eval --help
```

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat: CLI eval command"
```

---

### Task 10: YOLOv8 detector

**Files:**
- Create: `detectors/yolov8.py`
- Modify: `detectors/__init__.py`

**Interfaces:**
- `class YOLOV8Detector(Detector)` — Ultralytics YOLO for train/predict

**Steps:**

- [ ] **Step 1: Write `detectors/yolov8.py`**

```python
"""YOLOv8 detector using Ultralytics."""

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from detectors.base import Detector


class YOLOV8Detector(Detector):
    def __init__(self):
        self.model: YOLO | None = None

    @classmethod
    def architectures(cls) -> list[str]:
        return ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

    @classmethod
    def default_hparams(cls) -> dict:
        return {
            "lr": {"default": 0.001, "help": "(float) initial learning rate"},
            "epochs": {"default": 100, "help": "(int)"},
            "batch_size": {"default": 16, "help": "(int)"},
            "optimizer": {"default": "AdamW", "help": "AdamW | SGD | Adam"},
            "weight_decay": {"default": 0.0005, "help": "(float)"},
            "momentum": {"default": 0.937, "help": "(float) SGD momentum"},
            "scheduler": {"default": "cosine", "help": "cosine | linear | step | none"},
            "warmup_epochs": {"default": 3, "help": "(int)"},
            "patience": {"default": 10, "help": "(int) early stopping, 0 = disabled"},
            "imgsz": {"default": 640, "help": "(int) input image size"},
            "workers": {"default": 8, "help": "(int) dataloader workers"},
        }

    def _build_data_yaml(self, train_dir: str, val_dir: str, num_classes: int) -> str:
        data = {
            "path": ".",
            "train": train_dir,
            "val": val_dir,
            "names": {i: str(i) for i in range(num_classes)},
            "nc": num_classes,
        }
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="yolo_data_")
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        return path

    def train(self, train_loader, val_loader, config: dict, output_dir: Path) -> Path:
        arch = config["architecture"]
        num_classes = config["num_classes"]
        img_size = config.get("imgsz", 640)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_img_dir = output_dir / "train_images"
        val_img_dir = output_dir / "val_images"
        train_label_dir = output_dir / "train_labels"
        val_label_dir = output_dir / "val_labels"
        for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for loader, img_dir, lbl_dir in [
            (train_loader, train_img_dir, train_label_dir),
            (val_loader, val_img_dir, val_label_dir),
        ]:
            for images, targets in loader:
                for img_tensor, target in zip(images, targets):
                    boxes = target["boxes"].tolist()
                    labels = target["labels"].tolist()
                    img_id = target["image_id"].item()
                    img_name = f"{img_id}.jpg"

                    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(img_dir / img_name), img_np)

                    with open(lbl_dir / f"{img_id}.txt", "w") as f:
                        for box, label in zip(boxes, labels):
                            x1, y1, x2, y2 = box
                            cx = ((x1 + x2) / 2) / img_size
                            cy = ((y1 + y2) / 2) / img_size
                            bw = (x2 - x1) / img_size
                            bh = (y2 - y1) / img_size
                            f.write(f"{int(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        data_yaml = self._build_data_yaml(
            str(train_img_dir), str(val_img_dir), num_classes
        )

        model = YOLO(f"{arch}.pt")
        model.train(
            data=data_yaml,
            epochs=config.get("epochs", 100),
            batch=config.get("batch_size", 16),
            imgsz=img_size,
            lr0=config.get("lr", 0.001),
            optimizer=config.get("optimizer", "AdamW"),
            weight_decay=config.get("weight_decay", 0.0005),
            momentum=config.get("momentum", 0.937),
            warmup_epochs=config.get("warmup_epochs", 3),
            patience=config.get("patience", 10) if config.get("patience", 0) > 0 else 0,
            cos_lr=config.get("scheduler") == "cosine",
            project=str(output_dir),
            name="train",
            exist_ok=True,
            verbose=False,
        )

        best_path = output_dir / "best.pt"
        src_best = output_dir / "train" / "weights" / "best.pt"
        if src_best.exists():
            shutil.copy(str(src_best), str(best_path))

        self.model = model
        return best_path

    def predict(self, images: list) -> list:
        results = []
        for img in images:
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            output = self.model(img_np, verbose=False)[0]
            if output.boxes is not None:
                results.append({
                    "boxes": output.boxes.xyxy.cpu(),
                    "scores": output.boxes.conf.cpu(),
                    "labels": output.boxes.cls.cpu().long(),
                })
            else:
                results.append({
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                })
        return results

    def load(self, path: Path) -> None:
        self.model = YOLO(str(path))
```

- [ ] **Step 2: Register in `detectors/__init__.py`**

```python
from detectors.base import Detector
from detectors.faster_rcnn import FasterRCNNDetector
from detectors.yolov8 import YOLOV8Detector

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "faster_rcnn": FasterRCNNDetector,
    "yolov8": YOLOV8Detector,
}
```

- [ ] **Step 3: Verify registration**

```bash
python -c "from detectors import DETECTOR_REGISTRY; print(list(DETECTOR_REGISTRY.keys()))"
```

- [ ] **Step 4: Quick smoke test**

```bash
python cli.py train --detector yolov8 --arch yolov8n --epochs 1 --batch-size 2 2>&1 | head -30
```

- [ ] **Step 5: Commit**

```bash
git add detectors/yolov8.py detectors/__init__.py
git commit -m "feat: YOLOv8 detector (Ultralytics)"
```

---

### Task 11: DETR detector

**Files:**
- Create: `detectors/detr.py`
- Modify: `detectors/__init__.py`

**Interfaces:**
- `class DETRDetector(Detector)` — torchvision DETR (detr_resnet50/101)

**Steps:**

- [ ] **Step 1: Write `detectors/detr.py`**

```python
"""DETR detector using torchvision."""

from pathlib import Path

import torch
import torchvision

from detectors.base import Detector
from engine.trainer import train_torchvision_model


class DETRDetector(Detector):
    def __init__(self):
        self.model: torch.nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def architectures(cls) -> list[str]:
        return ["detr_resnet50", "detr_resnet101"]

    @classmethod
    def default_hparams(cls) -> dict:
        return {
            "lr": {"default": 0.0001, "help": "(float) initial learning rate (backbone lr = lr/10)"},
            "epochs": {"default": 150, "help": "(int)"},
            "batch_size": {"default": 8, "help": "(int)"},
            "optimizer": {"default": "AdamW", "help": "AdamW | Adam"},
            "weight_decay": {"default": 0.0001, "help": "(float)"},
            "scheduler": {"default": "step", "help": "step | cosine | none"},
            "lr_drop": {"default": 100, "help": "(int) epoch to drop LR by 10x (DETR convention)"},
            "patience": {"default": 20, "help": "(int) early stopping, 0 = disabled"},
        }

    def _build_model(self, arch: str, num_classes: int) -> torch.nn.Module:
        if arch == "detr_resnet50":
            model = torchvision.models.detection.detr_resnet50(
                weights="DEFAULT", num_classes=num_classes
            )
        elif arch == "detr_resnet101":
            model = torchvision.models.detection.detr_resnet50(
                weights=None, num_classes=num_classes
            )
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                "resnet101", weights=None
            )
            model.backbone = backbone
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        return model

    def train(self, train_loader, val_loader, config: dict, output_dir: Path) -> Path:
        model = self._build_model(config["architecture"], config["num_classes"])
        self.model = model

        config = dict(config)
        if config.get("scheduler") == "step":
            config["step_size"] = config.pop("lr_drop", 100)

        return train_torchvision_model(model, train_loader, val_loader, config, output_dir)

    def predict(self, images: list) -> list:
        self.model.eval()
        self.model.to(self.device)
        images = [img.to(self.device) for img in images]
        with torch.no_grad():
            outputs = self.model(images)
        results = []
        for output in outputs:
            results.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu(),
            })
        return results

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
```

- [ ] **Step 2: Register in `detectors/__init__.py`**

```python
from detectors.base import Detector
from detectors.faster_rcnn import FasterRCNNDetector
from detectors.yolov8 import YOLOV8Detector
from detectors.detr import DETRDetector

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    "faster_rcnn": FasterRCNNDetector,
    "yolov8": YOLOV8Detector,
    "detr": DETRDetector,
}
```

- [ ] **Step 3: Commit**

```bash
git add detectors/detr.py detectors/__init__.py
git commit -m "feat: DETR detector (torchvision)"
```

---

### Task 12: Sweeper engine

**Files:**
- Create: `engine/sweeper.py`

**Interfaces:**
- Produces:
  - `def generate_experiment_grid(config: dict) -> list[dict]` — cartesian product of sweep combos
  - `def run_sweep(experiments: list[dict], config: dict) -> None` — runs train+eval sequentially

**Steps:**

- [ ] **Step 1: Write `engine/sweeper.py`**

```python
"""Sweep engine: grid generation and train+eval orchestration."""

import itertools
import json
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from data.dataset import COCODataset
from data.transforms import get_train_transforms, get_val_transforms
from detectors import DETECTOR_REGISTRY
from engine.evaluator import evaluate_detector, save_metrics, append_csv_row
from utils.folds import split_folds

logger = logging.getLogger(__name__)


def _is_list(v):
    return isinstance(v, list)


def _collate_fn(batch):
    return tuple(zip(*batch))


def generate_experiment_grid(config: dict) -> list[dict]:
    """Generate all experiment combinations from sweep config.

    Scalar hparams are fixed; list hparams are swept (cartesian product).
    """
    experiments = []
    experiment_name = config.get("experiment", "sweep")
    base_seed = config.get("seed", 42)
    n_folds = config["folds"]["n_folds"]
    detectors = config.get("detectors", {})

    with open(config["dataset"]["coco_json"], "r") as f:
        coco = json.load(f)
    num_classes = len(coco["categories"])

    folds = split_folds(
        coco["images"], coco["annotations"],
        n_folds, config["folds"]["val_ratio"], base_seed,
    )

    for det_name, det_config in detectors.items():
        if det_config is None:
            continue
        det_cls = DETECTOR_REGISTRY.get(det_name)
        if det_cls is None:
            logger.warning(f"Unknown detector: {det_name}, skipping")
            continue

        hparams = det_config.get("hparams", {})
        sweep_keys = [k for k, v in hparams.items() if _is_list(v)]
        fixed_keys = [k for k, v in hparams.items() if not _is_list(v)]

        hparams_full = {key: info["default"] for key, info in det_cls.default_hparams().items()}
        for k in fixed_keys:
            hparams_full[k] = hparams[k]

        architectures = det_config.get("architectures", det_cls.architectures())

        sweep_values = []
        for key in sweep_keys:
            sweep_values.append([(key, v) for v in hparams[key]])
        if not sweep_values:
            sweep_values = [[]]

        for arch in architectures:
            for sweep_combo in itertools.product(*sweep_values):
                hps = dict(hparams_full)
                for key, val in sweep_combo:
                    hps[key] = val

                for fi in range(n_folds):
                    combo_parts = [det_name, arch]
                    combo_parts.extend([f"{k}={v}" for k, v in sweep_combo])
                    combo_name = "_".join(combo_parts)

                    experiments.append({
                        "detector": det_name,
                        "architecture": arch,
                        "hparams": hps,
                        "fold": fi,
                        "num_classes": num_classes,
                        "experiment": experiment_name,
                        "combo_name": combo_name,
                    })

    return experiments


def run_sweep(experiments: list[dict], config: dict) -> None:
    """Run all experiments sequentially."""
    images_dir = config["dataset"]["images_dir"]
    coco_json = config["dataset"]["coco_json"]

    with open(coco_json, "r") as f:
        coco = json.load(f)
    classes = {cat["id"]: cat["name"] for cat in coco["categories"]}
    n_folds = config["folds"]["n_folds"]
    base_seed = config.get("seed", 42)

    folds = split_folds(
        coco["images"], coco["annotations"],
        n_folds, config["folds"]["val_ratio"], base_seed,
    )

    output_dir = Path(config["output_dir"]) / config.get("experiment", "sweep")

    for exp in experiments:
        det_name = exp["detector"]
        arch = exp["architecture"]
        fi = exp["fold"]
        hps = exp["hparams"]
        hps["architecture"] = arch
        hps["num_classes"] = exp["num_classes"]
        hps["device"] = config.get("device", "cuda")
        img_size = hps.get("imgsz", 640)

        fold_dir = output_dir / f"fold_{fi + 1}" / arch
        fold_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{det_name}/{arch}] Fold {fi + 1}/{n_folds}")

        det_instance = DETECTOR_REGISTRY[det_name]()

        f = folds[fi]
        train_ds = COCODataset(coco_json, images_dir, f["train"], get_train_transforms(img_size))
        val_ds = COCODataset(coco_json, images_dir, f["val"], get_val_transforms(img_size))
        train_loader = DataLoader(train_ds, batch_size=hps.get("batch_size", 8), shuffle=True, collate_fn=_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=hps.get("batch_size", 8), shuffle=False, collate_fn=_collate_fn)

        best_path = det_instance.train(train_loader, val_loader, hps, fold_dir)

        test_ds = COCODataset(coco_json, images_dir, f["test"], get_val_transforms(img_size))
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=_collate_fn)

        det_instance.load(best_path)
        metrics = evaluate_detector(det_instance, test_loader, classes)

        save_metrics(metrics, fold_dir / "metrics.json")
        append_csv_row(output_dir / "summary.csv", {
            "detector": det_name,
            "architecture": arch,
            "fold": f"fold_{fi + 1}",
            **{k: v for k, v in hps.items() if k not in ("architecture", "num_classes", "device")},
            **metrics,
        })

    logger.info(f"Sweep complete. Results: {output_dir}")
```

- [ ] **Step 2: Commit**

```bash
git add engine/sweeper.py
git commit -m "feat: sweep engine with grid generation and sequential execution"
```

---

### Task 13: CLI — sweep, config dump, and aggregate commands

**Files:**
- Modify: `cli.py`

**Interfaces:**
- `sweep`: `python cli.py sweep --config configs/sweeps/baseline.yaml`
- `config --dump`: `python cli.py config --dump > my.yaml`
- `aggregate`: `python cli.py aggregate results/baseline_v2/`

**Steps:**

- [ ] **Step 1: Add remaining commands to `cli.py`**

Add to top imports:
```python
from engine.sweeper import generate_experiment_grid, run_sweep
from utils.config import dump_commented_config
```

Add before `if __name__ == "__main__":`:
```python
@app.command()
def sweep(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to sweep YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print grid without running"),
):
    """Run a hyperparameter sweep defined in a YAML config."""
    setup_logging("INFO")
    config = load_config(str(config_path))
    grid = generate_experiment_grid(config)

    if dry_run:
        typer.echo(f"Generated {len(grid)} experiments:")
        for exp in grid:
            typer.echo(f"  {exp['combo_name']} [fold={exp['fold'] + 1}]")
        raise typer.Exit()

    run_sweep(grid, config)


@app.command(name="config")
def config_cmd(
    dump: bool = typer.Option(False, "--dump", help="Write commented default config to stdout"),
):
    """Generate and inspect configuration."""
    if dump:
        import sys, tempfile
        fd, path = tempfile.mkstemp(suffix=".yaml")
        dump_commented_config(path)
        with open(path, "r") as f:
            sys.stdout.write(f.read())
    else:
        cfg = load_config()
        typer.echo(json.dumps(cfg, indent=2, default=str))


@app.command()
def aggregate(
    results_dir: Path = typer.Argument(..., help="Path to experiment results directory"),
):
    """Compute mean and std across folds for all architectures."""
    import pandas as pd

    summary_csv = results_dir / "summary.csv"
    if not summary_csv.exists():
        typer.echo(f"No summary.csv found in {results_dir}", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(summary_csv)
    metric_cols = [c for c in df.columns if c not in ("detector", "architecture", "fold")]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    group_cols = ["detector", "architecture"]

    mean_df = df.groupby(group_cols)[metric_cols].mean().add_suffix("_mean")
    std_df = df.groupby(group_cols)[metric_cols].std().add_suffix("_std")
    result = pd.concat([mean_df, std_df], axis=1).reset_index()

    out_path = results_dir / "summary_aggregated.csv"
    result.to_csv(out_path, index=False)
    typer.echo(f"Aggregated results saved to {out_path}")
    typer.echo(result.to_string())
```

- [ ] **Step 2: Verify all commands**

```bash
python cli.py --help
python cli.py sweep --help
python cli.py config --help
python cli.py aggregate --help
```

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat: CLI sweep, config dump, and aggregate commands"
```

---

### Task 14: Default config and example sweep

**Files:**
- Modify: `configs/default.yaml` (create/overwrite via dump)
- Create: `configs/sweeps/baseline.yaml`

**Steps:**

- [ ] **Step 1: Generate `configs/default.yaml`**

```bash
python cli.py config --dump > configs/default.yaml
```

Verify all three detectors appear:
```bash
grep "^\s\s\w" configs/default.yaml
```

- [ ] **Step 2: Write `configs/sweeps/baseline.yaml`**

```yaml
experiment: baseline_v2

folds:
  n_folds: 5
  val_ratio: 0.2

detectors:
  yolov8:
    architectures: [yolov8n, yolov8s]
    hparams:
      lr: [0.0001, 0.001]
      epochs: 50
      batch_size: 16
  faster_rcnn:
    architectures: [resnet50]
    hparams:
      lr: [0.0001, 0.001]
      epochs: 30
      batch_size: 8
  detr: ~
```

- [ ] **Step 3: Verify sweep dry-run**

```bash
python cli.py sweep --config configs/sweeps/baseline.yaml --dry-run
```

Expected: lists ~20 experiments (2 YOLO archs x 2 LRs x 5 folds + 1 FRCNN arch x 2 LRs x 5 folds)

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "feat: default config and example sweep"
```

---

### Task 15: Integration tests

**Files:**
- Create: `tests/test_folds.py`
- Create: `tests/test_config.py`

**Steps:**

- [ ] **Step 1: Write `tests/test_folds.py`**

```python
"""Test fold splitting logic."""
from utils.folds import split_folds


def test_split_folds_basic():
    images = [{"id": i} for i in range(100)]
    annotations = []
    folds = split_folds(images, annotations, n_folds=5, val_ratio=0.2, seed=42)

    assert len(folds) == 5
    for f in folds:
        assert "train" in f and "val" in f and "test" in f
        assert len(f["train"]) > 0
        assert len(f["val"]) > 0
        assert len(f["test"]) > 0

    all_test_ids = set()
    for f in folds:
        all_test_ids.update(f["test"])
    assert all_test_ids == set(range(100))


def test_split_folds_reproducible():
    images = [{"id": i} for i in range(20)]
    folds1 = split_folds(images, [], n_folds=3, val_ratio=0.3, seed=42)
    folds2 = split_folds(images, [], n_folds=3, val_ratio=0.3, seed=42)
    for f1, f2 in zip(folds1, folds2):
        assert f1["test"] == f2["test"]
```

- [ ] **Step 2: Write `tests/test_config.py`**

```python
"""Test config loading and merge."""
import tempfile
from utils.config import load_config


def test_load_defaults():
    config = load_config()
    assert config["experiment"] == "default"
    assert config["folds"]["n_folds"] == 5
    assert "coco_json" in config["dataset"]


def test_load_yaml_override():
    yaml_content = "experiment: test_override\nfolds:\n  n_folds: 3\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)
    assert config["experiment"] == "test_override"
    assert config["folds"]["n_folds"] == 3
    assert config["folds"]["val_ratio"] == 0.2


def test_cli_overrides():
    config = load_config(cli_overrides={"experiment": "cli_test", "seed": 999})
    assert config["experiment"] == "cli_test"
    assert config["seed"] == 999
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: unit tests for fold splitting and config"
```

---

### Task 16: End-to-end smoke test

**Files:**
- None new. Verifies full train→eval→aggregate pipeline.

**Steps:**

- [ ] **Step 1: Smoke test — train 1 epoch Faster R-CNN**

```bash
python cli.py train -d faster_rcnn -a resnet50 --epochs 1 --batch-size 2 --fold 1
```

Expected: creates `results/default/fold_1/resnet50/best.pth` and `logs.csv`

- [ ] **Step 2: Evaluate checkpoint**

```bash
python cli.py eval -d faster_rcnn -w results/default/fold_1/resnet50/best.pth --fold 1
```

Expected: prints mAP metrics, creates `metrics.json` and `summary.csv`

- [ ] **Step 3: Aggregate results**

```bash
python cli.py aggregate results/default/
```

Expected: creates `summary_aggregated.csv`

- [ ] **Step 4: Verify config dump includes all 3 detectors**

```bash
python cli.py config_cmd --dump | grep -c "architectures:"
```

Expected: 3 (one per detector family)

- [ ] **Step 5: Commit any final fixes**

```bash
git add -A
git status
git commit -m "chore: final integration fixes from E2E smoke test" || echo "No changes needed"
```
