import json
import os
from pathlib import Path
from typing import Optional

import torch

BATCH_SIZE = int(os.getenv("FASTER_BATCH", "16"))
RESIZE_TO = int(os.getenv("FASTER_RESIZE_TO", "640"))
NUM_EPOCHS = int(os.getenv("FASTER_EPOCHS", "1000"))
NUM_WORKERS = int(os.getenv("FASTER_WORKERS", "8"))
PATIENCE = int(os.getenv("FASTER_PATIENCE", "50"))
LR = float(os.getenv("FASTER_LR", "0.005"))
OPTIMIZER = "SGD"
MOMENTUM = float(os.getenv("FASTER_MOMENTUM", "0.9"))
WEIGHT_DECAY = float(os.getenv("FASTER_WEIGHT_DECAY", "0.0005"))
USE_AMP = os.getenv("FASTER_USE_AMP", "false").strip().lower() in {"1", "true", "yes"}
USE_COMPILE = os.getenv("FASTER_USE_COMPILE", "false").strip().lower() in {"1", "true", "yes"}
CLIP_GRAD_NORM = float(os.getenv("FASTER_CLIP_GRAD_NORM", "0.0"))

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ROOT_DATA_DIR: Optional[str] = None
TRAIN_DIR: Optional[str] = None
TRAIN_ANN_PATH: Optional[str] = None
VALID_DIR: Optional[str] = None
VAL_ANN_PATH: Optional[str] = None
CLASSES = ['Background']
NUM_CLASSES = 1

# location to save model and plots — override via FASTER_OUT_DIR env var
OUT_DIR = os.getenv('FASTER_OUT_DIR', './Faster')

def _load_classes(annotation_paths) -> None:
    global CLASSES, NUM_CLASSES

    categories_by_id = {}
    present_ids = set()
    for ann_path in annotation_paths:
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for category in data.get('categories', []):
            categories_by_id[int(category['id'])] = category['name']
        present_ids.update(int(ann['category_id']) for ann in data.get('annotations', []))

    if not categories_by_id:
        CLASSES = ['Background']
        NUM_CLASSES = 1
        return

    ordered_ids = sorted(present_ids) if present_ids else sorted(categories_by_id.keys())
    CLASSES = ['Background'] + [categories_by_id[idx] for idx in ordered_ids]
    NUM_CLASSES = len(CLASSES)


def configure_dataset(train_dir: Path, train_ann: Path, val_dir: Path, val_ann: Path) -> None:
    global ROOT_DATA_DIR, TRAIN_DIR, TRAIN_ANN_PATH, VALID_DIR, VAL_ANN_PATH

    train_dir = Path(train_dir).resolve()
    val_dir = Path(val_dir).resolve()
    train_ann = Path(train_ann).resolve()
    val_ann = Path(val_ann).resolve()

    for path in (train_dir, val_dir):
        if not path.exists():
            raise FileNotFoundError(f"Image directory not found for FasterRCNN: {path}")
    for path in (train_ann, val_ann):
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found for FasterRCNN: {path}")

    ROOT_DATA_DIR = str(train_dir.parent)
    TRAIN_DIR = str(train_dir)
    TRAIN_ANN_PATH = str(train_ann)
    VALID_DIR = str(val_dir)
    VAL_ANN_PATH = str(val_ann)

    _load_classes([train_ann, val_ann])


def _try_init_from_env() -> None:
    train_dir = os.getenv('FASTER_TRAIN_DIR')
    train_ann = os.getenv('FASTER_TRAIN_ANN')
    val_dir = os.getenv('FASTER_VAL_DIR')
    val_ann = os.getenv('FASTER_VAL_ANN')

    if all([train_dir, train_ann, val_dir, val_ann]):
        try:
            configure_dataset(train_dir, train_ann, val_dir, val_ann)
            return
        except FileNotFoundError:
            pass


_try_init_from_env()


def get_training_params() -> dict:
    return {
        "batch_size": BATCH_SIZE,
        "resize_to": RESIZE_TO,
        "num_epochs": NUM_EPOCHS,
        "num_workers": NUM_WORKERS,
        "patience": PATIENCE,
        "learning_rate": LR,
        "optimizer": OPTIMIZER,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "clip_grad_norm": CLIP_GRAD_NORM,
        "device": str(DEVICE),
        "out_dir": OUT_DIR,
        "root_data_dir": ROOT_DATA_DIR,
        "train_dir": TRAIN_DIR,
        "train_annotations": TRAIN_ANN_PATH,
        "valid_dir": VALID_DIR,
        "val_annotations": VAL_ANN_PATH,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
    }

__all__ = [
    "configure_dataset",
    "get_training_params",
    "ROOT_DATA_DIR",
    "TRAIN_DIR",
    "TRAIN_ANN_PATH",
    "VALID_DIR",
    "VAL_ANN_PATH",
    "CLASSES",
    "NUM_CLASSES",
    "BATCH_SIZE",
    "RESIZE_TO",
    "NUM_EPOCHS",
    "NUM_WORKERS",
    "LR",
    "OPTIMIZER",
    "MOMENTUM",
    "WEIGHT_DECAY",
    "PATIENCE",
    "DEVICE",
    "OUT_DIR",
]
