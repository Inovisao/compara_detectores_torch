import json
import os
from pathlib import Path
from typing import Optional

import torch

BATCH_SIZE = 4 # lote de imagens
RESIZE_TO = 640 # tamanho da imagem
NUM_EPOCHS = 1000 # Numero de epocas
NUM_WORKERS = 4 # Paciencia
LR = 0.0001 # Taxa de aprendizagem
PATIENCE = 150

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ROOT_DATA_DIR: Optional[str] = None
TRAIN_DIR: Optional[str] = None
TRAIN_ANN_PATH: Optional[str] = None
VALID_DIR: Optional[str] = None
VAL_ANN_PATH: Optional[str] = None
CLASSES = ['Background']
NUM_CLASSES = 1

# location to save model and plots
OUT_DIR = './Faster'

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

    fold = os.getenv('FASTER_FOLD')
    if not fold:
        return

    tile_root = PROJECT_ROOT / 'dataset' / 'tiles'
    fold_root = tile_root / fold
    train_dir = fold_root / 'train'
    val_dir = fold_root / 'val'
    train_ann = train_dir / '_annotations.coco.json'
    val_ann = val_dir / '_annotations.coco.json'

    try:
        configure_dataset(train_dir, train_ann, val_dir, val_ann)
    except FileNotFoundError:
        pass


_try_init_from_env()

__all__ = [
    "configure_dataset",
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
    "PATIENCE",
    "DEVICE",
    "OUT_DIR",
]
