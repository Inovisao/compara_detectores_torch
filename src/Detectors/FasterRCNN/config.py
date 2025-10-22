import torch
import json
import os
from pathlib import Path
from typing import Optional
BATCH_SIZE = 4 # lote de imagens
RESIZE_TO = 640 # tamanho da imagem
NUM_EPOCHS = 10 # Numero de epocas
NUM_WORKERS = 3 # Paciencia
LR = 0.0001 # Taxa de aprendizagem
PATIENCE = 30

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_TILE_ROOT = PROJECT_ROOT / 'dataset' / 'tiles'

ROOT_DATA_DIR: Optional[str] = None
TRAIN_DIR: Optional[str] = None
VALID_DIR: Optional[str] = None
CLASSES = ['Background']
NUM_CLASSES = 1

# location to save model and plots
OUT_DIR = './Faster'

def init_dataset(fold: str) -> None:
    global ROOT_DATA_DIR, TRAIN_DIR, VALID_DIR, CLASSES, NUM_CLASSES

    fold_root = DATASET_TILE_ROOT / fold / 'Faster'
    if not fold_root.exists():
        raise FileNotFoundError(f"Faster dataset not prepared for fold {fold}: {fold_root}")

    ROOT_DATA_DIR = str(fold_root)
    TRAIN_DIR = os.path.join(ROOT_DATA_DIR, 'train')
    VALID_DIR = os.path.join(ROOT_DATA_DIR, 'val')

    annotations_path = Path(TRAIN_DIR) / '_annotations.coco.json'
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found at {annotations_path}")

    with annotations_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    ann_ids = {ann['category_id'] for ann in data.get('annotations', [])}
    CLASSES = ['Background']
    for category in data.get('categories', []):
        if category['id'] in ann_ids:
            CLASSES.append(category['name'])
    NUM_CLASSES = len(CLASSES)


_fold_env = os.getenv('FASTER_FOLD')
if _fold_env:
    try:
        init_dataset(_fold_env)
    except FileNotFoundError:
        pass
