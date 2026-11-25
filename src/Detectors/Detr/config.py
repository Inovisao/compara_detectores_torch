import torch
import json
import os
import sys
from pathlib import Path

# train.py/train_detector.py rodam como script (subprocesso) e só enxergam a própria pasta;
# src/ precisa estar no path para achar hyperparameters.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperparameters import param  # noqa: E402

# Valores em src/hyperparameters.py (por dataset); DETR_* sobrescreve.
BATCH_SIZE = param("Detr", "batch_size", "DETR_BATCH")  # lote de imagens
RESIZE_TO = param("Detr", "resize_to", "DETR_RESIZE")  # tamanho da imagem
NUM_EPOCHS = param("Detr", "epochs", "DETR_EPOCHS")  # numero de epocas
NUM_WORKERS = param("Detr", "num_workers", "DETR_NUM_WORKERS")
LR = param("Detr", "learning_rate", "DETR_LR")  # taxa de aprendizagem
LR_BACKBONE = param("Detr", "lr_backbone", "DETR_LR_BACKBONE")
WEIGHT_DECAY = param("Detr", "weight_decay", "DETR_WEIGHT_DECAY")
DATA_PATH = os.path.join('..','dataset','all','dataDetr.yaml')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ROOT_DATA_DIR = os.path.join('..','dataset','all')
# training images and XML files directory
TRAIN_DIR = os.path.join(ROOT_DATA_DIR,'detr','train')
# validation images and XML files directory
VALID_DIR = os.path.join(ROOT_DATA_DIR,'detr','valid')

# classes: 0 index is reserved for background
CLASSES = [
    '__background__'
]
with open(os.path.join(ROOT_DATA_DIR, 'train', '_annotations.coco.json'), 'r') as f:
    data = json.load(f)

ann_ids = []
for anotation in data["annotations"]:
    if anotation["category_id"] not in ann_ids:
        ann_ids.append(anotation["category_id"])

for category in data["categories"]:
    if category["id"] in ann_ids:
        CLASSES.append(category["name"],)

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'detr'


def get_training_params() -> dict:
    return {
        "batch_size": BATCH_SIZE,
        "resize_to": RESIZE_TO,
        "num_epochs": NUM_EPOCHS,
        "num_workers": NUM_WORKERS,
        "learning_rate": LR,
        "lr_backbone": LR_BACKBONE,
        "weight_decay": WEIGHT_DECAY,
        "data_path": DATA_PATH,
        "device": str(DEVICE),
        "root_data_dir": ROOT_DATA_DIR,
        "train_dir": TRAIN_DIR,
        "valid_dir": VALID_DIR,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "visualize_transformed_images": VISUALIZE_TRANSFORMED_IMAGES,
        "out_dir": OUT_DIR,
    }
