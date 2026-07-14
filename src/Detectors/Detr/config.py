import torch
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
BATCH_SIZE = 16
RESIZE_TO = 640
NUM_EPOCHS = 1000
NUM_WORKERS = 8
PATIENCE = 50
LR = 0.0001
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.0001
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ROOT_DATA_DIR = os.environ.get(
    'DATASET_ROOT',
    str(_PROJECT_ROOT / 'dataset' / 'all')
)
DATA_PATH = os.path.join(ROOT_DATA_DIR, 'dataDetr.yaml')
# training images and XML files directory
TRAIN_DIR = os.path.join(ROOT_DATA_DIR,'detr','train')
# validation images and XML files directory
VALID_DIR = os.path.join(ROOT_DATA_DIR,'detr','valid')

# classes: 0 index is reserved for background
CLASSES = [
    '__background__'
]
_ann_candidates = [
    os.path.join(ROOT_DATA_DIR, 'train', '_annotations.coco.json'),
    os.path.join(ROOT_DATA_DIR, '_annotations.coco.json'),
]
_ann_path = next((p for p in _ann_candidates if os.path.exists(p)), None)
if _ann_path is None:
    raise FileNotFoundError(
        f"_annotations.coco.json not found. Tried: {_ann_candidates}"
    )
with open(_ann_path, 'r') as f:
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
        "patience": PATIENCE,
        "learning_rate": LR,
        "optimizer": OPTIMIZER,
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
