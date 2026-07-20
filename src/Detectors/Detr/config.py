import torch
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
BATCH_SIZE = int(os.getenv("DETR_BATCH", "16"))
RESIZE_TO = int(os.getenv("DETR_RESIZE_TO", "640"))
NUM_EPOCHS = int(os.getenv("DETR_EPOCHS", "1000"))
NUM_WORKERS = int(os.getenv("DETR_WORKERS", "8"))
PATIENCE = int(os.getenv("DETR_PATIENCE", "50"))
LR = float(os.getenv("DETR_LR", "0.0001"))
OPTIMIZER = "AdamW"
WEIGHT_DECAY = float(os.getenv("DETR_WEIGHT_DECAY", "0.0001"))
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

def _load_classes_from_contract(root_data_dir: str) -> list[str]:
    files_json_dir = Path(root_data_dir) / 'filesJSON'
    if not files_json_dir.exists():
        raise FileNotFoundError(
            f"Expected DATASET_ROOT/filesJSON for DETR class loading: {files_json_dir}"
        )

    json_paths = sorted(files_json_dir.glob('fold_*_train.json'))
    if not json_paths:
        json_paths = sorted(files_json_dir.glob('fold_*_*.json'))
    if not json_paths:
        raise FileNotFoundError(f"No fold COCO JSONs found in {files_json_dir}")

    with json_paths[0].open('r', encoding='utf-8') as f:
        data = json.load(f)

    categories_by_id = {
        int(category['id']): category['name']
        for category in data.get('categories', [])
    }
    present_ids = sorted({
        int(annotation['category_id'])
        for annotation in data.get('annotations', [])
    })
    selected_ids = present_ids or sorted(categories_by_id)
    if not selected_ids:
        raise ValueError(f"No categories found in {json_paths[0]}")

    return ['__background__'] + [categories_by_id[idx] for idx in selected_ids]


# classes: 0 index is reserved for background
CLASSES = _load_classes_from_contract(ROOT_DATA_DIR)

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
