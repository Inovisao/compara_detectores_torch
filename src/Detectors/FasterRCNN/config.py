import torch
import json
import os

BATCH_SIZE = 1
RESIZE_TO = 512
NUM_EPOCHS = 30
NUM_WORKERS = 0
LR = 0.0001
PATIENCE = 5

def get_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.zeros(1).to(torch.device('cuda'))
        return torch.device('cuda')
    except Exception:
        return torch.device('cpu')


DEVICE = get_device()
ROOT_DATA_DIR = os.path.join('..', 'dataset', 'all')

TRAIN_DIR = os.path.join(ROOT_DATA_DIR, 'Faster', 'train')
VALID_DIR = os.path.join(ROOT_DATA_DIR, 'Faster', 'val')

CLASSES = ['Background']
with open(os.path.join(ROOT_DATA_DIR, 'train', '_annotations.coco.json'), 'r') as f:
    data = json.load(f)

ann_ids = []
for annotation in data["annotations"]:
    if annotation["category_id"] not in ann_ids:
        ann_ids.append(annotation["category_id"])

for category in data["categories"]:
    if category["id"] in ann_ids:
        CLASSES.append(category["name"])

NUM_CLASSES = len(CLASSES)

OUT_DIR = './Faster'