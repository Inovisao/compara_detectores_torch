import torch
import json
import os

ROOT_DATA_DIR = os.path.join('..','dataset','all')
LR = 0.001
BATCH_SIZE = 8 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 5 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Training images and XML files directory.
TRAIN_IMG = '../dataset/all/retinanet/train/image'
TRAIN_ANNOT = '../dataset/all/retinanet/train/annotations'
# Validation images and XML files directory.
VALID_IMG = '../dataset/all/retinanet/valid/image'
VALID_ANNOT = '../dataset/all/retinanet/valid/annotations'

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

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'