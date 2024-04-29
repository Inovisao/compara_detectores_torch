import torch
import json
import os
BATCH_SIZE = 4 # lote de imagens
RESIZE_TO = 640 # tamanho da imagem
NUM_EPOCHS = 10 # Numero de epocas
NUM_WORKERS = 4 # Paciencia
LR = 0.001 # Taxa de aprendizagem

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ROOT_DATA_DIR = os.path.join('..','dataset','all')
# training images and XML files directory
TRAIN_DIR = os.path.join(ROOT_DATA_DIR,'faster','train')
# validation images and XML files directory
VALID_DIR = os.path.join(ROOT_DATA_DIR,'faster','valid')

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
OUT_DIR = 'FasterRCNN'
