import torch
import json
import os
BATCH_SIZE = 8 # lote de imagens
RESIZE_TO = 640 # tamanho da imagem
NUM_EPOCHS = 10 # Numero de epocas
NUM_WORKERS = 5 # Paciencia
LR = 0.0001 # Taxa de aprendizagem
PATIENCE = 5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ROOT_DATA_DIR = os.path.join('..','dataset','all')

TRAIN_DIR = os.path.join(ROOT_DATA_DIR,'Faster','train')

VALID_DIR = os.path.join(ROOT_DATA_DIR,'Faster','val')

# classes: 0 index is reserved for background
CLASSES = [
    'Background'
]
with open(os.path.join(ROOT_DATA_DIR,'train', '_annotations.coco.json'), 'r') as f:
    data = json.load(f)

ann_ids = []
for anotation in data["annotations"]:
    if anotation["category_id"] not in ann_ids:
        ann_ids.append(anotation["category_id"])

for category in data["categories"]:
    if category["id"] in ann_ids:
        CLASSES.append(category["name"],)

NUM_CLASSES = len(CLASSES)

# location to save model and plots
OUT_DIR = './Faster'