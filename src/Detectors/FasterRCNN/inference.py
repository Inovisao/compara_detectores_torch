import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import json

from Detectors.FasterRCNN.model import create_model

from Detectors.FasterRCNN.config import (
    NUM_CLASSES, DEVICE, CLASSES
)
def resultFaster(fold,image):
    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load the best model and trained weights
    model = create_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(f'model_checkpoints/{fold}/FasterRCNN/best_model.pth', map_location=DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    detection_threshold = 0.8


    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        results = model(image.to(DEVICE))

    # load all detection to CPU for further operations
    results = [{k: v.to('cpu') for k, v in t.items()} for t in results]
    # carry further only if there are detected boxes

    lista = [] 
    classes_usadas = []
    JsonData = '../dataset/all/train/_annotations.coco.json'
    with open(JsonData) as f:
        data = json.load(f)
    ann_ids = []
    for anotation in data["annotations"]:
        if anotation["category_id"] not in ann_ids:
            ann_ids.append(anotation["category_id"])
    for j in ann_ids:
        lista1 = []
        classe = j - 1
        for result in results:
            for i in range(len(result['labels'])):
                if result['labels'][i]-1 == classe:
                    score = result['scores'][i]
                    bbox = result['boxes'][i]
                    classes_usadas.append(classe)
                    lista1.append([bbox[0],bbox[1],bbox[2],bbox[3],score])
        lista.append(np.array(lista1,dtype='float32'))
    return lista
