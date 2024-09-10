import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse

from Detectors.Retinanet.model import create_model
from torchvision import transforms as transforms

from Detectors.Retinanet.config import (
    NUM_CLASSES, DEVICE, CLASSES
)

np.random.seed(42)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        default='outputs/best_model.pth',
        help='path to the model weights'
    )
    parser.add_argument(
        '--threshold',
        default=0.25,
        type=float,
        help='detection threshold'
    )
    args = parser.parse_args()
    return args

COLORS = [
    [0, 0, 0],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [255, 255, 255]
]

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)


def main(args,orig_image):
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()


    image_input = infer_transforms(orig_image)

    image_input = torch.unsqueeze(image_input, 0)
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))

    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= args.threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        lista = []

        for j,classe in enumerate(CLASSES):
            if j > 0:
                lista1 = []
                for i,bbox in enumerate(draw_boxes):
                    if classe == pred_classes[i]:
                        lista1.append([bbox[0],bbox[1],bbox[2],bbox[3],scores[i]])
                lista.append(np.array(lista1,dtype='float32'))
        return lista
    
def resultRetinanet(fold,image):
    weightsPath = os.path.join('model_checkpoints',fold,'Retinanet','best_model.pth')
    args = parse_opt()
    args.weights = weightsPath
    lista = main(args,image)
    return lista
            