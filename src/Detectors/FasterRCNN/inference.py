import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from Detectors.FasterRCNN.config import (DEVICE,NUM_CLASSES,CLASSES)
# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def prepare_image(frame):
    # Load the image using OpenCV (in BGR format)
    # Convert BGR to RGB (since OpenCV loads images in BGR by default)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_rgb).float() / 255.0  # Normalize the image
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Change to CxHxW and add batch dimension
    return image_tensor.to(DEVICE)

def xyxy_to_xywh(boxes):
    """
    Converte caixas delimitadoras no formato xyxy para xywh.

    :param boxes: Lista ou array de caixas no formato [x_min, y_min, x_max, y_max, conf, class].
    :return: Lista de caixas no formato [x, y, w, h, conf, class].
    """
    coco_boxes = []

    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([x_min, y_min, w, h, conf, cls])
    return coco_boxes

# Load the trained model

class ResultFaster:
    def resultFaster(frame,modelName,LIMIAR_THRESHOLD):
        model = get_model(NUM_CLASSES)
        model.load_state_dict(torch.load(modelName))
        model.to(DEVICE)
        model.eval()  # Set the model to evaluation mode

        # Load the unseen image
        image_tensor = prepare_image(frame)

        with torch.no_grad():  # Disable gradient computation for inference
            prediction = model(image_tensor)
        bbox = prediction[0]['boxes'].cpu().tolist()
        labels = prediction[0]['labels'].cpu().tolist()
        scores = prediction[0]['scores'].cpu().tolist()
        faster_box = []
        for i,box in enumerate(bbox):
            if scores[i] > LIMIAR_THRESHOLD:
                faster_box.append([int(box[0]),int(box[1]),int(box[2]),int(box[3]),int(labels[i]),scores[i]])
        #box = prediction['box']

        coco_boxes = xyxy_to_xywh(faster_box)

        return coco_boxes