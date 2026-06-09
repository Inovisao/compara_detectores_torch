import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from Detectors.FasterRCNN import config as faster_config
# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes=None):
    # Load pre-trained Faster R-CNN
    if num_classes is None:
        num_classes = faster_config.NUM_CLASSES
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def prepare_image(frame, device):
    # Load the image using OpenCV (in BGR format)
    # Convert BGR to RGB (since OpenCV loads images in BGR by default)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_rgb).float() / 255.0  # Normalize the image
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Change to CxHxW and add batch dimension
    return image_tensor.to(device)

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

_model_cache: dict = {}


class ResultFaster:
    def resultFaster(frame, modelName, LIMIAR_THRESHOLD):
        device = faster_config.DEVICE
        num_classes = faster_config.NUM_CLASSES

        if modelName not in _model_cache:
            model = get_model(num_classes)
            state_dict = torch.load(modelName, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            _model_cache[modelName] = model
        model = _model_cache[modelName]

        # Resize para mesma escala do treino (RESIZE_TO=640)
        resize_to = faster_config.RESIZE_TO
        orig_h, orig_w = frame.shape[:2]
        scale = resize_to / max(orig_h, orig_w)
        if scale != 1.0:
            frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))

        image_tensor = prepare_image(frame, device)

        with torch.no_grad():
            prediction = model(image_tensor)

        bbox   = prediction[0]['boxes'].cpu().tolist()
        labels = prediction[0]['labels'].cpu().tolist()
        scores = prediction[0]['scores'].cpu().tolist()

        faster_box = []
        for i, box in enumerate(bbox):
            if scores[i] > LIMIAR_THRESHOLD:
                # Reescala coordenadas de volta para o espaço original
                x1 = int(box[0] / scale)
                y1 = int(box[1] / scale)
                x2 = int(box[2] / scale)
                y2 = int(box[3] / scale)
                faster_box.append([x1, y1, x2, y2, int(labels[i]), scores[i]])

        return xyxy_to_xywh(faster_box)
