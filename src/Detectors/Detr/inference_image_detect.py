import torch
import cv2
import numpy as np
import argparse
import os
import time
import torchinfo
from vision_transformers.detection.detr.model import DETRModel
from utils.detection.detr.general import (
    set_infer_dir,
    load_weights
)
from utils.detection.detr.transforms import infer_transforms, resize
from utils.detection.detr.annotations import (
    convert_detections,
    inference_annotations, 
)
from utils.detection.detr.viz_attention import visualize_attention

np.random.seed(2023)

def xyxy_to_xywh(boxes):
    """
    Converte caixas delimitadoras do formato [x_min, y_min, x_max, y_max, conf, class] para [x, y, w, h, conf, class].
    """
    coco_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([x_min, y_min, w, h, conf, cls])
    return coco_boxes

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', 
        '--weights',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '--model', 
        default='detr_resnet50',
        help='name of the model'
    )
    parser.add_argument(
        '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '--track',
        action='store_true'
    )

    parser.add_argument(
        '-t', 
        '--threshold',
        type=float,
        default=0.5,
        help='confidence threshold for visualization'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='filter classes by visualization, --classes 1 2 3'
    )
    parser.add_argument(
        '--hide-labels',
        dest='hide_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    
    args = parser.parse_args()
    return args

def main(args,orig_image,LIMIAR_THRESHOLD):
    NUM_CLASSES = None
    CLASSES = None
    data_configs = None

    DEVICE = args.device
    model, CLASSES, data_path = load_weights(
        args, DEVICE, DETRModel, data_configs, NUM_CLASSES, CLASSES
    )
    _ = model.to(DEVICE).eval()

    NUM_CLASSES = len(CLASSES)

    frame_height, frame_width, _ = orig_image.shape
    
    image_resized = resize(orig_image, 640, square=True)
    image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = infer_transforms(image)
    input_tensor = torch.tensor(image, dtype=torch.float32)
    input_tensor = torch.permute(input_tensor, (2, 0, 1))
    input_tensor = input_tensor.unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor.to(DEVICE))
    end_time = time.time()
    
    fps = 1 / (end_time - start_time)

    if len(outputs['pred_boxes'][0]) != 0:
        draw_boxes, pred_classes, scores = convert_detections(
            outputs, 
            args.threshold,
            CLASSES,
            orig_image,
            args 
        )

    class_dict = {}
    class_list = []
    for i,Cls in enumerate(CLASSES):
        class_dict[i] = Cls

    for pred in pred_classes:
        for key in class_dict:
            if pred == class_dict[key]:
                class_list.append(int(key)-1)

    inferencebox = [] 
    for i,bbox in enumerate(draw_boxes):
        if scores[i] > LIMIAR_THRESHOLD:
            inferencebox.append([int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),int(class_list[i])+1,scores[i]])

    coco_box = xyxy_to_xywh(inferencebox)

    return coco_box

def resultDetr(fold,image,LIMIAR_THRESHOLD):
    weightsPath = os.path.join('model_checkpoints',fold,'Detr','training','best_model.pth')
    args = parse_opt()
    args.weights = weightsPath
    lista = main(args,image,LIMIAR_THRESHOLD)
    return lista
