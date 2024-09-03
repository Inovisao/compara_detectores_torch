import torch
import cv2
import numpy as np
import argparse
import yaml
import glob
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

def main(args,orig_image):
    NUM_CLASSES = None
    CLASSES = None
    data_configs = None

    DEVICE = args.device
    model, CLASSES, data_path = load_weights(
        args, DEVICE, DETRModel, data_configs, NUM_CLASSES, CLASSES
    )
    _ = model.to(DEVICE).eval()
    try:
        torchinfo.summary(
            model, 
            device=DEVICE, 
            input_size=(1, 3, args.imgsz, args.imgsz),
            row_settings=["var_names"]
        )
    except:
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
    NUM_CLASSES = len(CLASSES)

    # Colors for visualization.
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    RESIZE_TO = 640

    frame_height, frame_width, _ = orig_image.shape
    
    image_resized = resize(orig_image, RESIZE_TO, square=True)
    image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = infer_transforms(image)
    input_tensor = torch.tensor(image, dtype=torch.float32)
    input_tensor = torch.permute(input_tensor, (2, 0, 1))
    input_tensor = input_tensor.unsqueeze(0)
    h, w, _ = orig_image.shape

    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor.to(DEVICE))
    end_time = time.time()
    # Get the current fps.
    fps = 1 / (end_time - start_time)
    # Add `fps` to `total_fps`.
    total_fps += fps
    # Increment frame count.
    frame_count += 1

    if len(outputs['pred_boxes'][0]) != 0:
        draw_boxes, pred_classes, scores = convert_detections(
            outputs, 
            args.threshold,
            CLASSES,
            orig_image,
            args 
        )

    lista = []
    for j,classe in enumerate(CLASSES):
        if j > 0:
            lista1 = []
            for i,bbox in enumerate(draw_boxes):
                if classe == pred_classes[i]:
                    lista1.append([bbox[0],bbox[1],bbox[2],bbox[3],scores[i]])
            lista.append(np.array(lista1,dtype='float32'))
    return lista


def resultDetr(fold,image):
    weightsPath = os.path.join('model_checkpoints',fold,'Detr','training','best_model.pth')
    args = parse_opt()
    args.weights = weightsPath
    lista = main(args,image)
    return lista
