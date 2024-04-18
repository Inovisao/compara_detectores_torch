import os
import numpy as np
import subprocess

# yolov8, fasterrcnn, sabl
MODELS = ['MMdetections/sabl-faster-rcnn_r50_fpn_1x_coco']

ROOT_DATA_DIR = os.path.join('..', 'dataset','all')
DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
DOBRAS = len(os.listdir(DIR_PATH)) // 3

for model in MODELS:
    for f in np.arange(1,DOBRAS+1):

        fold = 'fold_'+str(f)
        fold_dir = os.path.join('model_checkpoints', fold)

        if model == 'YOLOV8':
            from Detectors.YOLOV8.RunYOLOV8 import runYOLOV8
            runYOLOV8(fold,fold_dir,ROOT_DATA_DIR)
            continue
    
        elif model == 'FasterCNN':
            from Detectors.FasterCNN.RunFaster import runFaster
            runFaster(fold,fold_dir,ROOT_DATA_DIR)
            continue

        elif model[0:12] == 'MMdetections':
            from Detectors.MMdetection.RunMMdetecion import runMMdetection
            runMMdetection(model,fold_dir)
