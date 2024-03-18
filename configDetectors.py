import os
import numpy as np


from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8

MODELS = ['YOLOV8']
dir_path = r'dataset/filesJSON'
DOBRAS = len(os.listdir(dir_path)) // 3
print(DOBRAS)
for model in MODELS:
    if model == 'YOLOV8':
        for f in np.arange(1,DOBRAS+1):
            fold = 'fold_'+str(f)
            CriarLabelsYOLOV8(fold)
            print(model)