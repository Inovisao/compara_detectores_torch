import os
import numpy as np
import shutil
import subprocess
from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from Detectors.FasterCNN.GeraDobras import convert_coco_to_voc

# yolov8, fasterrcnn, sabl
MODELS = ['FasterCNN','YOLOV8']

ROOT_DATA_DIR = os.path.join('..', 'dataset')
DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
DOBRAS = len(os.listdir(DIR_PATH)) // 3

for model in MODELS:
    for f in np.arange(1,DOBRAS+1):
        fold = 'fold_'+str(f)
        fold_dir = os.path.join(ROOT_DATA_DIR, fold)
        
        if model == 'YOLOV8':
                CriarLabelsYOLOV8(fold)
                treino = os.path.join('Detectors', 'YOLOV8', 'TreinoYOLOV8.sh') # 'Detectors/YOLOV8/TreinoYOLOV8.sh'
                subprocess.run([treino]) # Roda o bash para treino

                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)

                if os.path.exists(os.path.join(fold_dir, 'YOLOV8')):  
                    shutil.rmtree(os.path.join(fold_dir, "YOLOV8")) 
                os.rename('YOLOV8', os.path.join(fold_dir, 'YOLOV8'))
                shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'YOLO'))
                continue

        elif model == 'FasterCNN':
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            if not os.path.exists('FasterCNN'):
                os.makedirs('FasterCNN')
            if os.path.exists(os.path.join(fold_dir,"FasterCNN")):  
                shutil.rmtree(os.path.join(fold_dir,"FasterCNN")) 
            convert_coco_to_voc(fold)
            treino = os.path.join('Detectors','FasterCNN','TreinoFaster.sh')
            subprocess.run([treino]) # Roda o bash para treino
            os.rename("./FasterCNN", os.path.join(fold_dir,"FasterCNN"))
            shutil.rmtree(os.path.join(ROOT_DATA_DIR,'faster'))
            shutil.rmtree('runs')
            continue
        elif model[0:12] == 'MMdetections':
            print(model)