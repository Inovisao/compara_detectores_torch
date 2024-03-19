import os
import numpy as np
import shutil
import subprocess
from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from Detectors.FasterCNN.GeraDobras import convert_coco_to_voc

MODELS = ['FasterCNN']
dir_path = r'dataset/filesJSON'
DOBRAS = len(os.listdir(dir_path)) // 3
for model in MODELS:
    for f in np.arange(1,DOBRAS+1):
        fold = 'fold_'+str(f)
        if model == 'YOLOV8':
                CriarLabelsYOLOV8(fold)
                treino = 'Detectors/YOLOV8/TreinoYOLOV8.sh'
                subprocess.run([treino]) # Roda o bash para treino
                if not os.path.exists("dataset/"+fold):
                    os.makedirs("dataset/"+fold)
                if os.path.exists(f"dataset/{fold}/YOLOV8"):  
                    shutil.rmtree(f"dataset/{fold}/YOLOV8") 
                os.rename("./YOLOV8", f"dataset/{fold}/YOLOV8")
                shutil.rmtree('dataset/YOLO')
        if model == 'FasterCNN':
            if not os.path.exists("dataset/"+fold):
                os.makedirs("dataset/"+fold)
            if not os.path.exists('FasterCNN'):
                os.makedirs('FasterCNN')
            if os.path.exists(f"dataset/{fold}/FasterCNN"):  
                shutil.rmtree(f"dataset/{fold}/FasterCNN") 
            convert_coco_to_voc(fold)
            treino = 'Detectors/FasterCNN/TreinoFaster.sh'
            subprocess.run([treino]) # Roda o bash para treino
            os.rename("./FasterCNN", f"dataset/{fold}/FasterCNN")
            shutil.rmtree('dataset/Faster')
            shutil.rmtree('runs')