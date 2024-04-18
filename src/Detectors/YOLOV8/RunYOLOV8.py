import os
from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
import subprocess
import shutil

def runYOLOV8(fold,fold_dir,ROOT_DATA_DIR):
    print(fold)
    input()
    CriarLabelsYOLOV8(fold)
    treino = os.path.join('Detectors', 'YOLOV8', 'TreinoYOLOV8.sh') # 'Detectors/YOLOV8/TreinoYOLOV8.sh'

    subprocess.run([treino]) # Roda o bash para treino

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    if os.path.exists(os.path.join(fold_dir, 'YOLOV8')):  
        shutil.rmtree(os.path.join(fold_dir, "YOLOV8")) 
    os.rename('YOLOV8', os.path.join(fold_dir, 'YOLOV8'))
    shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'YOLO'))