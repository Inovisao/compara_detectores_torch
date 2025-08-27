import os
from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from Detectors.YOLOV8.TrocaSettings import Settings
import subprocess
import shutil

def runYOLOV8(fold,fold_dir,ROOT_DATA_DIR):

    Settings()
    CriarLabelsYOLOV8(fold) # Função para criar as labels do treino da YOLOV8
    treino = os.path.join('Detectors', 'YOLOV8', 'TreinoYOLOV8.sh') # 'Detectors/YOLOV8/TreinoYOLOV8.sh'
    # Remove se over Resultados na pasta model_checkpoints
    if os.path.exists(os.path.join(fold_dir, 'YOLOV8')):  
        shutil.rmtree(os.path.join(fold_dir, "YOLOV8")) 
    subprocess.run([treino]) # Roda o bash para treino
    # Verifica que a pasta Fold_num existe
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    os.rename('YOLOV8', os.path.join(fold_dir, 'YOLOV8'))# Move os dados Dos treinos para model_checkpoints
    shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'YOLO'))# Remove as labels Geradas
