import os
from Detectors.FasterRCNN.geradataset import geredata
import shutil
import subprocess

def runFaster(fold,fold_dir,ROOT_DATA_DIR):
    geredata(fold) # Função para cirar as labels do treino da YOLOV8
    treino = os.path.join('Detectors', 'FasterRCNN', 'TreinoFaster.sh') 
    # Remove se over Resultados na pasta model_checkpoints
    if os.path.exists(os.path.join(fold_dir, 'Faster')):  
        shutil.rmtree(os.path.join(fold_dir, "Faster")) 
    subprocess.run([treino]) # Roda o bash para treino
    # Verifica que a pasta Fold_num existe
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    os.rename('Faster', os.path.join(fold_dir, 'Faster'))# Move os dados Dos treinos para model_checkpoints
    shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'Faster'))# Remove as labels Geradas
