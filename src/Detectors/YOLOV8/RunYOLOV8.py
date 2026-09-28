import os
import sys
from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from Detectors.YOLOV8.TrocaSettings import Settings
import subprocess
import shutil

def runYOLOV8(fold,fold_dir,ROOT_DATA_DIR):

    Settings()
    CriarLabelsYOLOV8(fold) # Função para criar as labels do treino da YOLOV8
    treino = os.path.join('Detectors', 'YOLOV8', 'config.py')
    # Remove se over Resultados na pasta model_checkpoints
    if os.path.exists(os.path.join(fold_dir, 'YOLOV8')):  
        shutil.rmtree(os.path.join(fold_dir, "YOLOV8")) 
    subprocess.run([sys.executable, treino], check=True)

    weights_path = os.path.join('YOLOV8', 'train', 'weights', 'best.pt')
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"O treino YOLO terminou sem gerar o checkpoint esperado: {weights_path}. "
            "Verifique a saída do treinamento e a configuração do dataset."
        )

    os.makedirs(fold_dir, exist_ok=True)

    os.rename('YOLOV8', os.path.join(fold_dir, 'YOLOV8'))
    shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'YOLO'))# Remove as labels Geradas
