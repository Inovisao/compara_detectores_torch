import os
import shutil
import subprocess
from pathlib import Path

from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
from Detectors.YOLOV8.TrocaSettings import Settings


def runYOLOV8(fold, fold_dir, ROOT_DATA_DIR):
    Settings()
    CriarLabelsYOLOV8(fold)  # Função para criar as labels do treino da YOLOV8

    src_dir = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    treino_script = Path(__file__).resolve().parent / 'TreinoYOLOV8.sh'
    output_dir = os.path.join(fold_dir, 'YOLOV8')

    # Remove se houver resultados antigos na pasta model_checkpoints
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    temp_output_dir = os.path.join(repo_root, 'YOLOV8')
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    try:
        subprocess.run(['bash', str(treino_script)], cwd=str(src_dir), check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'Treino YOLOV8 falhou com código {exc.returncode}') from exc

    if not os.path.exists(temp_output_dir):
        raise FileNotFoundError(f'Pasta de saída do treino não encontrada: {temp_output_dir}')

    # Verifica que a pasta Fold_num existe
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    os.rename(temp_output_dir, output_dir)  # Move os dados dos treinos para model_checkpoints
    shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'YOLO'), ignore_errors=True)  # Remove as labels Geradas
