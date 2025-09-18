import os
import shutil
import subprocess

from Detectors.YOLOV5_TPH.GeraLabels import CriarLabelsYOLOV5TPH

ROOT_DATA_DIR = os.path.join('..', 'dataset', 'all')


def runYOLOV5TPH(fold, fold_dir, ROOT_DATA_DIR):
    CriarLabelsYOLOV5TPH(fold)
    treino = os.path.join('Detectors', 'YOLOV5_TPH', 'TreinoYOLOV5TPH.sh')

    if os.path.exists(os.path.join(fold_dir, 'YOLOV5_TPH')):
        shutil.rmtree(os.path.join(fold_dir, 'YOLOV5_TPH'))

    if os.path.exists('YOLOV5_TPH'):
        shutil.rmtree('YOLOV5_TPH')

    subprocess.run([treino], check=True)

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    if os.path.exists('YOLOV5_TPH'):
        os.rename('YOLOV5_TPH', os.path.join(fold_dir, 'YOLOV5_TPH'))
    else:
        raise FileNotFoundError(
            "YOLOV5_TPH training output not found. Ensure the tph-yolov5 repository is cloned and training succeeded."
        )

    yolo_tph_dir = os.path.join(ROOT_DATA_DIR, 'YOLOV5_TPH')
    if os.path.exists(yolo_tph_dir):
        shutil.rmtree(yolo_tph_dir)
