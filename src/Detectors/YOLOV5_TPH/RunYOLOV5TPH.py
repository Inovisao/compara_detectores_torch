import os
import shutil
import subprocess
from pathlib import Path

from Detectors.YOLOV5_TPH.GeraLabels import CriarLabelsYOLOV5TPH

ROOT_DATA_DIR = os.path.join('..', 'dataset', 'all')


def _resolve_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    project_name = os.getenv("TPH_PROJECT", "YOLOV5_TPH")
    return project_root / project_name


def runYOLOV5TPH(fold, fold_dir, ROOT_DATA_DIR):
    CriarLabelsYOLOV5TPH(fold)
    treino = os.path.join('Detectors', 'YOLOV5_TPH', 'TreinoYOLOV5TPH.sh')

    target_dir = Path(fold_dir) / 'YOLOV5_TPH'
    if target_dir.exists():
        shutil.rmtree(target_dir)

    subprocess.run([treino], check=True)

    output_dir = _resolve_output_dir()
    if not output_dir.exists():
        raise FileNotFoundError(
            f"YOLOV5_TPH training output not found at {output_dir}. Ensure training succeeded."
        )

    Path(fold_dir).mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_dir), str(target_dir))

    yolo_tph_dir = os.path.join(ROOT_DATA_DIR, 'YOLOV5_TPH')
    if os.path.exists(yolo_tph_dir):
        shutil.rmtree(yolo_tph_dir)
