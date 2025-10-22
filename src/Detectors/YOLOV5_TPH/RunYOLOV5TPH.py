import os
import shutil
import subprocess
from pathlib import Path

from Detectors.YOLOV5_TPH.GeraLabels import CriarLabelsYOLOV5TPH


def _resolve_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    project_name = os.getenv("TPH_PROJECT", "YOLOV5_TPH")
    return project_root / project_name


def runYOLOV5TPH(fold, fold_dir, root_data_dir):
    dataset_root = Path(root_data_dir).resolve()
    data_yaml_path = CriarLabelsYOLOV5TPH(fold, dataset_root)
    treino = os.path.join('Detectors', 'YOLOV5_TPH', 'TreinoYOLOV5TPH.sh')

    target_dir = Path(fold_dir) / 'YOLOV5_TPH'
    if target_dir.exists():
        shutil.rmtree(target_dir)

    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore")
    env["TPH_DATA"] = str(data_yaml_path)
    subprocess.run([treino], check=True, env=env)

    output_dir = _resolve_output_dir()
    if not output_dir.exists():
        raise FileNotFoundError(
            f"YOLOV5_TPH training output not found at {output_dir}. Ensure training succeeded."
        )

    Path(fold_dir).mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_dir), str(target_dir))

    yolo_tph_dir = dataset_root / 'YOLOV5_TPH'
    if yolo_tph_dir.exists():
        shutil.rmtree(yolo_tph_dir)

    if data_yaml_path.exists():
        data_yaml_path.unlink()
