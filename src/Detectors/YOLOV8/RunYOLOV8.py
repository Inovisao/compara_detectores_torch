import os
import shutil
import subprocess
from pathlib import Path

from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8


def _training_output_dir() -> Path:
    project = os.getenv("YOLOV8_PROJECT")
    if project:
        project_path = Path(project)
        return project_path.resolve() if project_path.is_absolute() else project_path.resolve()

    src_root = Path(__file__).resolve().parents[2]
    candidates = [
        src_root / "runs" / "detect" / "YOLOV8",
        Path.cwd() / "runs" / "detect" / "YOLOV8",
        Path("YOLOV8").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def runYOLOV8(fold, fold_dir, root_data_dir):
    dataset_root = Path(root_data_dir).resolve()
    data_yaml_path = CriarLabelsYOLOV8(fold, dataset_root)
    treino = Path('Detectors') / 'YOLOV8' / 'TreinoYOLOV8.sh'

    target_dir = Path(fold_dir) / 'YOLOV8'
    if target_dir.exists():
        shutil.rmtree(target_dir)

    env = os.environ.copy()
    env["YOLOV8_DATA"] = str(data_yaml_path)
    env.setdefault("YOLOV8_PROJECT", str(Path(__file__).resolve().parents[2] / "runs" / "detect" / "YOLOV8"))
    subprocess.run([str(treino)], check=True, env=env)

    output_dir = _training_output_dir()
    if not output_dir.exists():
        raise FileNotFoundError(
            f"YOLOV8 training output not found at {output_dir}. Ensure training succeeded."
        )

    Path(fold_dir).mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_dir), str(target_dir))

    shutil.rmtree(dataset_root / 'YOLO', ignore_errors=True)
    data_yaml_path.unlink(missing_ok=True)
