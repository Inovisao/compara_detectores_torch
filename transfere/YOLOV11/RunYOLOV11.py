from __future__ import annotations

import os
import shutil
from pathlib import Path

from Detectors.YOLOV11.GeraLabels import CriarLabelsYOLOV11
from Detectors.YOLOV11.config import treino as treino_yolov11


def _training_project_dir() -> Path:
    project = os.getenv("YOLOV11_PROJECT", "YOLOV11")
    project_path = Path(project)
    return project_path.resolve() if project_path.is_absolute() else project_path


def runYOLOV11(fold: str, fold_dir: str, root_data_dir: str | Path) -> None:
    dataset_root = Path(root_data_dir).resolve()
    data_yaml_path = CriarLabelsYOLOV11(fold, dataset_root)

    target_dir = Path(fold_dir) / "YOLOV11"
    if target_dir.exists():
        shutil.rmtree(target_dir)

    treino_yolov11(data_yaml_path)

    project_dir = _training_project_dir()
    if not project_dir.exists():
        raise FileNotFoundError(
            f"YOLOV11 training output not found at {project_dir}. "
            "Check if the training routine finished successfully."
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(project_dir), str(target_dir))

    shutil.rmtree(dataset_root / "YOLOV11", ignore_errors=True)
    data_yaml_path.unlink(missing_ok=True)
