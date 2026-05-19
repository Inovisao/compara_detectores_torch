from __future__ import annotations

import os
import shutil
from pathlib import Path

from Detectors.YOLO26.GeraLabels import CriarLabelsYOLO26
from Detectors.YOLO26.config import treino as treino_yolo26


def _training_project_dir() -> Path:
    project = os.getenv("YOLO26_PROJECT", "YOLO26")
    project_path = Path(project)
    if project_path.is_absolute():
        return project_path.resolve()

    src_root = Path(__file__).resolve().parents[2]
    candidates = [
        (src_root / "runs" / "detect" / project_path),
        (Path.cwd() / "runs" / "detect" / project_path),
        project_path.resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def runYOLO26(fold: str, fold_dir: str, root_data_dir: str | Path) -> None:
    dataset_root = Path(root_data_dir).resolve()
    data_yaml_path = CriarLabelsYOLO26(fold, dataset_root)

    target_dir = Path(fold_dir) / "YOLO26"
    if target_dir.exists():
        shutil.rmtree(target_dir)

    os.environ.setdefault(
        "YOLO26_PROJECT",
        str(Path(__file__).resolve().parents[2] / "runs" / "detect" / "YOLO26"),
    )
    treino_yolo26(data_yaml_path)

    project_dir = _training_project_dir()
    if not project_dir.exists():
        raise FileNotFoundError(
            f"YOLO26 training output not found at {project_dir}. "
            "Check if the training routine finished successfully."
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(project_dir), str(target_dir))

    shutil.rmtree(dataset_root / "YOLO26", ignore_errors=True)
    data_yaml_path.unlink(missing_ok=True)
