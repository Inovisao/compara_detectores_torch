from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_DIR = Path(__file__).resolve().parent / "tph-yolov5"
OUTPUT_PROJECT = Path(__file__).resolve().parents[3] / "YOLOV5_TPH"
DATA_YAML = PROJECT_ROOT / "dataset" / "all" / "data_yolov5_tph.yaml"


def _ensure_prerequisites() -> None:
    if not REPO_DIR.exists():
        raise FileNotFoundError(
            "Repository tph-yolov5 not found. Clone https://github.com/cv516Buaa/tph-yolov5 "
            "into src/Detectors/YOLOV5_TPH/tph-yolov5 before running the training."
        )
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Data configuration not found at {DATA_YAML}. Run the label generation step before training."
        )


def train() -> None:
    _ensure_prerequisites()

    command = [
        sys.executable,
        str(REPO_DIR / "train.py"),
        "--img",
        os.getenv("TPH_IMG", "640"),
        "--batch",
        os.getenv("TPH_BATCH", "4"),
        "--epochs",
        os.getenv("TPH_EPOCHS", "300"),
        "--data",
        str(DATA_YAML),
        "--cfg",
        os.getenv("TPH_CFG", "models/tph/yolov5s.yaml"),
        "--project",
        str(OUTPUT_PROJECT),
        "--name",
        "train",
        "--exist-ok",
    ]

    hyp = os.getenv("TPH_HYP")
    if hyp:
        command.extend(["--hyp", hyp])

    weights = os.getenv("TPH_PRETRAINED")
    if weights:
        command.extend(["--weights", weights])

    device = os.getenv("TPH_DEVICE")
    if device:
        command.extend(["--device", device])

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    repo_path = str(REPO_DIR)
    if repo_path not in pythonpath:
        env["PYTHONPATH"] = f"{repo_path}:{pythonpath}" if pythonpath else repo_path

    subprocess.run(command, cwd=REPO_DIR, check=True, env=env)


if __name__ == "__main__":
    train()
