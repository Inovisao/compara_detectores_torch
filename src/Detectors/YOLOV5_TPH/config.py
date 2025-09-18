from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Desativa integrações do Weights & Biases para evitar pedidos de login
os.environ.setdefault("WANDB_DISABLED", "true")

#https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treino


def _env_override(name: str, default):
    """Read an environment override while preserving the default type."""
    value = os.getenv(name)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in {"1", "true", "t", "yes", "y"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_DIR = Path(__file__).resolve().parent / "tph-yolov5"
PROJECT_NAME = os.getenv("TPH_PROJECT", "YOLOV5_TPH")
OUTPUT_PROJECT = PROJECT_ROOT / PROJECT_NAME
DATA_YAML = PROJECT_ROOT / "dataset" / "all" / "data_yolov5_tph.yaml"


# Hyperparâmetros e opções de treino configuráveis
CFG = _env_override("TPH_CFG", "models/tph/yolov5s.yaml")
IMG_SIZE = _env_override("TPH_IMG", 640)
EPOCHS = _env_override("TPH_EPOCHS", 10)
PATIENCE = _env_override("TPH_PATIENCE", 3)
BATCH = _env_override("TPH_BATCH", 8)
OPTIMIZER = _env_override("TPH_OPTIMIZER", "AdamW")
SINGLE_CLS = _env_override("TPH_SINGLE_CLS", False)
RECT = _env_override("TPH_RECT", False)
COS_LR = _env_override("TPH_COS_LR", True)
LR0 = _env_override("TPH_LR0", 1e-4)
LRF = _env_override("TPH_LRF", 1e-2)
PLOTS = _env_override("TPH_PLOTS", True)

HYP_PATH = os.getenv("TPH_HYP")
WEIGHTS_PATH = os.getenv("TPH_PRETRAINED")
DEVICE = os.getenv("TPH_DEVICE")
RUN_NAME = os.getenv("TPH_RUN_NAME", "train")
DEFAULT_HYP = REPO_DIR / "data" / "hyps" / "hyp.scratch.yaml"
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


# Função para Rodar o Treino da YOLOV5 TPH
def treino():
    _ensure_prerequisites()

    if HYP_PATH:
        hyp_file = Path(HYP_PATH)
    else:
        with open(DEFAULT_HYP, "r", encoding="utf-8") as f:
            hyperparams = yaml.safe_load(f)
        hyperparams["lr0"] = LR0
        hyperparams["lrf"] = LRF

        OUTPUT_PROJECT.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="hyp_auto_", suffix=".yaml", dir=OUTPUT_PROJECT)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(hyperparams, f, sort_keys=False)
        hyp_file = Path(path)
    command = [
        sys.executable,
        str(REPO_DIR / "train.py"),
        "--img",
        str(IMG_SIZE),
        "--batch-size",
        str(BATCH),
        "--epochs",
        str(EPOCHS),
        "--patience",
        str(PATIENCE),
        "--data",
        str(DATA_YAML),
        "--cfg",
        str(CFG),
        "--project",
        str(OUTPUT_PROJECT),
        "--name",
        RUN_NAME,
        "--exist-ok",
        "--hyp",
        str(hyp_file),
    ]

    if SINGLE_CLS:
        command.append("--single-cls")
    if RECT:
        command.append("--rect")
    if not COS_LR:
        command.append("--linear-lr")
    if OPTIMIZER and OPTIMIZER.lower() in {"adam", "adamw"}:
        command.append("--adam")

    if WEIGHTS_PATH:
        command.extend(["--weights", WEIGHTS_PATH])
    if DEVICE:
        command.extend(["--device", DEVICE])

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    repo_path = str(REPO_DIR)
    if repo_path not in pythonpath:
        env["PYTHONPATH"] = f"{repo_path}:{pythonpath}" if pythonpath else repo_path

    subprocess.run(command, cwd=REPO_DIR, check=True, env=env)


treino()
treino()
