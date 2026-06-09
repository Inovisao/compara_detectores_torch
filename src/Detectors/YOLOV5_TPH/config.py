from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import warnings

# Desativa integrações do Weights & Biases para evitar pedidos de login
os.environ.setdefault("WANDB_DISABLED", "true")
warnings.filterwarnings("ignore")

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
DEFAULT_DATA_YAML = PROJECT_ROOT / "dataset" / "all" / "data_yolov5_tph.yaml"


# Hyperparâmetros e opções de treino configuráveis
CFG = _env_override("TPH_CFG", "yolov5s.yaml")
IMG_SIZE = _env_override("TPH_IMG", 640)
EPOCHS = _env_override("TPH_EPOCHS", 100)
PATIENCE = _env_override("TPH_PATIENCE", 10)
BATCH = _env_override("TPH_BATCH", 32)
OPTIMIZER = _env_override("TPH_OPTIMIZER", "SGD")
SINGLE_CLS = _env_override("TPH_SINGLE_CLS", True)
RECT = _env_override("TPH_RECT", False)
COS_LR = _env_override("TPH_COS_LR", True)
LR0 = _env_override("TPH_LR0", 0.01)
LRF = _env_override("TPH_LRF", 0.2)
MOMENTUM = _env_override("TPH_MOMENTUM", 0.937)
WEIGHT_DECAY = _env_override("TPH_WEIGHT_DECAY", 0.0005)
MOSAIC = _env_override("TPH_MOSAIC", 0.0)
MIXUP = _env_override("TPH_MIXUP", 0.0)
COPY_PASTE = _env_override("TPH_COPY_PASTE", 0.0)
WORKERS = _env_override("TPH_WORKERS", 8)
PLOTS = _env_override("TPH_PLOTS", True)

HYP_PATH = os.getenv("TPH_HYP")
WEIGHTS_PATH = os.getenv("TPH_PRETRAINED")
DEVICE = os.getenv("TPH_DEVICE")
RUN_NAME = os.getenv("TPH_RUN_NAME", "train")
DEFAULT_HYP = REPO_DIR / "data" / "hyps" / "hyp.scratch.yaml"


def get_training_params(data_yaml: Path | None = None) -> dict:
    data_yaml_path = Path(os.getenv("TPH_DATA", data_yaml or DEFAULT_DATA_YAML))
    return {
        "cfg": CFG,
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH,
        "optimizer": OPTIMIZER,
        "single_cls": SINGLE_CLS,
        "rect": RECT,
        "cos_lr": COS_LR,
        "lr0": LR0,
        "lrf": LRF,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "mosaic": MOSAIC,
        "mixup": MIXUP,
        "copy_paste": COPY_PASTE,
        "workers": WORKERS,
        "plots": PLOTS,
        "hyp_path": HYP_PATH,
        "pretrained_weights": WEIGHTS_PATH,
        "device": DEVICE,
        "run_name": RUN_NAME,
        "project": str(OUTPUT_PROJECT),
        "data": str(data_yaml_path),
        "repo_dir": str(REPO_DIR),
    }


def _ensure_prerequisites(data_yaml_path: Path) -> None:
    if not REPO_DIR.exists():
        raise FileNotFoundError(
            "Repository tph-yolov5 not found. Clone https://github.com/cv516Buaa/tph-yolov5 "
            "into src/Detectors/YOLOV5_TPH/tph-yolov5 before running the training."
        )
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Data configuration not found at {data_yaml_path}. Run the label generation step before training."
        )


# Função para Rodar o Treino da YOLOV5 TPH
def treino(data_yaml: Path | None = None):
    data_yaml_path = Path(os.getenv("TPH_DATA", data_yaml or DEFAULT_DATA_YAML))
    _ensure_prerequisites(data_yaml_path)

    if HYP_PATH:
        hyp_file = Path(HYP_PATH)
    else:
        with open(DEFAULT_HYP, "r", encoding="utf-8") as f:
            hyperparams = yaml.safe_load(f)
        hyperparams["lr0"] = LR0
        hyperparams["lrf"] = LRF
        hyperparams["momentum"] = MOMENTUM
        hyperparams["weight_decay"] = WEIGHT_DECAY
        hyperparams["mosaic"] = MOSAIC
        hyperparams["mixup"] = MIXUP
        hyperparams["copy_paste"] = COPY_PASTE

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
        str(data_yaml_path),
        "--cfg",
        str(CFG),
        "--project",
        str(OUTPUT_PROJECT),
        "--name",
        RUN_NAME,
        "--exist-ok",
        "--hyp",
        str(hyp_file),
        "--workers",
        str(WORKERS),
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


if __name__ == "__main__":
    custom_data = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    treino(custom_data)
