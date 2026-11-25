from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import warnings

from hyperparameters import param

# Desativa integrações do Weights & Biases para evitar pedidos de login
os.environ.setdefault("WANDB_DISABLED", "true")
warnings.filterwarnings("ignore")

#https://docs.ultralytics.com/pt/modes/train/#resuming-interrupted-trainings Link para os parametros de treino


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_DIR = Path(__file__).resolve().parent / "tph-yolov5"
PROJECT_NAME = os.getenv("TPH_PROJECT", "YOLOV5_TPH")
OUTPUT_PROJECT = PROJECT_ROOT / PROJECT_NAME
DEFAULT_DATA_YAML = PROJECT_ROOT / "dataset" / "all" / "data_yolov5_tph.yaml"


# Hyperparâmetros: valores em src/hyperparameters.py (por dataset); TPH_* sobrescreve.
CFG = param("YOLOV5_TPH", "cfg", "TPH_CFG")
IMG_SIZE = param("YOLOV5_TPH", "img_size", "TPH_IMG")
EPOCHS = param("YOLOV5_TPH", "epochs", "TPH_EPOCHS")
PATIENCE = param("YOLOV5_TPH", "patience", "TPH_PATIENCE")
BATCH = param("YOLOV5_TPH", "batch", "TPH_BATCH")
OPTIMIZER = param("YOLOV5_TPH", "optimizer", "TPH_OPTIMIZER")
SINGLE_CLS = param("YOLOV5_TPH", "single_cls", "TPH_SINGLE_CLS")
RECT = param("YOLOV5_TPH", "rect", "TPH_RECT")
COS_LR = param("YOLOV5_TPH", "cos_lr", "TPH_COS_LR")
LR0 = param("YOLOV5_TPH", "lr0", "TPH_LR0")
LRF = param("YOLOV5_TPH", "lrf", "TPH_LRF")
PLOTS = param("YOLOV5_TPH", "plots", "TPH_PLOTS")

HYP_PATH = param("YOLOV5_TPH", "hyp_path", "TPH_HYP")
WEIGHTS_PATH = param("YOLOV5_TPH", "pretrained_weights", "TPH_PRETRAINED")
DEVICE = param("YOLOV5_TPH", "device", "TPH_DEVICE")
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
