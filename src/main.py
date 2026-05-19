from __future__ import annotations

import os
import json
from pathlib import Path
from ResultsDetections import create_csv, print_to_file, RESULTS_CSV_PATH, COUNTING_CSV_PATH, RESULTS_PATH
from ResultsDetectionsbyclass import generate_results, RESULTS_BY_CLASS_CSV_PATH
import shutil
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "tiles"
TILING_MODE = os.getenv("TILING_MODE", "basic")
TRAINING_PARAMS_JSON_PATH = PROJECT_ROOT / "results" / "training_params.json"

# Disable Weights & Biases logging unless explicitly re-enabled outside.
os.environ.setdefault("WANDB_DISABLED", "true")

SUPPORTED_MODELS = (
    "YOLOV8",
    "YOLOV11",
    "YOLO26",
    "YOLOV5_TPH",
    "Faster",
    "RetinaNet",
    "Detr",
    "SSDLite",
)

MODEL_NAME_ALIASES = {
    "YOLOV8": "YOLOV8",
    "YOLOV11": "YOLOV11",
    "YOLO26": "YOLO26",
    "YOLOV5_TPH": "YOLOV5_TPH",
    "YOLOV5-TPH": "YOLOV5_TPH",
    "FASTER": "Faster",
    "FASTERRCNN": "Faster",
    "RETINANET": "RetinaNet",
    "DETR": "Detr",
    "SSDLITE": "SSDLite",
    "SSD-LITE": "SSDLite",
    "SSD_LITE": "SSDLite",
}


def clear_dataset_cache(base_path: Path) -> None:
    for cache_file in base_path.rglob("*.cache"):
        try:
            cache_file.unlink()
        except OSError:
            pass


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    if not normalized:
        raise ValueError("Nome de modelo vazio encontrado na configuração.")

    alias_key = normalized.replace(" ", "").upper()
    if alias_key in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[alias_key]

    raise ValueError(
        f"Modelo não suportado: {model_name}. "
        f"Use um destes: {', '.join(SUPPORTED_MODELS)}"
    )


def normalize_models(models: list[str]) -> list[str]:
    normalized_models = []
    seen = set()

    for model_name in models:
        canonical_name = normalize_model_name(model_name)
        if canonical_name not in seen:
            normalized_models.append(canonical_name)
            seen.add(canonical_name)

    return normalized_models


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _get_model_training_params(model: str) -> dict:
    if model == "YOLOV8":
        from Detectors.YOLOV8.config import get_training_params
        return get_training_params()
    if model == "YOLOV11":
        from Detectors.YOLOV11.config import get_training_params
        return get_training_params()
    if model == "YOLO26":
        from Detectors.YOLO26.config import get_training_params
        return get_training_params()
    if model == "YOLOV5_TPH":
        from Detectors.YOLOV5_TPH.config import get_training_params
        return get_training_params()
    if model == "Faster":
        from Detectors.FasterRCNN.config import get_training_params
        return get_training_params()
    if model == "RetinaNet":
        from Detectors.RetinaNet.config import get_training_params
        return get_training_params()
    if model == "Detr":
        from Detectors.Detr.config import get_training_params
        return get_training_params()
    if model == "SSDLite":
        from Detectors.SSDLite.config import get_training_params
        return get_training_params()
    raise ValueError(f"Modelo não suportado para log de parâmetros: {model}")


def _new_training_params_payload() -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_models": MODELS,
        "folds": FOLD_NAMES,
        "models": {},
        "runs": [],
    }


def _write_training_params_json(payload: dict) -> None:
    TRAINING_PARAMS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_PARAMS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def _resolve_training_params_dir(model: str, model_path: str | None) -> Path:
    if not model_path:
        return TRAINING_PARAMS_JSON_PATH.parent

    checkpoint_path = Path(model_path)
    if model in {"YOLOV8", "YOLOV11", "YOLO26", "YOLOV5_TPH"} and checkpoint_path.name == "best.pt":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _write_run_training_params_json(run_payload: dict) -> None:
    output_dir = _resolve_training_params_dir(run_payload["rede"], run_payload.get("model_path"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_params.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(run_payload), f, indent=2, ensure_ascii=False)


def register_training_params(model: str, fold: str, root: str, mode: str, model_path: str | None) -> None:
    if model not in TRAINING_PARAMS_LOG["models"]:
        TRAINING_PARAMS_LOG["models"][model] = {
            "rede": model,
            "hyperparametros": _get_model_training_params(model),
        }

    run_payload = {
        "rede": model,
        "fold": fold,
        "modo": mode,
        "dataset_root": root,
        "model_path": model_path,
        "hyperparametros": TRAINING_PARAMS_LOG["models"][model]["hyperparametros"],
    }
    TRAINING_PARAMS_LOG["runs"].append(run_payload)
    _write_run_training_params_json(run_payload)
    _write_training_params_json(TRAINING_PARAMS_LOG)

# Remove todos os resultados presentes dos outros treinamentos
def resetar_pasta(caminho):
    shutil.rmtree(caminho, ignore_errors=True)  # Remove a pasta inteira
    os.makedirs(caminho, exist_ok=True)  # Recria a pasta vazia

# Função que ira verificar qual modelo sera utilizado para o treinamento
def train_model(model,fold,fold_dir,ROOT_DATA_DIR):
    
    check_save_path = os.path.join(fold_dir,model)

    if os.path.exists(check_save_path):
        if CONTINUE:
            return test_model(model, fold_dir)
        shutil.rmtree(check_save_path)
    if model == 'YOLOV8':
        from Detectors.YOLOV8.RunYOLOV8 import runYOLOV8
        runYOLOV8(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')

    elif model == 'YOLO26':
        from Detectors.YOLO26.RunYOLO26 import runYOLO26
        runYOLO26(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'train', 'weights', 'best.pt')

    elif model == 'Faster':
        from Detectors.FasterRCNN.runFaster import runFaster
        runFaster(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'best.pth')
    
    elif model == 'YOLOV5_TPH':
        from Detectors.YOLOV5_TPH.RunYOLOV5TPH import runYOLOV5TPH
        runYOLOV5TPH(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'train', 'weights', 'best.pt')

    elif model == 'YOLOV11':
        from Detectors.YOLOV11.RunYOLOV11 import runYOLOV11
        runYOLOV11(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'train', 'weights', 'best.pt')

    elif model == 'RetinaNet':
        from Detectors.RetinaNet.RunRetinaNet import runRetinaNet
        runRetinaNet(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'best.pth')

    elif model == 'Detr':
        from Detectors.Detr.runDetr import runDetr
        runDetr(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'training','best_model.pth')

    elif model == 'SSDLite':
        from Detectors.SSDLite.RunSSDLite import runSSDLite
        runSSDLite(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'best.pth')

    return model_path
# Função que server para selecionar os modelos que ja foram treinados
def test_model(model,fold_dir):
    if model == 'YOLOV8':
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')
    elif model == 'YOLO26':
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')
    elif model == 'YOLOV11':
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')
    elif model == 'Faster':
        model_path = os.path.join(fold_dir,model,'best.pth')
    elif model == 'YOLOV5_TPH':
        model_path = os.path.join(fold_dir, model, 'train', 'weights', 'best.pt')
    elif model == 'RetinaNet':
        model_path = os.path.join(fold_dir, model, 'best.pth')
    elif model == 'Detr':
        model_path = os.path.join(fold_dir,model,'training','best_model.pth')
    elif model == 'SSDLite':
        model_path = os.path.join(fold_dir, model, 'best.pth')
    else:
        model_path = os.path.join(fold_dir,model,'latest.pth')
    return model_path

# YOLOV8, YOLOV11, YOLO26, YOLOV5_TPH, Faster, RetinaNet, Detr
DEFAULT_MODELS = ['YOLOV8', 'YOLOV11', 'YOLO26', 'Faster']


def _get_models_to_run():
    raw = os.getenv('MODELS_TO_RUN')
    if not raw:
        return normalize_models(DEFAULT_MODELS)
    parsed = [model.strip() for model in raw.split(',') if model.strip()]
    return normalize_models(parsed if parsed else DEFAULT_MODELS)


MODELS = _get_models_to_run()
APENAS_TESTE = False # True para apenas testar modelos treinados False para Treinar e Testar.

# Tiled dataset configuration
USE_TILED_DATASET = False

if USE_TILED_DATASET:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Tiled dataset directory not found: {DATASET_PATH}")
    clear_dataset_cache(DATASET_PATH)
    fold_dirs = sorted(
        d.name for d in DATASET_PATH.iterdir()
        if d.is_dir() and d.name.startswith('fold_')
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No folds found inside tiled dataset directory: {DATASET_PATH}")
    DOBRAS = len(fold_dirs)
    tiles_root = str(DATASET_PATH)
    ROOT_DATA_DIR = None  # Will be set per fold
    FOLD_NAMES = fold_dirs
else:
    ROOT_DATA_DIR = os.path.join('..', 'dataset','all')
    DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
    DOBRAS = int(len(os.listdir(DIR_PATH))/3)
    FOLD_NAMES = [f'fold_{i}' for i in range(1, DOBRAS + 1)]

TRAINING_PARAMS_LOG = _new_training_params_payload()
_write_training_params_json(TRAINING_PARAMS_LOG)

GeraRult = True # True para gerar Resultados False para não gerar
save_imgs = True # True para salvar imagens em predictes False para não salvar
GeraResultByClass = False # True para Salvar Resultados Por classes
CONTINUE = False # True para Continuar sem apagar os pesos ja treinados
resetar_pasta(str(RESULTS_PATH))

if GeraRult:
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print_to_file('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore', RESULTS_CSV_PATH, 'w')
    print_to_file('ml,fold,groundtruth,predicted,TP,FP,dif,fileName', COUNTING_CSV_PATH, 'w')# Inicia o arquivo de Results

if GeraResultByClass:
    RESULTS_BY_CLASS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print_to_file('ml,fold,classes,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore', RESULTS_BY_CLASS_CSV_PATH, 'w')

# Loop Para o selecionar o Modelo
for model in MODELS:
    print(f"[INFO] Processando modelo: {model}")
    # Loop Para Treinar o Modelo na referente a Dobra
    for fold in FOLD_NAMES:
        fold_dir = os.path.join('model_checkpoints', fold)
        print(f"[INFO] Iniciando fold {fold} para {model}")

        # Set ROOT_DATA_DIR for this fold if using tiled datasets
        if USE_TILED_DATASET:
            current_root = os.path.join(tiles_root, fold)
            if not os.path.exists(current_root):
                print(f"Warning: Tiled dataset not found for {fold} at {current_root}, skipping...")
                continue
        else:
            current_root = ROOT_DATA_DIR

        if not APENAS_TESTE:
            model_path = train_model(model,fold,fold_dir,current_root)
            if model_path == None:
                continue
        else:
            model_path =  test_model(model,fold_dir)

        register_training_params(
            model=model,
            fold=fold,
            root=current_root,
            mode="test" if APENAS_TESTE else "train",
            model_path=model_path,
        )

        if GeraRult:
            create_csv(
                root=current_root,
                fold=fold,
                selected_model=model,
                model_path=model_path,
                save_imgs=save_imgs,
                tiling_mode=TILING_MODE,
            )
        if GeraResultByClass:
            generate_results(
                root=current_root,
                fold=fold,
                model=model_path,
                model_name=model,
                save_imgs=save_imgs,
                tiling_mode=TILING_MODE,
            )
