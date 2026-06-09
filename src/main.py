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
DEFAULT_DATASET_CANDIDATES = (
    PROJECT_ROOT / "dataset" / "all_320",
    PROJECT_ROOT / "dataset" / "all",
    Path("/home/neto/development/buracos/dataset_problematico/all_320"),
    Path("/home/neto/development/buracos/dataset_problematico/all"),
)

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


def _resolve_dataset_root() -> Path:
    env_root = os.getenv("DATASET_ROOT")
    candidates = [Path(env_root).expanduser()] if env_root else list(DEFAULT_DATASET_CANDIDATES)

    for candidate in candidates:
        files_json_dir = candidate / "filesJSON"
        if files_json_dir.exists():
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Dataset COCO não encontrado. Defina DATASET_ROOT apontando para uma pasta "
        f"com filesJSON/.\nCaminhos verificados:\n{checked}"
    )


def _collect_fold_names(files_json_dir: Path) -> list[str]:
    fold_names = {
        "_".join(path.stem.split("_")[:2])
        for path in files_json_dir.glob("fold_*_*.json")
        if path.is_file()
    }
    if not fold_names:
        raise FileNotFoundError(f"Nenhum arquivo fold_*_*.json encontrado em {files_json_dir}")
    return sorted(fold_names, key=lambda name: int(name.split("_")[1]))

# Remove todos os resultados presentes dos outros treinamentos
def resetar_pasta(caminho):
    shutil.rmtree(caminho, ignore_errors=True)  # Remove a pasta inteira
    os.makedirs(caminho, exist_ok=True)  # Recria a pasta vazia

# Função que ira verificar qual modelo sera utilizado para o treinamento
def train_model(model,fold,fold_dir,ROOT_DATA_DIR):
    
    check_save_path = os.path.join(fold_dir,model)

    if os.path.exists(check_save_path):
        if CONTINUE:
            existing_model_path = test_model(model, fold_dir)
            if os.path.exists(existing_model_path):
                return existing_model_path
            print(
                f"[INFO] Checkpoint incompleto para {model} em {fold}: "
                f"{existing_model_path} não existe. Retreinando.",
                flush=True,
            )
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — edite apenas este bloco antes de rodar
# ─────────────────────────────────────────────────────────────────────────────

# Modelos que serão treinados e avaliados quando MODELS_TO_RUN não for definido
# via variável de ambiente. Adicione ou remova nomes conforme necessário.
# Opções disponíveis: YOLOV8 | YOLOV11 | YOLO26 | YOLOV5_TPH | Faster | RetinaNet | Detr | SSDLite
DEFAULT_MODELS = ['YOLOV8', 'Faster', 'Detr']


def _get_models_to_run():
    raw = os.getenv('MODELS_TO_RUN')
    if not raw:
        return normalize_models(DEFAULT_MODELS)
    parsed = [model.strip() for model in raw.split(',') if model.strip()]
    return normalize_models(parsed if parsed else DEFAULT_MODELS)


MODELS = _get_models_to_run()

# False → treina cada modelo e em seguida avalia (fluxo completo).
# True  → pula o treinamento e avalia os pesos já salvos em model_checkpoints/.
#         Use quando o treinamento já foi feito e só quer rever as métricas.
APENAS_TESTE = False

# False → usa dataset padrão em dataset/all/ com anotações COCO em filesJSON/.
# True  → usa dataset tileado em dataset/tiles/<fold_N>/ (imagens recortadas).
#         Exige que a pasta dataset/tiles/ exista com subpastas fold_1/, fold_2/ ...
USE_TILED_DATASET = False

# True  → calcula e salva métricas (mAP, MAE, RMSE, F1 …) em results/results.csv.
# False → roda só o treinamento, sem gerar arquivos de avaliação.
GeraRult = True

# True  → salva imagens com bounding boxes preditos em results/prediction/.
#         Útil para inspeção visual, mas ocupa espaço em disco.
# False → descarta as imagens; só os CSVs são gerados.
save_imgs = True

# True  → salva métricas discriminadas por classe em results/results_by_class.csv.
# False → gera apenas o resultado agregado (suficiente para comparação geral).
GeraResultByClass = False

# False → apaga os pesos anteriores de model_checkpoints/ antes de treinar.
#         Garante um treino limpo a cada execução.
# True  → mantém os pesos já treinados e pula o treino daquela dobra/modelo.
#         Use para retomar uma execução interrompida sem retreinar do zero.
CONTINUE = False

# Inicializado vazio; preenchido por main() antes de qualquer treinamento.
TRAINING_PARAMS_LOG: dict = {}


def main() -> None:
    global TRAINING_PARAMS_LOG, FOLD_NAMES, ROOT_DATA_DIR

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
        tiles_root = str(DATASET_PATH)
        ROOT_DATA_DIR = None
        FOLD_NAMES = fold_dirs
    else:
        dataset_root = _resolve_dataset_root()
        ROOT_DATA_DIR = str(dataset_root)
        DIR_PATH = dataset_root / 'filesJSON'
        FOLD_NAMES = _collect_fold_names(DIR_PATH)
        print(f"[INFO] Dataset: {ROOT_DATA_DIR} | folds={len(FOLD_NAMES)}", flush=True)

    TRAINING_PARAMS_LOG = _new_training_params_payload()
    _write_training_params_json(TRAINING_PARAMS_LOG)

    resetar_pasta(str(RESULTS_PATH))

    if GeraRult:
        RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        print_to_file('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore', RESULTS_CSV_PATH, 'w')
        print_to_file('ml,fold,groundtruth,predicted,TP,FP,dif,fileName', COUNTING_CSV_PATH, 'w')

    if GeraResultByClass:
        RESULTS_BY_CLASS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        print_to_file('ml,fold,classes,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore', RESULTS_BY_CLASS_CSV_PATH, 'w')

    for model in MODELS:
        print(f"[INFO] Processando modelo: {model}")
        for fold in FOLD_NAMES:
            fold_dir = os.path.join('model_checkpoints', fold)
            print(f"[INFO] Iniciando fold {fold} para {model}")

            if USE_TILED_DATASET:
                current_root = os.path.join(tiles_root, fold)
                if not os.path.exists(current_root):
                    print(f"Warning: Tiled dataset not found for {fold} at {current_root}, skipping...")
                    continue
            else:
                current_root = ROOT_DATA_DIR

            if not APENAS_TESTE:
                model_path = train_model(model, fold, fold_dir, current_root)
                if model_path is None:
                    continue
            else:
                model_path = test_model(model, fold_dir)

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


if __name__ == "__main__":
    main()
