from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from ResultsDetections import create_csv, print_to_file, RESULTS_CSV_PATH, COUNTING_CSV_PATH, RESULTS_PATH
from ResultsDetectionsbyclass import generate_results, RESULTS_BY_CLASS_CSV_PATH
import shutil
import time
from datetime import datetime
from dataset_contract import resolve_evaluation_tiling_mode, validate_dataset_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUESTED_TILING_MODE = os.getenv("TILING_MODE")
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
    "ViT",
)

EVALUATION_ARCH_DIR = {
    "YOLOV8": "yolo",
    "Faster": "faster_rcnn",
    "Detr": "detr",
}

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
    "VIT": "ViT",
    "SMALL_VIT": "ViT",
    "SMALL-VIT": "ViT",
    "YOLOS": "ViT",
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
    if model == "ViT":
        from Detectors.ViT.config import get_training_params
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


def _dataset_mode(dataset_root: str | Path) -> str:
    return Path(dataset_root).resolve().name


def _fold_number(fold: str) -> int | None:
    try:
        return int(str(fold).removeprefix("fold_"))
    except ValueError:
        return None


def _canonical_models_root() -> Path:
    raw = os.getenv("EVAL_MODELS_ROOT", "models")
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    repo_root = PROJECT_ROOT.parents[1]
    return (repo_root / path).resolve()


def _write_checkpoint_manifest(
    *,
    model: str,
    fold: str,
    dataset_root: str,
    model_path: str,
    fold_dir: str,
) -> None:
    checkpoint = Path(model_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint esperado não encontrado: {checkpoint}")

    mode = _dataset_mode(dataset_root)
    manifest = {
        "mode": mode,
        "fold": _fold_number(fold),
        "fold_name": fold,
        "architecture": model,
        "checkpoint": str(checkpoint),
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_annotations": str(Path(dataset_root).resolve() / "filesJSON" / f"{fold}_train.json"),
        "val_annotations": str(Path(dataset_root).resolve() / "filesJSON" / f"{fold}_val.json"),
        "test_annotations": str(Path(dataset_root).resolve() / "filesJSON" / f"{fold}_test.json"),
        "source": "train_model",
    }

    if os.getenv("WRITE_LOCAL_WEIGHT_MANIFESTS", "false").strip().lower() in {"1", "true", "yes"}:
        local_manifest = Path(fold_dir) / model / "manifest.json"
        local_manifest.parent.mkdir(parents=True, exist_ok=True)
        local_manifest.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    arch_dir = EVALUATION_ARCH_DIR.get(model)
    if arch_dir:
        eval_manifest = _canonical_models_root() / mode / fold / arch_dir / "manifest.json"
        eval_manifest.parent.mkdir(parents=True, exist_ok=True)
        eval_manifest.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")


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
    if not env_root:
        raise RuntimeError(
            "DATASET_ROOT é obrigatório. Aponte para dataset/sahi, "
            "dataset/asahi ou dataset/asahi_rect."
        )

    candidate = Path(env_root).expanduser()
    files_json_dir = candidate / "filesJSON"
    if not files_json_dir.exists():
        raise FileNotFoundError(
            f"DATASET_ROOT inválido: {candidate}. Diretório filesJSON/ não encontrado."
        )
    return candidate.resolve()


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
    check_save_path = os.path.join(fold_dir, model)
    print(f"[PIPELINE] train_model: model={model} fold={fold}", flush=True)
    print(f"[PIPELINE]   check_save_path={check_save_path} exists={os.path.exists(check_save_path)}", flush=True)
    print(f"[PIPELINE]   CONTINUE={CONTINUE}", flush=True)

    if os.path.exists(check_save_path):
        if CONTINUE:
            existing_model_path = test_model(model, fold_dir)
            print(f"[PIPELINE]   CONTINUE=True → verificando checkpoint: {existing_model_path} exists={os.path.exists(existing_model_path)}", flush=True)
            if os.path.exists(existing_model_path):
                print(f"[PIPELINE]   Checkpoint encontrado, PULANDO treino.", flush=True)
                return existing_model_path
            print(
                f"[INFO] Checkpoint incompleto para {model} em {fold}: "
                f"{existing_model_path} não existe. Retreinando.",
                flush=True,
            )
        print(f"[PIPELINE]   CONTINUE=False → removendo pasta antiga e retreinando.", flush=True)
        shutil.rmtree(check_save_path)
    else:
        print(f"[PIPELINE]   Pasta não existe → iniciando treino do zero.", flush=True)

    print(f"[PIPELINE]   Chamando run{model}(fold={fold})...", flush=True)
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
        print(f"[PIPELINE]   → runFaster iniciado", flush=True)
        runFaster(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'best.pth')
        print(f"[PIPELINE]   → runFaster concluído, model_path={model_path} exists={os.path.exists(model_path)}", flush=True)
    
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
        print(f"[PIPELINE]   → runDetr iniciado", flush=True)
        runDetr(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'training','best_model.pth')
        print(f"[PIPELINE]   → runDetr concluído, model_path={model_path} exists={os.path.exists(model_path)}", flush=True)

    elif model == 'SSDLite':
        from Detectors.SSDLite.RunSSDLite import runSSDLite
        runSSDLite(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'best.pth')

    elif model == 'ViT':
        from Detectors.ViT.RunViT import runViT
        runViT(fold, fold_dir, ROOT_DATA_DIR)
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
    elif model == 'ViT':
        model_path = os.path.join(fold_dir, model, 'best.pth')
    else:
        model_path = os.path.join(fold_dir,model,'latest.pth')
    return model_path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO — edite apenas este bloco antes de rodar
# ─────────────────────────────────────────────────────────────────────────────

# Modelos que serão treinados e avaliados quando MODELS_TO_RUN não for definido
# via variável de ambiente. Adicione ou remova nomes conforme necessário.
# Opções disponíveis: YOLOV8 | YOLOV11 | YOLO26 | YOLOV5_TPH | Faster | RetinaNet | Detr | SSDLite | ViT
# DEFAULT_MODELS = ['Detr', 'Faster', 'YOLOV8', 'YOLOV5_TPH']
DEFAULT_MODELS = ['YOLOV8', 'Faster', 'Detr']
#DEFAULT_MODELS = ['YOLOV8', 'Faster', 'Detr']

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
CONTINUE = True

# Inicializado vazio; preenchido por main() antes de qualquer treinamento.
TRAINING_PARAMS_LOG: dict = {}


def _apply_smoke_test_env() -> None:
    smoke_defaults = {
        "YOLOV8_EPOCHS": "1",
        "YOLOV8_BATCH": "2",
        "YOLOV8_WORKERS": "0",
        "YOLOV8_PATIENCE": "1",
        "YOLOV8_PLOTS": "false",
        "FASTER_EPOCHS": "1",
        "FASTER_BATCH": "2",
        "FASTER_WORKERS": "0",
        "FASTER_PATIENCE": "1",
        "FASTER_LR": "0.0001",
        "FASTER_CLIP_GRAD_NORM": "1.0",
        "DETR_EPOCHS": "1",
        "DETR_BATCH": "2",
        "DETR_WORKERS": "0",
        "DETR_PATIENCE": "1",
    }
    for key, value in smoke_defaults.items():
        os.environ.setdefault(key, value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train detectors on explicit dataset folds.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a lightweight 1-epoch training pass per selected model/fold.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip metric/image generation after training.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global TRAINING_PARAMS_LOG, FOLD_NAMES, ROOT_DATA_DIR, GeraRult, GeraResultByClass, save_imgs

    args = _parse_args(argv)
    if args.smoke_test:
        _apply_smoke_test_env()
        GeraRult = False
        GeraResultByClass = False
        save_imgs = False
        print("[SMOKE] 1 época por fold; avaliação e visualizações desativadas.", flush=True)
    if args.no_eval:
        GeraRult = False
        GeraResultByClass = False
        save_imgs = False

    dataset_root = _resolve_dataset_root()
    ROOT_DATA_DIR = str(dataset_root)
    DIR_PATH = dataset_root / 'filesJSON'
    FOLD_NAMES = _collect_fold_names(DIR_PATH)
    print(f"[INFO] Dataset: {ROOT_DATA_DIR} | folds={len(FOLD_NAMES)}", flush=True)
    contract_errors = validate_dataset_contract(dataset_root)
    if contract_errors:
        preview = "\n".join(f"  - {error}" for error in contract_errors[:20])
        raise ValueError(f"Dataset inválido para o contrato esperado:\n{preview}")

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

    _ckpt_root = os.getenv('MODEL_CHECKPOINTS_ROOT', 'model_checkpoints')

    for model in MODELS:
        print(f"[INFO] Processando modelo: {model}")
        for fold in FOLD_NAMES:
            fold_dir = os.path.join(_ckpt_root, fold)
            print(f"[INFO] Iniciando fold {fold} para {model}")

            current_root = ROOT_DATA_DIR
            tiling_mode = resolve_evaluation_tiling_mode(current_root, REQUESTED_TILING_MODE)

            if not APENAS_TESTE:
                t0 = time.time()
                model_path = train_model(model, fold, fold_dir, current_root)
                elapsed = time.time() - t0
                print(f"[TEMPO] {model} | {fold}: {elapsed/60:.1f} min ({elapsed:.0f}s)", flush=True)
                if model_path is None:
                    continue
            else:
                model_path = test_model(model, fold_dir)

            _write_checkpoint_manifest(
                model=model,
                fold=fold,
                dataset_root=current_root,
                model_path=model_path,
                fold_dir=fold_dir,
            )

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
                    tiling_mode=tiling_mode,
                )
            if GeraResultByClass:
                generate_results(
                    root=current_root,
                    fold=fold,
                    model=model_path,
                    model_name=model,
                    save_imgs=save_imgs,
                    tiling_mode=tiling_mode,
                )


if __name__ == "__main__":
    main()
