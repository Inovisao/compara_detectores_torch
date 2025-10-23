import os
from pathlib import Path
from ResultsDetections import create_csv, print_to_file, RESULTS_CSV_PATH, COUNTING_CSV_PATH, RESULTS_PATH
from ResultsDetectionsbyclass import generate_results, RESULTS_BY_CLASS_CSV_PATH
import shutil
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "tiles"
TILING_MODE = os.getenv("TILING_MODE", "basic")


def clear_dataset_cache(base_path: Path) -> None:
    for cache_file in base_path.rglob("*.cache"):
        try:
            cache_file.unlink()
        except OSError:
            pass

# Remove todos os resultados presentes dos outros treinamentos
def resetar_pasta(caminho):
    shutil.rmtree(caminho, ignore_errors=True)  # Remove a pasta inteira
    os.makedirs(caminho, exist_ok=True)  # Recria a pasta vazia

# Função que ira verificar qual modelo sera utilizado para o treinamento
def train_model(model,fold,fold_dir,ROOT_DATA_DIR):
    
    check_save_path = os.path.join(fold_dir,model)

    if os.path.exists(check_save_path):
        if CONTINUE:
            return None
        shutil.rmtree(check_save_path)
    if model == 'YOLOV8':
        from Detectors.YOLOV8.RunYOLOV8 import runYOLOV8
        runYOLOV8(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')

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
    return model_path
# Função que server para selecionar os modelos que ja foram treinados
def test_model(model,fold_dir):
    if model == 'YOLOV8':
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
    else:
        model_path = os.path.join(fold_dir,model,'latest.pth')
    return model_path

# YOLOV8, YOLOV11, YOLOV5_TPH, Faster, RetinaNet, Detr
DEFAULT_MODELS = ['YOLOV11', 'RetinaNet']


def _get_models_to_run():
    raw = os.getenv('MODELS_TO_RUN')
    if not raw:
        return DEFAULT_MODELS
    parsed = [model.strip() for model in raw.split(',') if model.strip()]
    return parsed if parsed else DEFAULT_MODELS


MODELS = _get_models_to_run()
APENAS_TESTE = False # True para apenas testar modelos treinados False para Treinar e Testar.

# Tiled dataset configuration
USE_TILED_DATASET = os.getenv('USE_TILED_DATASET', 'true').lower() == 'true'

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
    # Loop Para Treinar o Modelo na referente a Dobra
    for fold in FOLD_NAMES:
        fold_dir = os.path.join('model_checkpoints', fold)

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
