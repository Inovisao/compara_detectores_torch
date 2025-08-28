import os
import numpy as np
from ResultsDetections import create_csv, print_to_file
from ResultsDetectionsbyclass import generate_results
import shutil

# Remove todos os resultados presentes dos outros treinamentos
def resetar_pasta(caminho):
    shutil.rmtree(caminho, ignore_errors=True)  # Remove a pasta inteira
    os.makedirs(caminho, exist_ok=True)  # Recria a pasta vazia

# Função que ira verificar qual modelo sera utilizado para o treinamento
def train_model(model, fold, fold_dir, ROOT_DATA_DIR):
    model_path = None
    check_save_path = os.path.join(fold_dir, model)

    if os.path.exists(check_save_path):
        if CONTINUE:
            print(f"Retomando {model} no {fold} a partir do checkpoint existente.")
            return None
        shutil.rmtree(check_save_path)

    if model == 'YOLOV8':
        from Detectors.YOLOV8.RunYOLOV8 import runYOLOV8
        runYOLOV8(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'train', 'weights', 'best.pt')

    elif model == 'Faster':
        from Detectors.FasterRCNN.runFaster import runFaster
        runFaster(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'best.pth')

    elif model == 'Detr':
        from Detectors.Detr.runDetr import runDetr
        runDetr(fold, fold_dir, ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir, model, 'training', 'best_model.pth')

    else:
        raise ValueError(f'Modelo {model} not suported')
    return model_path

# Função que server para selecionar os modelos que ja foram treinados
def test_model(model, fold_dir):
    if model == 'YOLOV8':
        model_path = os.path.join(fold_dir, model, 'train', 'weights', 'best.pt')
    elif model == 'Faster':
        model_path = os.path.join(fold_dir, model, 'best.pth')
    elif model == 'Detr':
        model_path = os.path.join(fold_dir, model, 'training', 'best_model.pth')
    else:
        model_path = os.path.join(fold_dir, model, 'latest.pth')
    return model_path

# YOLOV8, Faster, Detr
MODELS = ['Faster']  # Variavel para selecionar os modelos

APENAS_TESTE = False  # True para apenas testar modelos treinados False para Treinar e Testar.
ROOT_DATA_DIR = os.path.join('..', 'dataset', 'all')
DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
DOBRAS = int(len(os.listdir(DIR_PATH)) / 3)
GeraRult = True  # True para gerar Resultados False para não gerar
save_imgs = True  # True para salvar imagens em predictes False para não salvar
GeraResultByClass = False  # True para Salvar Resultados Por classes
CONTINUE = True  # True para Continuar sem apagar os pesos ja treinados

resetar_pasta(os.path.join("..", "results", "prediction"))

if GeraRult:
    if not os.path.exists('../results'):
        os.makedirs('../results')
    print_to_file('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore', '../results/results.csv', 'w')
    print_to_file('ml,fold,groundtruth,predicted,TP,FP,dif,fileName', '../results/counting.csv', 'w')  # Inicia o arquivo de Results

if GeraResultByClass:
    if not os.path.exists('../results'):
        os.makedirs('../results')
    print_to_file('ml,fold,classes,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore', '../results/resultsbyclass.csv', 'w')

# === Detectar último fold treinado ===
folds_existentes = [d for d in os.listdir("model_checkpoints") if d.startswith("fold_")]
if folds_existentes:
    last_trained = max([int(f.split("_")[1]) for f in folds_existentes])
    START_FOLD = last_trained
    print(f"➡️ Retomando a partir do {START_FOLD}")
else:
    START_FOLD = 1
    print("➡️ Nenhum fold anterior encontrado, começando do fold_1")

# Loop Para o selecionar o Modelo
for model in MODELS:
    # Loop Para Treinar o Modelo na referente a Dobra
    for f in np.arange(START_FOLD, DOBRAS + 1):
        fold = 'fold_' + str(f)  # Selecione a Pasta referente a dobra
        fold_dir = os.path.join('model_checkpoints', fold)
        if not APENAS_TESTE:
            model_path = train_model(model, fold, fold_dir, ROOT_DATA_DIR)
            if model_path is None:
                continue
        else:
            model_path = test_model(model, fold_dir)

        if GeraRult:
            create_csv(root=ROOT_DATA_DIR, fold=fold, selected_model=model, model_path=model_path, save_imgs=save_imgs)
        if GeraResultByClass:
            generate_results(root=ROOT_DATA_DIR, fold=fold, model=model_path, model_name=model, save_imgs=save_imgs)
