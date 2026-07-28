import argparse
import os
import numpy as np
from ResultsDetections import create_csv, print_to_file
from ResultsDetectionsbyclass import generate_results
import shutil
import time

# --------------------------------------------------------------------------- #
# Linha de comando: permite escolher os modelos sem editar o arquivo.
#
#   python main.py                          -> usa o MODELS definido abaixo
#   python main.py --models YOLOV8           -> só o YOLOv8
#   python main.py --models Faster           -> só o Faster
#   python main.py --models YOLOV8 Faster    -> os dois (ordem = ordem de treino)
#
# Valores aceitos: YOLOV8, Faster, Detr (os mesmos nomes que o if/elif de
# train_model reconhece). Qualquer outro nome é rejeitado na hora, com uma
# mensagem clara, em vez de travar depois com UnboundLocalError.
# --------------------------------------------------------------------------- #
MODELOS_VALIDOS = ['YOLOV8', 'Faster', 'Detr']


def parse_args():
    ap = argparse.ArgumentParser(
        description="Treina e avalia detectores por validação cruzada."
    )
    ap.add_argument(
        '--models', '--model', dest='models', nargs='+', default=None,
        metavar='MODELO',
        help=(
            "Quais modelos rodar, em ordem, separados por espaço. "
            f"Valores aceitos: {', '.join(MODELOS_VALIDOS)}. "
            "Se omitido, usa a lista MODELS definida no topo do main.py."
        ),
    )
    args = ap.parse_args()
    if args.models is not None:
        invalidos = [m for m in args.models if m not in MODELOS_VALIDOS]
        if invalidos:
            ap.error(
                f"Modelo(s) inválido(s): {invalidos}. "
                f"Valores aceitos: {MODELOS_VALIDOS}."
            )
    return args


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
    
    elif model == 'Detr':
        from Detectors.Detr.runDetr import runDetr
        runDetr(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'training','best_model.pth')
    return model_path
# Função que server para selecionar os modelos que ja foram treinados
def test_model(model,fold_dir):
    if model == 'YOLOV8':
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')
    elif model == 'Faster':
        model_path = os.path.join(fold_dir,model,'best.pth')
    elif model == 'Detr':
        model_path = os.path.join(fold_dir,model,'training','best_model.pth')
    else:
        model_path = os.path.join(fold_dir,model,'latest.pth')
    return model_path

_args = parse_args()

# YOLOV8, Faster, Detr
MODELS = ['YOLOV8', 'Faster'] #Variavel para selecionar os modelos (usada se --models não for passado)

if _args.models is not None:
    MODELS = _args.models
    print(f"Modelos selecionados via linha de comando: {MODELS}")

APENAS_TESTE = True # True para apenas testar modelos treinados False para Treinar e Testar.
ROOT_DATA_DIR = os.path.join('..', 'dataset','all')
DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
DOBRAS = int(len(os.listdir(DIR_PATH))/3)
print(f"Total de Dobra: {DOBRAS}")
GeraRult = True # True para gerar Resultados False para não gerar
save_imgs = True # True para salvar imagens em predictes False para não salvar
GeraResultByClass = False # True para Salvar Resultados Por classes
CONTINUE = False # True para Continuar sem apagar os pesos ja treinados
resetar_pasta(os.path.join("..","results","prediction"))

if GeraRult:
    if not os.path.exists('../results'):
        os.makedirs('../results')
    print_to_file('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore','../results/results.csv','w')
    print_to_file('ml,fold,groundtruth,predicted,TP,FP,dif,fileName','../results/counting.csv','w')# Inicia o arquivo de Results

if GeraResultByClass:
    if not os.path.exists('../results'):
        os.makedirs('../results')
    print_to_file('ml,fold,classes,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore','../results/resultsbyclass.csv','w')

# Loop Para o selecionar o Modelo
for model in MODELS:
    inicio = time.time()
    # Loop Para Treinar o Modelo na referente a Dobra
    for f in np.arange(1,DOBRAS+1):
        fold = 'fold_'+str(f) # Selecione a Pasta referente a dobra
        fold_dir = os.path.join('model_checkpoints', fold)
        if not APENAS_TESTE:
            model_path = train_model(model,fold,fold_dir,ROOT_DATA_DIR)
            if model_path == None:
                continue
        else:
            model_path =  test_model(model,fold_dir)

        if GeraRult:
            create_csv(root=ROOT_DATA_DIR,fold=fold,selected_model=model,model_path=model_path,save_imgs=save_imgs)
        if GeraResultByClass:
            generate_results(root=ROOT_DATA_DIR,fold=fold,model=model_path,model_name=model,save_imgs=save_imgs)
    fim = time.time()
    print(f"Tempo de execução: {fim - inicio:.4f} segundos do modelo{model}")
