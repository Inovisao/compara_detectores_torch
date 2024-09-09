import os
import numpy as np
from ResultsDetections import criaCSV, printToFile

def train_model(model,fold,fold_dir,ROOT_DATA_DIR):

    if model == 'YOLOV8':
        from Detectors.YOLOV8.RunYOLOV8 import runYOLOV8
        runYOLOV8(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')

    elif model == 'FasterRCNN':
        from Detectors.FasterRCNN.RunFaster import runFaster
        runFaster(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'best_model.pth')

    elif model == 'Detr':
        from Detectors.Detr.runDetr import runDetr
        runDetr(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'training','best_model.pth')
    elif model == 'Retinanet':
        from Detectors.Retinanet.RunRetinanet import runRetinanet
        runRetinanet(fold,fold_dir,ROOT_DATA_DIR)
        model_path = os.path.join(fold_dir,model,'best_model.pth')
    return model_path

def test_model(model,fold_dir):
    if model == 'YOLOV8':
        model_path = os.path.join(fold_dir,model,'train','weights','best.pt')

    elif model == 'FasterRCNN':
        model_path = os.path.join(fold_dir,model,'best_model.pth')

    elif model == 'Detr':
        model_path = os.path.join(fold_dir,model,'training','best_model.pth')

    elif model == 'Retinanet':
        model_path = os.path.join(fold_dir,model,'best_model.pth')

    return model_path


# YOLOV8, FasterRCNN, Detr
MODELS = ['Retinanet'] #Variavel para selecionar os modelos

APENAS_TESTE = False # True para apenas testar modelos treinados False para Treinar e Testar.
ROOT_DATA_DIR = os.path.join('..', 'dataset','all')
DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
DOBRAS = int(len(os.listdir(DIR_PATH))/3)
GeraRult = False # True para gerar Resultados False para não gerar
save_imgs = False # True para salvar imagens em predictes False para não salvar

if GeraRult:
    if not os.path.exists('../results'):
        os.makedirs('../results')
    printToFile('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore','../results/results.csv','w')
    printToFile('ml,fold,groundtruth,predicted,TP,FP,dif,fileName','../results/counting.csv','w')# Inicia o arquivo de Results
# Loop Para o selecionar o Modelo
for model in MODELS:
    # Loop Para Treinar o Modelo na referente a Dobra
    for f in np.arange(1,DOBRAS+1):
        fold = 'fold_'+str(f) # Selecione a Pasta referente a dobra
        fold_dir = os.path.join('model_checkpoints', fold)
        if not APENAS_TESTE:
           model_path = train_model(model,fold,fold_dir,ROOT_DATA_DIR)
        else:
           model_path =  test_model(model,fold_dir)
        if GeraRult:
            try:
                criaCSV(num_dobra=f,root=ROOT_DATA_DIR,fold=fold,selected_model=model,model_path=model_path,save_imgs=save_imgs)
            except:
                pass