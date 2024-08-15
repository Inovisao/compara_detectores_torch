import os
import numpy as np
from ResultsDetections import criaCSV, printToFile

# YOLOV8, FasterRCNN, 'MMdetections/sabl-faster-rcnn_r50_fpn_1x_coco','MMdetections/faster-rcnn_r50_fpn_1x_coco','MMdetections/fovea_r50_fpn_4xb4-1x_coco'
MODELS = ['YOLOV8'] #Variavel para selecionar os modelos
APENAS_TESTE = False # Decide se ira Treinar = False ou so fazer o Teste = True dos modelos 
ROOT_DATA_DIR = os.path.join('..', 'dataset','all')
DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
DOBRAS = int(len(os.listdir(DIR_PATH))/3)
GeraRult = True
if GeraRult:
    printToFile('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore','../results/results.csv','w')
    printToFile('ml,fold,groundtruth,predicted,TP,FP,dif,fileName','../results/counting.csv','w')# Inicia o arquivo de Results
# Loop Para o selecionar o Modelo
for model in MODELS:
    # Loop Para Treinar o Modelo na referente a Dobra
    for f in np.arange(1,DOBRAS+1):
        fold = 'fold_'+str(f) # Selecione a Pasta referente a dobra
        fold_dir = os.path.join('model_checkpoints', fold)
        if not APENAS_TESTE:
            if model == 'YOLOV8':
                from Detectors.YOLOV8.RunYOLOV8 import runYOLOV8
                runYOLOV8(fold,fold_dir,ROOT_DATA_DIR)
                model_path = os.path.join(fold_dir,model,'train','weights','best.pt')
                model_name2 = model

            elif model == 'FasterRCNN':
                from Detectors.FasterRCNN.RunFaster import runFaster
                runFaster(fold,fold_dir,ROOT_DATA_DIR)
                model_path = os.path.join(fold_dir,model,'best_model.pth')
                model_name2 = model

            elif model == 'Detr':
                from Detectors.Detr.runDetr import runDetr
                runDetr(fold,fold_dir,ROOT_DATA_DIR)
        else:
            if model == 'YOLOV8':
                model_path = os.path.join(fold_dir,model,'train','weights','best.pt')
                model_name2 = model
            elif model == 'FasterRCNN':
                model_path = os.path.join(fold_dir,model,'best_model.pth')
                model_name2 = model
            elif model == 'Detr':
                model_path = os.path.join(fold_dir,model,'training','best_model.pth')
                model_name2 = model
        if GeraRult:
            try:
                criaCSV(num_dobra=f,root=ROOT_DATA_DIR,fold=fold,selected_model=model_name2,model_path=model_path)
            except:
                pass