import os
import numpy as np
from RestultsDetections import criaCSV, printToFile
from Detectors.MMdetection.CheckPoint import selectmodel

# YOLOV8, FasterRCNN, 'MMdetections/sabl-faster-rcnn_r50_fpn_1x_coco','MMdetections/detr_r50_8xb2-150e_coco','MMdetections/fovea_r50_fpn_4xb4-1x_coco'
MODELS = ['YOLOV8','MMdetections/sabl-faster-rcnn_r50_fpn_1x_coco','MMdetections/detr_r50_8xb2-150e_coco','MMdetections/fovea_r50_fpn_4xb4-1x_coco'] #Variavel para selecionar os modelos
APENAS_TESTE = False # Decide se ira Treinar = False ou so fazer o Teste = True dos modelos 
ROOT_DATA_DIR = os.path.join('..', 'dataset','all')
DIR_PATH = os.path.join(ROOT_DATA_DIR, 'filesJSON')
DOBRAS = len(os.listdir(DIR_PATH)) // 3
printToFile('ml,fold,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore','../Results/results.csv','w')# Inicia o arquivo de Results
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
                from Detectors.FasterCNN.RunFaster import runFaster
                runFaster(fold,fold_dir,ROOT_DATA_DIR)
                
            elif model[0:12] == 'MMdetections':
                from Detectors.MMdetection.RunMMdetecion import runMMdetection
                runMMdetection(model,fold_dir)
                model_name = model.split('/')[1]
                model_name2 = model.split('/')[0]
                modeloTreinado = selectmodel(fold,model_name)
                model_path = os.path.join(fold_dir,model_name,modeloTreinado)  
        else:
            if model[0:12] == 'MMdetections':
                model_name = model.split('/')[1]
                model_name2 = model.split('/')[0]
                modeloTreinado = selectmodel(fold,model_name)
                model_path = os.path.join(fold_dir,model_name,modeloTreinado)

            elif model == 'YOLOV8':
                model_path = os.path.join(fold_dir,model,'train','weights','best.pt')
                model_name2 = model
                
        criaCSV(num_dobra=f,root=ROOT_DATA_DIR,fold=fold,selected_model=model_name2,model_path=model_path)