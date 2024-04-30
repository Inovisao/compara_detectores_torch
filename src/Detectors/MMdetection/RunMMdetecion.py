import os
import subprocess
from Detectors.MMdetection.ConfigTrain import configTrain
from Detectors.MMdetection.CheckPoint import Checkpoint
# Função para Rodar o Modelo
def runMMdetection(model,fold_dir):
    modelname = model.split('/')[-1]

    path = model.split('/')[1] # Pega o nome da pasta

    caminho = f'Detectors/MMdetection/mmdetection/configs/{path}' # Caminho para a pasta de config do modelo
    os.system(f'mim download mmdet --config {modelname} --dest Detectors/MMdetection/checkpoints') # baixa o peso do modelo
    checkpoint = Checkpoint(modelname.split('-')[0]) # Pega o peso do modelo
    configTrain(model=modelname,checkpoint=checkpoint,path=caminho) # Roda a função para criar o arquivo de config do Treino
    save_model = os.path.join(fold_dir,modelname) # Pasta onde os modelos seram salvos

    run_model = os.path.join(caminho,'train.py') # Parametros para passar para o bash 

    treino = os.path.join('Detectors','MMdetection','Treinommdetection.sh') # outro parametro

    args = [run_model,save_model] # junta os parametros em uma lista
    subprocess.run([treino]+args) # chama o treino