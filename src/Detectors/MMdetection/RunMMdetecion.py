import os
import subprocess

# Função que Serve para pegar o caminho da config dos Modelos
def encontrar_pasta_por_parte_do_nome(parte_nome, diretorio):
    for pasta in os.listdir(diretorio):
        if os.path.isdir(os.path.join(diretorio, pasta)) and parte_nome in pasta:
            return os.path.join(diretorio, pasta)
    return None
# Função para Rodar o Modelo
def runMMdetection(model,fold_dir):
    model = model.split('/')[1]
    # Ira verificar se o caminho é com - ou _
    for letra in model:
        if letra == '-':
            splitar = '-'
            break
        elif letra == '_':
            splitar = '_'
            break
    path = model.split(splitar)[0] # Pega o nome do modelo
    caminho = encontrar_pasta_por_parte_do_nome(path,'Detectors/MMdetection/mmdetection/configs') # Caminho para a pasta de config do modelo
    os.system(f'mim download mmdet --config {model} --dest Detectors/MMdetection/checkpoints') # baixa o peso do modelo
    from Detectors.MMdetection.ConfigTrain import configTrain
    from Detectors.MMdetection.CheckPoint import Checkpoint
    checkpoint = Checkpoint(path) # Pega o peso do modelo
    configTrain(model=model,checkpoint=checkpoint,path=caminho) # Roda a função para criar o arquivo de config do Treino
    save_model = os.path.join(fold_dir,model) # Pasta onde os modelos seram salvos
    run_model = os.path.join(caminho,'train.py') # Parametros para passar para o bash 
    treino = os.path.join('Detectors','MMdetection','Treinommdetection.sh') # outro parametro
    args = [run_model,save_model] # junta os parametros em uma lista
    subprocess.run([treino]+args) # chama o treino