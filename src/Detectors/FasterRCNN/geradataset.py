import os
import shutil
import json

# Caminhos
ROOT_DATA_DIR = os.path.join('..','dataset','all')

# Criar a pasta de destino se não existir
def geredata(fold):

    destination_folder = os.path.join(ROOT_DATA_DIR,'Faster')
    os.makedirs(destination_folder, exist_ok=True)
    foldsUsadas = []
    caminhos = (os.listdir(os.path.join(ROOT_DATA_DIR,'filesJSON')))
    #Pega o caminho do arquivo coco que esta sendo usada
    for caminho in caminhos:

        fold_check = caminho.split("_")[0] + "_" +caminho.split("_")[1]

        if str(fold_check) == str(fold):
            foldsUsadas.append(caminho)
    # Carregar o JSON
    for fold in foldsUsadas:
        path = os.path.join(destination_folder,fold.split('_')[-1][0:-5])
        os.makedirs(path, exist_ok=True)
        json_path = os.path.join(ROOT_DATA_DIR,'filesJSON',fold)
        path_new_json = os.path.join(path,'_annotations.coco.json')
        shutil.copy(json_path, path_new_json)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for imgs in data['images']:
            img_name = imgs['file_name']
            img_path = os.path.join(ROOT_DATA_DIR,'train',img_name)
            shutil.copy(img_path, path)
