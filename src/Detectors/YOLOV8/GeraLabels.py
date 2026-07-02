import json
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT_DATA_DIR = REPO_ROOT / 'dataset' / 'all'


def map_category_id_to_class_index(category_id, class_names):
    if len(class_names) == 1:
        return 0
    return max(0, int(category_id) - 1)


def normalize_bbox_to_yolo(bbox, image_width, image_height):
    x1 = float(bbox[0])
    y1 = float(bbox[1])
    width = float(bbox[2])
    height = float(bbox[3])

    x2 = x1 + width
    y2 = y1 + height

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    x_center_norm = max(0.0, min(x_center / image_width, 1.0))
    y_center_norm = max(0.0, min(y_center / image_height, 1.0))
    width_norm = max(0.0, min(width / image_width, 1.0))
    height_norm = max(0.0, min(height / image_height, 1.0))

    return [x_center_norm, y_center_norm, width_norm, height_norm]


# Função para criar o dataset para treinamento da YOLOV8
def CriarLabelsYOLOV8(fold):
    # Abre o arquivo Json onde temos as anotações de cada imagem
    with open(os.path.join(ROOT_DATA_DIR, 'train', '_annotations.coco.json'), 'r') as f:
        data = json.load(f)
    
    ann_ids = []
    for anotation in data["annotations"]:
        if anotation["category_id"] not in ann_ids:
            ann_ids.append(anotation["category_id"])
    Classe = []
    for category in data["categories"]:
        if category["id"] in ann_ids:
            Classe.append(category["name"])
    caminho_arquivo_yaml = os.path.join(ROOT_DATA_DIR, 'data.yaml')
    
    # Carregue o conteúdo do arquivo YAML
    with open(caminho_arquivo_yaml, 'r') as arquivo:
        conteudo = yaml.safe_load(arquivo)

    # Altere o conteúdo conforme necessário
    conteudo['nc'] = len(Classe)
    conteudo['names'] = Classe

    # Salve o conteúdo alterado de volta no arquivo YAML
    with open(caminho_arquivo_yaml, 'w') as arquivo:
        yaml.dump(conteudo, arquivo)

    if os.path.exists(os.path.join(ROOT_DATA_DIR,'YOLO')):
        shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'YOLO'))

    for c1 in ['train', 'test', 'valid']:
        for c2 in ['labels', 'images']:
            os.makedirs(os.path.join(ROOT_DATA_DIR, 'YOLO', c1, c2), exist_ok=True)

    caminhos = (os.listdir(os.path.join(ROOT_DATA_DIR,'filesJSON')))
    foldsUsadas = []
    #Pega o caminho do arquivo coco que esta sendo usada
    for caminho in caminhos:

        fold_check = caminho.split("_")[0] + "_" +caminho.split("_")[1]

        if str(fold_check) == str(fold):
            foldsUsadas.append(caminho)

    for Caminho in foldsUsadas:
        path = Caminho.split('_')[-1][0:-5]

        caminho = os.path.join(ROOT_DATA_DIR,'filesJSON',Caminho)
        # Lendo o arquivo JSON
        with open(caminho, 'r') as f:
            anotacaoDobras = json.load(f)
        # Exibindo os dados lidos
        NameFile = []
        imageID = []
        anotacao = {}
        idAnotcao = {}
        #Salva a lista de id das imagens
        for i in range(len(anotacaoDobras['annotations'])):
            imageID.append(anotacaoDobras['annotations'][i]['image_id'])

        #Faz a lista sem repetir os IDs
        imageID = (list(set(imageID)))

        #Cria um dicionario Com ID das imagens
        for i in imageID:
            anotacao[i] = []
            idAnotcao[i] = [] 
        #Ira pegar as anotações es classes de cada imagens e salvar em seus dicionarios
        for id in imageID:
            for i in range(len(anotacaoDobras['annotations'])):
                if anotacaoDobras['annotations'][i]['image_id'] == id:
                    anotacao[id].append(anotacaoDobras['annotations'][i]['bbox'])
                    category_id = map_category_id_to_class_index(
                        anotacaoDobras['annotations'][i]['category_id'],
                        Classe,
                    )
                    idAnotcao[id].append(category_id)
        #Salva o nome de cada imagem
        for i in imageID:
            for j in range(len(anotacaoDobras['images'])):
                if (anotacaoDobras['images'][j]['id']) == i:
                    NameFile.append(anotacaoDobras['images'][j]['file_name'])
        
        slectImage = 0
        #Converte as anotações para o formato da YOLOV8
        for id in imageID:
            linhas = []
            image_name = NameFile[slectImage]
            image_path = os.path.join(ROOT_DATA_DIR, 'train', image_name)
            with Image.open(image_path) as image_file:
                image_width, image_height = image_file.size

            for i in range(len(anotacao[id])):
                bbox = anotacao[id][i]
                normalized_bbox = normalize_bbox_to_yolo(bbox, image_width, image_height)
                class_id = int(idAnotcao[id][i])
                if class_id < 0:
                    continue
                if class_id >= len(Classe):
                    continue
                linhas.append(
                    f"{class_id} {normalized_bbox[0]:.6f} {normalized_bbox[1]:.6f} {normalized_bbox[2]:.6f} {normalized_bbox[3]:.6f}\n"
                )

            arq = image_name[0:-4] + '.txt'
            with open(arq, 'w') as arquivo:
                arquivo.writelines(linhas)
            slectImage += 1

            if path == 'val':
                path = 'valid'

            shutil.move(arq, os.path.join(ROOT_DATA_DIR, 'YOLO', path, 'labels'))
            shutil.copy(image_path, os.path.join(ROOT_DATA_DIR, 'YOLO', path, 'images'))