from mmdet.apis import DetInferencer
import json
import numpy as np

# Função para retornar as detecções de objetos
def resultMM(img,checkpoint,config):

    device = 'cuda:0' # Seleciona o Dispositivo CPU O GPU
    # Initialize the DetInferencer
    inferencer = DetInferencer(config, checkpoint, device) # Ira carregar o modelo

    results = inferencer(img) # pega o objetos detectados da imagem

    lista = [] 
    classes_usadas = []
    JsonData = '../dataset/all/train/_annotations.coco.json'
    with open(JsonData) as f:
        data = json.load(f)
    ann_ids = []
    for anotation in data["annotations"]:
        if anotation["category_id"] not in ann_ids:
            ann_ids.append(anotation["category_id"])
    for j in ann_ids:
        lista1 = []
        classe = j - 1
        for result in results['predictions']:
            for i in range(len(result['labels'])):
                if result['labels'][i] == classe:
                    score = result['scores'][i]
                    bbox = result['bboxes'][i]
                    classes_usadas.append(classe)
                    lista1.append([bbox[0],bbox[1],bbox[2],bbox[3],score])
        lista.append(np.array(lista1,dtype='float32'))
    return lista