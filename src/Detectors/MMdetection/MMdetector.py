from mmdet.apis import DetInferencer
import numpy as np

# Função para retornar as detecções de objetos
def resultMM(img,checkpoint,config):

    device = 'cuda:0' # Seleciona o Dispositivo CPU O GPU
    # Initialize the DetInferencer
    inferencer = DetInferencer(config, checkpoint, device) # Ira carregar o modelo

    results = inferencer(img) # pega o objetos detectados da imagem

    lista = [] 
    classes_usadas = []
    classes = [0]
    # Loop apra fazer o array com todos os objetos detectados
    for j in classes:
        lista1 = []
        for result in results['predictions']:
            for i in range(len(result['labels'])):
                if result['labels'][i] == j:
                    score = result['scores'][i]
                    bbox = result['bboxes'][i]
                    classes_usadas.append(j)
                    lista1.append([bbox[0],bbox[1],bbox[2],bbox[3],score])

        lista.append(np.array(lista1,dtype='float32'))
    return lista