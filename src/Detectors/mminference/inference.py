import cv2
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector


def xyxy_to_xywh(boxes):
    """
    Converte caixas delimitadoras do formato [x_min, y_min, x_max, y_max, conf, class] para [x, y, w, h, conf, class].
    """
    coco_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([x_min, y_min, w, h, conf, cls])
    return coco_boxes


def detectar_objetos_cv2(config_path: str, checkpoint_path: str, frame, device: str = 'cuda'):
    """
    Carrega um modelo treinado no MMDetection e faz inferência em um frame usando OpenCV.

    Parâmetros:
    - config_path (str): Caminho para o arquivo de configuração do modelo (.py).
    - checkpoint_path (str): Caminho para o arquivo de pesos do modelo (.pth).
    - frame (numpy.ndarray): Frame de entrada para inferência.
    - device (str): Dispositivo para executar a inferência ('cuda' ou 'cpu').

    Retorno:
    - model: Modelo carregado.
    - resultados: Lista de caixas detectadas por classe.
    - frame: Imagem processada.
    """
    device = device if torch.cuda.is_available() else 'cpu'

    # Inicializa o modelo
    model = init_detector(config_path, checkpoint_path, device=device)

    # Faz a inferência
    resultados = inference_detector(model, frame)

    return model, resultados, frame

def runMMdetection(model_name, frame, LIMIAR_THRESHOLD=0.8, NMS_THRESHOLD=0.5):

    config_file = model_name[0:-10]+model_name.split('/')[2]+'.py'  

    model, resultados, image = detectar_objetos_cv2(config_file, model_name, frame)

    boxes = []
    scores = []
    classes = []

    # Coleta todas as caixas detectadas, suas pontuações e classes
    for class_id, result in enumerate(resultados):  # class_id representa a classe detectada
        for box in result:
            if len(box) < 5:
                continue
            x1, y1, x2, y2, score = box
            if score > LIMIAR_THRESHOLD:  
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(score))
                classes.append(class_id)  # Guarda a classe correspondente

    # Converte para formato esperado pelo OpenCV
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), LIMIAR_THRESHOLD, NMS_THRESHOLD)

    # # Desenha apenas as caixas aprovadas pelo NMS
    # if len(indices) > 0:
    #     for i in indices.flatten():  # Garante que os índices sejam tratados corretamente
    #         x1, y1, x2, y2 = boxes[i]
    #         class_id = classes[i]  # Obtém a classe associada à detecção

    #         # Cor e rótulo baseados na classe
    #         color = (0, 255, 0)  # Pode ser ajustado por classe se necessário
    #         label = f"Classe {class_id}: {scores[i]:.2f}"

    #         # Desenha a caixa e o rótulo
    #         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    detections = []
    for i,bbox in enumerate(boxes):
        detections.append([int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),classes[i],scores[i]])
    coco_box = xyxy_to_xywh(detections)
    return coco_box