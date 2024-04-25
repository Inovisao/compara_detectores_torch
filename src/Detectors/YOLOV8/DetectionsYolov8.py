import numpy as np
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
from ultralytics import YOLO
import cv2


# Classe Usada para detectar os objetos
class resultYOLO:
    # Função do detector da YOLOV8
    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack((
            detections.xyxy,
            detections.confidence[:, np.newaxis]
        ))
    # Função onde passamos a imagem e o modelo treinado
    def result(frame,modelName):
        
        MODEL=modelName 
        model = YOLO(MODEL) # Lendo o modelo Treinado
        model.fuse()

        results = model(frame) # Ira ler a imagem e marcar os Objetos
        # Chama a função para facilitar a visualização dos objetos
        detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )

        lista = [] # Lista com todas as detecções da Imagem
        classes_usadas = [] # Salva as classes usadas
        classes = [0] # Numero de classes 
        # Loop para fazer a conversão da YOLOV8 para um arry gigante
        for j in classes:
            lista1 = []
            for i in range(len(detections.xyxy)):
                if detections.class_id[i] == j:
                    classes_usadas.append(j)
                    lista1.append([detections.xyxy[i][0],detections.xyxy[i][1],detections.xyxy[i][2],detections.xyxy[i][3],detections.confidence[i]])
            
            lista.append(np.array(lista1,dtype='float32'))

        return lista
