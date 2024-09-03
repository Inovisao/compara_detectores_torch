import numpy as np
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
from ultralytics import YOLO
import cv2
import json

# Classe Usada para detectar os objetos
MOSTRAIMAGE = False
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
        if MOSTRAIMAGE:
            CLASS_NAMES_DICT = model.model.names
            CLASS_ID = [0]
            box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=0, text_scale=1)
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            imagem_com_retangulo = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

            cv2.imshow('Quadrados',imagem_com_retangulo)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        lista = [] # Lista com todas as detecções da Imagem
        classes_usadas = [] # Salva as classes usadas
        JsonData = '../dataset/all/train/_annotations.coco.json'
        with open(JsonData) as f:
            data = json.load(f)
        ann_ids = []
        for anotation in data["annotations"]:
            if anotation["category_id"] not in ann_ids:
                ann_ids.append(anotation["category_id"])

        for j in ann_ids:
            lista1 = []
            classes = j - 1
            for i in range(len(detections.xyxy)):
                if detections.class_id[i] == classes:
                    classes_usadas.append(classes)
                    lista1.append([detections.xyxy[i][0],detections.xyxy[i][1],detections.xyxy[i][2],detections.xyxy[i][3],detections.confidence[i]])
            
            lista.append(np.array(lista1,dtype='float32'))

        return lista
