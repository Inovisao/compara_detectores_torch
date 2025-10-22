import numpy as np
from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator
from ultralytics import YOLO
import cv2
import json

# Classe Usada para detectar os objetos
MOSTRAIMAGE = False
    # Função do detector da YOLOV8
def xyxy_to_xywh(boxes: list)-> list:
    """
    Converte caixas delimitadoras no formato xyxy para xywh.

    :param boxes: Lista ou array de caixas no formato [x_min, y_min, x_max, y_max, conf, class].
    :return: Lista de caixas no formato [x, y, w, h, conf, class].
    """
    coco_boxes = []

    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([x_min, y_min, w, h, conf, cls])
    return coco_boxes
    
class resultYOLO:

    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack((
            detections.xyxy,
            detections.confidence[:, np.newaxis]
        ))
    # Função onde passamos a imagem e o modelo treinado
    def result(frame,modelName,LIMIAR_THRESHOLD):
        yolo_box = []
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

        for i,bbox in enumerate(detections.xyxy):
            if detections.confidence[i] > LIMIAR_THRESHOLD:
                yolo_box.append([int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]),int(detections.class_id[i])+1,detections.confidence[i]])


        coco_boxes = xyxy_to_xywh(yolo_box)

        return coco_boxes

