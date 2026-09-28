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
    Converte caixas [x1, y1, x2, y2, class_id, confidence] para xywh.

    :param boxes: Lista ou array de caixas no formato [x1, y1, x2, y2, class_id, confidence].
    :return: Lista de caixas no formato [x, y, w, h, class_id, confidence].
    """
    coco_boxes = []

    for box in boxes:
        x_min, y_min, x_max, y_max, class_id, confidence = box
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([x_min, y_min, w, h, class_id, confidence])
    return coco_boxes
    
class resultYOLO:

    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack((
            detections.xyxy,
            detections.confidence[:, np.newaxis]
        ))
    # Função onde passamos a imagem e o modelo treinado
    def result(frame, modelName, LIMIAR_THRESHOLD, class_ids=None):
        yolo_box = []
        model_is_loaded = hasattr(modelName, 'predict')
        model = modelName if model_is_loaded else YOLO(modelName)
        if not model_is_loaded and hasattr(model, 'fuse'):
            model.fuse()

        results = model.predict(source=frame, conf=LIMIAR_THRESHOLD, verbose=False)
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
                class_index = int(detections.class_id[i])
                if class_ids is None:
                    class_id = class_index + 1
                elif isinstance(class_ids, dict):
                    class_names = model.names
                    class_name = class_names[class_index]
                    class_id = class_ids[class_name]
                else:
                    class_id = class_ids[class_index]
                yolo_box.append([
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    class_id, float(detections.confidence[i])
                ])


        coco_boxes = xyxy_to_xywh(yolo_box)

        return coco_boxes

