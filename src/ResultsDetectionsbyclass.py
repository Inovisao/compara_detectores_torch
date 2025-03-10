import os
import json
import numpy as np
import cv2
import torch
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy
import shutil
import sys
import csv

# Importações dos modelos de detecção
from Detectors.YOLOV8.DetectionsYolov8 import resultYOLO
from Detectors.FasterRCNN.inference import ResultFaster
from Detectors.Detr.inference_image_detect import resultDetr
# Constantes
LIMIAR_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
RESULTS_PATH = os.path.join("..", "results", "prediction")
os.makedirs(RESULTS_PATH, exist_ok=True)  # Garante que a pasta existe

def print_to_file(line='', file_path='../results/results.csv', mode='a'):
    """Função para escrever uma linha em um arquivo."""
    original_stdout = sys.stdout  # Salva a referência para a saída padrão original
    with open(file_path, mode) as f:
        sys.stdout = f  # Altera a saída padrão para o arquivo criado
        print(line)
        sys.stdout = original_stdout  # Restaura a saída padrão para o valor original

def generate_csv(data):
    """Gera um arquivo CSV com os dados fornecidos."""
    file_name = '../results/counting.csv'
    headers = ['ml', 'fold', 'groundtruth', 'predicted', 'TP', 'FP', 'dif', 'fileName']
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        for row in data:
            writer.writerow(row)

def get_classes(json_path):
    """Extrai as classes de um arquivo JSON no formato COCO."""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {category["id"]: category["name"] for category in data["categories"]}

def load_dataset(fold_path):
    """Carrega o dataset a partir de um arquivo JSON."""
    with open(fold_path, 'r') as f:
        data = json.load(f)

    image_info_list = []
    for image in data['images']:
        image_id = image['id']
        file_name = image['file_name']
        annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]
        
        bboxes = [annotation['bbox'] for annotation in annotations]
        labels = [annotation['category_id'] for annotation in annotations]
        
        annotation_info = {
            'bboxes': bboxes,
            'labels': labels,
            'bboxes_ignore': np.array([]),
            'masks': [[]],
            'seg_map': file_name
        }
        
        image_info = {
            'image_id': image_id,
            'file_name': file_name,
            'annotations': annotation_info
        }
        image_info_list.append(image_info)

    return image_info_list

def xywh_to_xyxy(bbox):
    """Converte bbox de formato (x, y, w, h) para (x_min, y_min, x_max, y_max)."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def calculate_iou(box1, box2):
    """Calcula a interseção sobre união (IoU) entre duas bboxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def process_predictions(ground_truth, predictions, classes, save_img, root, fold, model_name,cls,class_dict):
    """Processa as previsões e calcula métricas como TP, FP, precisão e recall."""
    ground_truth_list = []
    predict_list = []
    data = []
    for key in predictions:
        img_path = os.path.join(root, "train", key)
        image = cv2.imread(img_path)

        gt_count = len(ground_truth[key][cls])
        pred_count = len(predictions[key][cls])

        cv2.putText(image, f"GT: {gt_count}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
        cv2.putText(image, f"PRED: {pred_count}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

        true_positives = 0
        false_positives = 0
        matched_gt = set()

        for bbox_pred in predictions[key][cls]:
            x1_max, y1_max = int(bbox_pred[0] + bbox_pred[2]), int(bbox_pred[1] + bbox_pred[3])
            best_iou = 0
            best_gt = None

            for i, bbox_gt in enumerate(ground_truth[key][cls]):
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                iou = calculate_iou(bbox_pred[:4], bbox_gt[:4])

                if iou >= IOU_THRESHOLD and iou > best_iou and i not in matched_gt:
                    best_iou = iou
                    best_gt = i

            if best_gt is not None:
                matched_gt.add(best_gt)
                gt_class = ground_truth[key][cls][best_gt][-1]

                ground_truth_list.append(1)
                predict_list.append(1)

                color = (0, 255, 0) if gt_class == bbox_pred[4] else (0, 0, 255)
                cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (x1_max, y1_max), color, thickness=2)
                cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), y1_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if gt_class == bbox_pred[4]:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (x1_max, y1_max), (0, 0, 255), thickness=2)
                cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), y1_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                ground_truth_list.append(0)  # Falso Positivo
                predict_list.append(1)
                false_positives += 1

        for i, bbox_gt in enumerate(ground_truth[key][cls]):
            if i not in matched_gt:
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                cv2.rectangle(image, (int(bbox_gt[0]), int(bbox_gt[1])), (x2_max, y2_max), (255, 0, 0), thickness=2)
                cv2.putText(image, str(classes[bbox_gt[-1]]), (x2_max, y2_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                ground_truth_list.append(1)
                predict_list.append(0)  # Falso Negativo

        precision = round(true_positives / (true_positives + false_positives), 3) if (true_positives + false_positives) > 0 else 0
        recall = round(true_positives / gt_count, 3) if gt_count > 0 else 0

        cv2.putText(image, f"P: {precision}", (5, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
        cv2.putText(image, f"R: {recall}", (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
        
        if save_img:
            save_path = os.path.join(RESULTS_PATH, fold,model_name,class_dict[cls])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, key)
            cv2.imwrite(save_path, image)
        data.append({'ml': model_name, 'fold': fold, 'groundtruth': gt_count, 'predicted': pred_count, 'TP': true_positives, 'FP': false_positives, 'dif': int(gt_count - pred_count), 'fileName': key})
    generate_csv(data)
    return ground_truth_list, predict_list

def compute_metrics(preds, targets, num_classes=1):
    """Calcula métricas de classificação como precisão, recall, F1-score e acurácia."""
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    precision = BinaryPrecision()(preds, targets)
    recall = BinaryRecall()(preds, targets)
    fscore = BinaryF1Score()(preds, targets)
    accuracy = BinaryAccuracy()(preds, targets)

    
    return precision.item(), recall.item(), fscore.item(), accuracy.item()

def generate_results(root, fold, model, model_name, save_imgs):
    """Gera resultados para um modelo específico e salva as métricas."""
    test_json_path = os.path.join(root, 'filesJSON', fold + '_test.json')
    annotations_path = os.path.join(root, 'train', '_annotations.coco.json')

    classes_dict = get_classes(annotations_path)
    coco_test = load_dataset(test_json_path)
    predictions = {}
    ground_truth = {}
    for image in coco_test:
        ground_truth_list_class = []
        for cls in classes_dict:
            ground_truth_list = []

            for i, bbox in enumerate(image['annotations']['bboxes']):
                x1, y1, width, height = bbox
                label = image["annotations"]['labels'][i]
                if label == cls:
                    ground_truth_list.append([x1, y1, width, height, label])
            ground_truth_list_class.append(ground_truth_list)

        ground_truth[image['file_name']] = ground_truth_list_class
        image_path = os.path.join(root, 'train', image['file_name'])

    
        frame = cv2.imread(image_path)
        result_list_classe = []
        if model_name == "YOLOV8":
            result = resultYOLO.result(frame, model,LIMIAR_THRESHOLD)
        elif model_name == "Faster":
            print(image_path)
            result = ResultFaster.resultFaster(frame,model,LIMIAR_THRESHOLD)
        elif model_name == "Detr":
            print(image_path)
            result =   resultDetr(fold,frame,LIMIAR_THRESHOLD)
        for cls in classes_dict:
            result_list = []
            for bbox in result:
                if bbox[4] == cls:
                    result_list.append(bbox)
            result_list_classe.append(result_list)
        predictions[image['file_name']] = result_list_classe


    for cls in classes_dict:
        if cls == 0:
            continue
        ground_truth_map = []
        predictions_map = []

        for key in ground_truth:
            bbox_list = []
            label_list = []
            for values in ground_truth[key][cls]:
                bbox = xywh_to_xyxy(values[:4])
                bbox_list.append(bbox)
                label_list.append(values[-1])
            ground_truth_map.append({"boxes": torch.tensor(bbox_list), "labels": torch.tensor(label_list)})
 
        for key in predictions:
            bbox_list = []
            label_list = []
            score_list = []
            for values in predictions[key][cls]:
                bbox = xywh_to_xyxy(values[:4])
                bbox_list.append(bbox)
                label_list.append(values[4])
                score_list.append(values[5])
            predictions_map.append({"boxes": torch.tensor(bbox_list), "scores": torch.tensor(score_list), "labels": torch.tensor(label_list)})

        metric = MeanAveragePrecision()
        metric.update(predictions_map, ground_truth_map)
        result_map = metric.compute()

        mAP = result_map["map"]
        mAP50 = result_map["map_50"]
        mAP75 = result_map["map_75"]

        ground_truth_counts = []
        for key in ground_truth: 
            ground_truth_counts.append(len(ground_truth[key][cls]))
        ground_truth_counts = torch.tensor(ground_truth_counts)

        prediction_counts = []
        for key in predictions:
            prediction_counts.append(len(predictions[key][cls]))
        prediction_counts = torch.tensor(prediction_counts)

        mae = MeanAbsoluteError()(prediction_counts, ground_truth_counts)
        rmse = MeanSquaredError(squared=False)(prediction_counts, ground_truth_counts)
        
   
        ground_truth_list, predict_list = process_predictions(ground_truth, predictions, classes_dict, save_imgs, root, fold, model_name,cls,classes_dict)

        # num_classes = len(classes_dict)
        precision, recall, fscore, accuracy = compute_metrics(predict_list, ground_truth_list)
        print("_"*30)
        print(classes_dict[cls])
        print("_"*30)
        print('mAP:',mAP)
        print("mAP50:",mAP50)
        print("mAP75:",mAP75)
        print("mae:",mae)
        print("rmse:",rmse)
        print("precision:",precision)
        print("recall:",recall)
        print("fscore:",fscore)
        print("accuracy:",accuracy)
        print("_"*30)
generate_results(os.path.join('..', 'dataset','all'), 'fold_1', '/home/pedroeduardo/Documentos/compara_detectores_torch/src/model_checkpoints/fold_1/YOLOV8/train/weights/best.pt', 'YOLOV8', True)

# def create_csv(selected_model, fold, root, model_path, save_imgs):
#     """Cria um arquivo CSV com os resultados das métricas."""
#     mAP, mAP50, mAP75, MAE, RMSE, precision, recall, fscore, accuracy = generate_results(root, fold, model_path, selected_model, save_imgs)

#     results_path = os.path.join('..', 'results', 'results.csv')
#     file_exists = os.path.isfile(results_path)

#     with open(results_path, mode="a", newline="") as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             writer.writerow(["ml", "fold", "mAP", "mAP50", "mAP75", "MAE", "RMSE", "accuracy", "precision", "recall", "fscore"])
#         writer.writerow([selected_model, fold, mAP, mAP50, mAP75, MAE, RMSE, accuracy, precision, recall, fscore])