import os
import json
import numpy as np
import cv2
import torch
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy
import shutil
import sys
import csv
from pathlib import Path

# Importações dos modelos de detecção
from Detectors.YOLOV8.DetectionsYolov8 import resultYOLO
from Detectors.YOLOV11.DetectionsYOLOV11 import ResultYOLOV11
from Detectors.YOLO26.DetectionsYOLO26 import ResultYOLO26
from Detectors.RetinaNet.DetectionsRetinaNet import ResultRetinaNet
try:
    from Detectors.FasterRCNN.inference import ResultFaster
except (FileNotFoundError, ModuleNotFoundError) as _faster_exc:
    ResultFaster = None
    _FASTER_IMPORT_ERROR = _faster_exc
#from Detectors.Detr.inference_image_detect import resultDetr
from sage import SageAggregator, detect_sage_dataset

# Constantes
LIMIAR_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "prediction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_CSV_PATH = RESULTS_DIR / "results.csv"
RESULTS_BY_CLASS_CSV_PATH = RESULTS_DIR / "resultsbyclass.csv"
COUNTING_CSV_PATH = RESULTS_DIR / "counting.csv"


def _resolve_tiling_mode(root: str, requested_mode: str = "auto") -> bool:
    """Return True if SAGE aggregation should be used based on the desired tiling mode."""
    normalized = (requested_mode or "auto").strip().lower()
    if normalized not in {"auto", "sage", "basic", "normal", "none"}:
        raise ValueError(f"Tiling mode inválido: {requested_mode}")
    if normalized == "sage":
        return True
    if normalized in {"basic", "normal", "none"}:
        return False
    return detect_sage_dataset(root)


def _has_filesjson(root: str) -> bool:
    return os.path.exists(os.path.join(root, "filesJSON"))


def _resolve_test_split(root: str, fold: str):
    if _has_filesjson(root):
        json_path = os.path.join(root, "filesJSON", f"{fold}_test.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Test JSON not found: {json_path}")
        images_dir = os.path.join(root, "train")
        return json_path, images_dir

    candidates = [
        ("test", os.path.join(root, "test", "_annotations.coco.json")),
        ("val", os.path.join(root, "val", "_annotations.coco.json")),
        ("valid", os.path.join(root, "valid", "_annotations.coco.json")),
    ]
    for split_name, json_path in candidates:
        if os.path.exists(json_path):
            images_dir = os.path.join(root, split_name if split_name != "valid" else "val")
            return json_path, images_dir

    raise FileNotFoundError(
        "Could not locate evaluation annotations for per-class metrics. "
        "Expected split JSONs in 'filesJSON' or '_annotations.coco.json' inside split folders."
    )


def _resolve_class_annotations(root: str) -> str:
    candidates = [
        os.path.join(root, "train", "_annotations.coco.json"),
        os.path.join(root, "val", "_annotations.coco.json"),
        os.path.join(root, "valid", "_annotations.coco.json"),
        os.path.join(root, "test", "_annotations.coco.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Unable to locate a COCO annotations file for class discovery in the dataset root."
    )

def print_to_file(line: str = '', file_path: Path = RESULTS_CSV_PATH, mode: str = 'a'):
    """Função para escrever uma linha em um arquivo."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        original_stdout = sys.stdout  # Salva a referência para a saída padrão original
        with file_path.open(mode) as f:
            sys.stdout = f  # Altera a saída padrão para o arquivo criado
            print(line)
            sys.stdout = original_stdout  # Restaura a saída padrão para o valor original
    except Exception as e:
        print(f"[ERRO] Falha ao escrever no arquivo {file_path}: {e}")

def generate_csv(data):
    """Gera um arquivo CSV com os dados fornecidos."""
    file_path = COUNTING_CSV_PATH
    headers = ['ml', 'fold', 'groundtruth', 'predicted', 'TP', 'FP', 'dif', 'fileName']
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open(mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            for row in data:
                writer.writerow(row)
    except Exception as e:
        print(f"[ERRO] Falha ao salvar CSV de contagem em {file_path}: {e}")

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

def process_predictions(ground_truth, predictions, classes, save_img, images_source, fold, model_name, cls, class_dict):
    """Processa as previsões e calcula métricas como TP, FP, precisão e recall."""
    ground_truth_list = []
    predict_list = []
    ground_truth_list_count = []
    predict_list_count = []
    data = []
    for key in ground_truth:
        if callable(images_source):
            img_path = images_source(key)
        else:
            img_path = os.path.join(images_source, key)
        image = cv2.imread(img_path) if img_path and os.path.exists(img_path) else None

        gt_items = [bbox for bbox in ground_truth[key] if int(bbox[-1]) == cls]
        pred_items = [bbox for bbox in predictions.get(key, []) if int(bbox[4]) == cls]

        gt_count = len(gt_items)
        pred_count = len(pred_items)
        ground_truth_list_count.append(gt_count)
        predict_list_count.append(pred_count)

        if image is not None:
            cv2.putText(image, f"GT: {gt_count}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
            cv2.putText(image, f"PRED: {pred_count}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

        true_positives = 0
        false_positives = 0
        matched_gt = set()

        for bbox_pred in pred_items:
            x1_max, y1_max = int(bbox_pred[0] + bbox_pred[2]), int(bbox_pred[1] + bbox_pred[3])
            best_iou = 0
            best_gt = None

            for i, bbox_gt in enumerate(gt_items):
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                iou = calculate_iou(bbox_pred[:4], bbox_gt[:4])
                if image is not None:
                    cv2.putText(image, str(classes[bbox_gt[-1]]), (int(bbox_gt[0]), int(bbox_gt[1] + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if iou >= IOU_THRESHOLD and iou > best_iou and i not in matched_gt:
                    best_iou = iou
                    best_gt = i

            if best_gt is not None:
                matched_gt.add(best_gt)
                gt_class = gt_items[best_gt][-1]

                ground_truth_list.append(1)
                predict_list.append(1)

                color = (0, 255, 0) if int(gt_class) == int(bbox_pred[4]) else (0, 0, 255)
                if image is not None:
                    cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (x1_max, y1_max), color, thickness=2)
                    cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), y1_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if int(gt_class) == int(bbox_pred[4]):
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if image is not None:
                    cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (x1_max, y1_max), (0, 0, 255), thickness=2)
                    cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), y1_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                ground_truth_list.append(0)  # Falso Positivo
                predict_list.append(1)
                false_positives += 1

        for i, bbox_gt in enumerate(gt_items):
            if i not in matched_gt:
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                if image is not None:
                    cv2.rectangle(image, (int(bbox_gt[0]), int(bbox_gt[1])), (x2_max, y2_max), (255, 0, 0), thickness=2)
                    cv2.putText(image, str(classes[bbox_gt[-1]]), (x2_max, y2_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                ground_truth_list.append(1)
                predict_list.append(0)  # Falso Negativo

        precision = round(true_positives / (true_positives + false_positives), 3) if (true_positives + false_positives) > 0 else 0
        recall = round(true_positives / gt_count, 3) if gt_count > 0 else 0

        if image is not None:
            cv2.putText(image, f"P: {precision}", (5, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
            cv2.putText(image, f"R: {recall}", (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
            if save_img and (gt_count > 0 or pred_count > 0):
                try:
                    save_path = os.path.join(RESULTS_PATH, fold, model_name, class_dict[cls])
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_path = os.path.join(save_path, key)
                    cv2.imwrite(save_path, image)
                except Exception as e:
                    print(f"[ERRO] Falha ao salvar imagem de predição em {save_path}: {e}")

        data.append(
            {
                'ml': model_name,
                'fold': fold,
                'groundtruth': gt_count,
                'predicted': pred_count,
                'TP': true_positives,
                'FP': false_positives,
                'dif': int(gt_count - pred_count),
                'fileName': key,
            }
        )
    generate_csv(data)
    if not ground_truth_list_count or not predict_list_count:
        return ground_truth_list, predict_list, torch.tensor(0.0)

    ground_truth_list_count = torch.tensor(ground_truth_list_count)
    predict_list_count = torch.tensor(predict_list_count)

    pearson = PearsonCorrCoef()
    r = pearson(predict_list_count.float(), ground_truth_list_count.float())
    return ground_truth_list, predict_list, r

def compute_metrics(preds, targets, num_classes=1):
    """Calcula métricas de classificação como precisão, recall, F1-score e acurácia."""
    if not preds or not targets:
        return 0.0, 0.0, 0.0
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    precision = BinaryPrecision()(preds, targets)
    recall = BinaryRecall()(preds, targets)
    fscore = BinaryF1Score()(preds, targets)
    accuracy = BinaryAccuracy()(preds, targets)

    return precision.item(), recall.item(), fscore.item()

def generate_results(root, fold, model, model_name, save_imgs, tiling_mode="auto"):
    """Gera resultados para um modelo específico e salva as métricas por classe."""
    test_json_path, tile_images_dir = _resolve_test_split(root, fold)
    use_sage = _resolve_tiling_mode(root, tiling_mode)

    if use_sage:
        aggregator = SageAggregator(root, fold)
        classes_dict = aggregator.classes_dict
        ground_truth = None
        predictions = None
    else:
        annotations_path = _resolve_class_annotations(root)
        classes_dict = get_classes(annotations_path)
        ground_truth = {}
        predictions = {}

    coco_test = load_dataset(test_json_path)
    for image in coco_test:
        file_name = image['file_name']
        image_path = os.path.join(tile_images_dir, file_name)
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Imagem de teste não encontrada em {image_path}")

        if model_name == "YOLOV8":
            result = resultYOLO.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "YOLO26":
            result = ResultYOLO26.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "YOLOV11":
            result = ResultYOLOV11.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "Faster":
            if ResultFaster is None:
                raise RuntimeError(
                    "FasterRCNN indisponível. Verifique dataset/configuração. "
                    f"Detalhes: {_FASTER_IMPORT_ERROR}"
                )
            print(image_path)
            result = ResultFaster.resultFaster(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "RetinaNet":
            result = ResultRetinaNet.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "Detr":
            print(image_path)
            result = []
        else:
            raise ValueError(f"Modelo de inferência não suportado: {model_name}")

        if use_sage:
            aggregator.add_tile_prediction(file_name, result)
        else:
            gt_items = []
            for i, bbox in enumerate(image['annotations']['bboxes']):
                x1, y1, width, height = bbox
                label = image['annotations']['labels'][i]
                gt_items.append([x1, y1, width, height, label])
            ground_truth[file_name] = gt_items
            predictions[file_name] = result

    if use_sage:
        ground_truth, predictions, image_source, classes_dict = aggregator.finalize()
    else:
        image_source = tile_images_dir

    for cls in sorted(classes_dict.keys()):
        if cls == 0:
            continue

        ground_truth_map = []
        predictions_map = []
        ground_truth_counts = []
        prediction_counts = []

        for key in ground_truth:
            gt_filtered = [bbox for bbox in ground_truth[key] if int(bbox[-1]) == cls]
            pred_filtered = [bbox for bbox in predictions.get(key, []) if int(bbox[4]) == cls]

            gt_boxes = [xywh_to_xyxy(values[:4]) for values in gt_filtered]
            pred_boxes = [xywh_to_xyxy(values[:4]) for values in pred_filtered]
            scores = [values[5] for values in pred_filtered]

            ground_truth_map.append({
                "boxes": torch.tensor(gt_boxes),
                "labels": torch.tensor([cls] * len(gt_boxes))
            })
            predictions_map.append({
                "boxes": torch.tensor(pred_boxes),
                "scores": torch.tensor(scores),
                "labels": torch.tensor([cls] * len(pred_boxes))
            })

            ground_truth_counts.append(len(gt_filtered))
            prediction_counts.append(len(pred_filtered))

        metric = MeanAveragePrecision()
        metric.update(predictions_map, ground_truth_map)
        result_map = metric.compute()

        mAP = result_map.get("map", torch.tensor(0.0))
        mAP50 = result_map.get("map_50", torch.tensor(0.0))
        mAP75 = result_map.get("map_75", torch.tensor(0.0))

        gt_counts_tensor = torch.tensor(ground_truth_counts, dtype=torch.float32)
        pred_counts_tensor = torch.tensor(prediction_counts, dtype=torch.float32)
        if gt_counts_tensor.numel() == 0:
            mae = torch.tensor(0.0)
            rmse = torch.tensor(0.0)
        else:
            mae = MeanAbsoluteError()(pred_counts_tensor, gt_counts_tensor)
            rmse = MeanSquaredError(squared=False)(pred_counts_tensor, gt_counts_tensor)

        ground_truth_list, predict_list, r = process_predictions(
            ground_truth,
            predictions,
            classes_dict,
            save_imgs,
            image_source,
            fold,
            model_name,
            cls,
            classes_dict,
        )

        precision_cls, recall_cls, fscore_cls = compute_metrics(predict_list, ground_truth_list)

        create_csv(
            model_name,
            fold,
            classes_dict,
            cls,
            mAP.item(),
            mAP50.item(),
            mAP75.item(),
            mae.item(),
            rmse.item(),
            precision_cls,
            recall_cls,
            fscore_cls,
            r.item(),
        )
def create_csv(selected_model, fold,classes_dict,cls,mAP, mAP50, mAP75, MAE, RMSE, precision, recall, fscore, r ):
    """Cria um arquivo CSV com os resultados das métricas."""
    results_path = RESULTS_BY_CLASS_CSV_PATH
    try:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = results_path.exists()
        with results_path.open(mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["ml", "fold", 'classes',"mAP", "mAP50", "mAP75", "MAE", "RMSE",'r',"precision", "recall", "fscore"])
            writer.writerow([selected_model, fold, classes_dict[cls] ,mAP, mAP50, mAP75, MAE, RMSE, r, precision, recall, fscore])
        print(f"[INFO] Resultados por classe salvos com sucesso em {results_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar resultados por classe em {results_path}: {e}")
