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
from Detectors.YOLOV5_TPH.DetectionsYOLOV5TPH import ResultYOLOV5TPH
from Detectors.YOLOV8.DetectionsYolov8 import resultYOLO
from Detectors.YOLOV11.DetectionsYOLOV11 import ResultYOLOV11
from Detectors.YOLO26.DetectionsYOLO26 import ResultYOLO26
from Detectors.RetinaNet.DetectionsRetinaNet import ResultRetinaNet
from Detectors.SSDLite.DetectionsSSDLite import ResultSSDLite
_FASTER_IMPORT_ERROR = None
try:
    from Detectors.FasterRCNN.inference import ResultFaster
    from Detectors.FasterRCNN.geradataset import geredata as faster_geradata
    from Detectors.FasterRCNN import config as faster_config
except (FileNotFoundError, ModuleNotFoundError) as _faster_exc:
    ResultFaster = None
    faster_geradata = None
    faster_config = None
    _FASTER_IMPORT_ERROR = _faster_exc
from Detectors.Detr.DetectionsDetr import ResultDetr
from Detectors.ViT.DetectionsViT import ResultViT
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
    """Return the annotation path and image directory for evaluation."""
    if _has_filesjson(root):
        json_path = os.path.join(root, "filesJSON", f"{fold}_test.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Test JSON not found: {json_path}")
        images_dir = os.path.join(root, "train")
        return json_path, images_dir

    # Tiled dataset layout: use the dedicated split folders
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
        "Could not locate evaluation annotations. "
        "Expected 'filesJSON' split files or '_annotations.coco.json' inside test/val folders."
    )


def _resolve_class_annotations(root: str, fold=None) -> str:
    files_json_dir = os.path.join(root, "filesJSON")
    if os.path.exists(files_json_dir):
        candidates = []
        if fold:
            candidates.extend(
                os.path.join(files_json_dir, f"{fold}_{split}.json")
                for split in ("train", "val", "test")
            )
        candidates.extend(
            str(path)
            for path in sorted(Path(files_json_dir).glob("fold_*_*.json"))
        )
        for path in candidates:
            if os.path.exists(path):
                return path

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


def _configure_faster_inference(root: str, fold: str) -> None:
    """Ensure the FasterRCNN config matches the current dataset for inference."""
    if ResultFaster is None or faster_config is None or faster_geradata is None:
        return
    dataset_config = faster_geradata(fold, Path(root))
    faster_config.configure_dataset(
        dataset_config.train_dir,
        dataset_config.train_annotations,
        dataset_config.val_dir,
        dataset_config.val_annotations,
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

def process_predictions(ground_truth, predictions, classes, save_img, images_source, fold, model_name):
    """Processa as previsões e calcula métricas como TP, FP, precisão e recall."""
    ground_truth_list = []
    predict_list = []
    ground_truth_list_count = []
    predict_list_count = []
    data = []
    for key in predictions:
        if callable(images_source):
            img_path = images_source(key)
        else:
            img_path = os.path.join(images_source, key)
        image = cv2.imread(img_path) if img_path and os.path.exists(img_path) else None

        gt_count = len(ground_truth[key])
        pred_count = len(predictions[key])
        ground_truth_list_count.append(gt_count)
        predict_list_count.append(pred_count)
        if image is not None:
            cv2.putText(image, f"GT: {gt_count}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
            cv2.putText(image, f"PRED: {pred_count}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

        true_positives = 0
        false_positives = 0
        matched_gt = set()

        for bbox_pred in predictions[key]:
            x1_max, y1_max = int(bbox_pred[0] + bbox_pred[2]), int(bbox_pred[1] + bbox_pred[3])
            best_iou = 0
            best_gt = None

            for i, bbox_gt in enumerate(ground_truth[key]):
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                iou = calculate_iou(bbox_pred[:4], bbox_gt[:4])
                if image is not None:
                    cv2.putText(image, str(classes[bbox_gt[-1]]), (int(bbox_gt[0]), int(bbox_gt[1]+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if iou >= IOU_THRESHOLD and iou > best_iou and i not in matched_gt:
                    best_iou = iou
                    best_gt = i

            if best_gt is not None:
                matched_gt.add(best_gt)
                gt_class = ground_truth[key][best_gt][-1]

                ground_truth_list.append(gt_class)
                predict_list.append(bbox_pred[4])

                color = (0, 255, 0) if gt_class == bbox_pred[4] else (0, 0, 255)
                if image is not None:
                    cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (int(x1_max), int(y1_max)), color, thickness=2)
                    cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), int(y1_max)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if gt_class == bbox_pred[4]:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if image is not None:
                    cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (int(x1_max), int(y1_max)), (0, 0, 255), thickness=2)
                    cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), int(y1_max)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                ground_truth_list.append(0)  # Falso Positivo
                predict_list.append(bbox_pred[4])
                false_positives += 1

        for i, bbox_gt in enumerate(ground_truth[key]):
            if i not in matched_gt:
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                if image is not None:
                    cv2.rectangle(image, (int(bbox_gt[0]), int(bbox_gt[1])), (x2_max, y2_max), (255, 0, 0), thickness=2)
                    cv2.putText(image, str(classes[bbox_gt[-1]]), (x2_max, y2_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                ground_truth_list.append(bbox_gt[-1])
                predict_list.append(0)  # Falso Negativo

        precision = round(true_positives / (true_positives + false_positives), 3) if (true_positives + false_positives) > 0 else 0
        recall = round(true_positives / gt_count, 3) if gt_count > 0 else 0

        if image is not None:
            cv2.putText(image, f"P: {precision}", (5, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
            cv2.putText(image, f"R: {recall}", (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
        
        if save_img and image is not None:
            try:
                save_path = os.path.join(RESULTS_PATH, fold,model_name,'all_classes')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path, key)
                cv2.imwrite(save_path, image)
            except Exception as e:
                print(f"[ERRO] Falha ao salvar imagem de predição em {save_path}: {e}")
        data.append({'ml': model_name, 'fold': fold, 'groundtruth': gt_count, 'predicted': pred_count, 'TP': true_positives, 'FP': false_positives, 'dif': int(gt_count - pred_count), 'fileName': key})
    generate_csv(data)
    if not ground_truth_list_count or not predict_list_count:
        return ground_truth_list, predict_list, torch.tensor(0.0)

    ground_truth_list_count = torch.tensor(ground_truth_list_count)
    predict_list_count = torch.tensor(predict_list_count)

    pearson = PearsonCorrCoef()
    r = pearson(predict_list_count.float(), ground_truth_list_count.float())
    return ground_truth_list, predict_list,r

def compute_metrics(preds, targets, num_classes=1):
    """Calcula métricas de classificação como precisão, recall, F1-score e acurácia."""
    if not preds or not targets:
        return 0.0, 0.0, 0.0
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    
    if num_classes <= 2:
        precision = BinaryPrecision()(preds, targets)
        recall = BinaryRecall()(preds, targets)
        fscore = BinaryF1Score()(preds, targets)
        accuracy = BinaryAccuracy()(preds, targets)
    else:
        precision = MulticlassPrecision(num_classes=num_classes, average='macro')(preds, targets)
        recall = MulticlassRecall(num_classes=num_classes, average='macro')(preds, targets)
        fscore = MulticlassF1Score(num_classes=num_classes, average='macro')(preds, targets)
        accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')(preds, targets)
    

            # Para multiclasse, converter logits para probabilidades com softmax
        #preds_prob = preds.float().softmax(dim=1).argmax(dim=1)  # Pegando a classe mais provável

    return precision.item(), recall.item(), fscore.item()

def generate_results(root, fold, model, model_name, save_imgs, tiling_mode="auto"):
    """Gera resultados para um modelo específico e salva as métricas."""
    if model_name == "Faster":
        _configure_faster_inference(root, fold)

    test_json_path, tile_images_dir = _resolve_test_split(root, fold)
    use_sage = _resolve_tiling_mode(root, tiling_mode)

    if use_sage:
        aggregator = SageAggregator(root, fold)
        classes_dict = aggregator.classes_dict
        predictions = None
        ground_truth = None
    else:
        annotations_path = _resolve_class_annotations(root, fold)
        classes_dict = get_classes(annotations_path)
        predictions = {}
        ground_truth = {}

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
                    "FasterRCNN indisponível. Verifique se o dataset/configuração está completo. "
                    f"Detalhes: {_FASTER_IMPORT_ERROR}"
                )
            print(image_path)
            result = ResultFaster.resultFaster(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "YOLOV5_TPH":
            print(image_path)
            result = ResultYOLOV5TPH.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "RetinaNet":
            result = ResultRetinaNet.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "SSDLite":
            result = ResultSSDLite.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "Detr":
            print(image_path)
            result = ResultDetr.result(frame, model, LIMIAR_THRESHOLD)
        elif model_name == "ViT":
            print(image_path)
            result = ResultViT.result(frame, model, LIMIAR_THRESHOLD)
        else:
            raise ValueError(f"Modelo de inferência não suportado: {model_name}")

        if use_sage:
            aggregator.add_tile_prediction(file_name, result)
        else:
            gt_items = []
            for i, bbox in enumerate(image['annotations']['bboxes']):
                x1, y1, width, height = bbox
                label = image["annotations"]['labels'][i]
                gt_items.append([x1, y1, width, height, label])
            ground_truth[file_name] = gt_items
            predictions[file_name] = result

    if use_sage:
        ground_truth, predictions, image_source, classes_dict = aggregator.finalize()
    else:
        image_source = tile_images_dir

    class_ids = sorted(classes_dict.keys())
    class_to_index = {cls_id: idx for idx, cls_id in enumerate(class_ids)}

    ground_truth_map = []
    predictions_map = []

    for key in ground_truth:
        bbox_list = []
        label_list = []
        for values in ground_truth[key]:
            bbox = xywh_to_xyxy(values[:4])
            bbox_list.append(bbox)
            label_list.append(values[-1])
        ground_truth_map.append({"boxes": torch.tensor(bbox_list), "labels": torch.tensor(label_list)})

        preds_for_key = predictions.get(key, [])
        bbox_list_pred = []
        label_list_pred = []
        score_list_pred = []
        for values in preds_for_key:
            bbox = xywh_to_xyxy(values[:4])
            bbox_list_pred.append(bbox)
            label_list_pred.append(values[4])
            score_list_pred.append(values[5])
        predictions_map.append(
            {"boxes": torch.tensor(bbox_list_pred), "scores": torch.tensor(score_list_pred), "labels": torch.tensor(label_list_pred)}
        )

    metric = MeanAveragePrecision()
    metric.update(predictions_map, ground_truth_map)
    result_map = metric.compute()

    ground_truth_counts = []
    for key in ground_truth:
        count_classes = [0] * len(class_to_index)
        for bbox in ground_truth[key]:
            cls_id = int(bbox[-1])
            idx = class_to_index.get(cls_id)
            if idx is not None:
                count_classes[idx] += 1
        ground_truth_counts.append(count_classes)
    ground_truth_counts = torch.tensor(ground_truth_counts) if ground_truth_counts else torch.zeros((0, len(class_to_index)))

    prediction_counts = []
    for key in ground_truth:
        count_classes = [0] * len(class_to_index)
        preds_for_key = predictions.get(key, [])
        for bbox in preds_for_key:
            for gt_bbox in ground_truth[key]:
                iou = calculate_iou(bbox[:4], gt_bbox[:4])
                if iou >= IOU_THRESHOLD and int(bbox[4]) == int(gt_bbox[-1]):
                    idx = class_to_index.get(int(bbox[4]))
                    if idx is not None:
                        count_classes[idx] += 1
        prediction_counts.append(count_classes)
    prediction_counts = torch.tensor(prediction_counts) if prediction_counts else torch.zeros((0, len(class_to_index)))

    if prediction_counts.numel() == 0:
        pred_counts = torch.tensor([])
        gt_counts = torch.tensor([])
    else:
        pred_counts = prediction_counts.sum(dim=1)
        gt_counts = ground_truth_counts.sum(dim=1)

    mae = MeanAbsoluteError()(pred_counts, gt_counts) if pred_counts.numel() > 0 else torch.tensor(0.0)
    rmse = MeanSquaredError(squared=False)(pred_counts, gt_counts) if pred_counts.numel() > 0 else torch.tensor(0.0)

    mAP = result_map.get("map", torch.tensor(0.0))
    mAP50 = result_map.get("map_50", torch.tensor(0.0))
    mAP75 = result_map.get("map_75", torch.tensor(0.0))

    ground_truth_list, predict_list, r = process_predictions(
        ground_truth,
        predictions,
        classes_dict,
        save_imgs,
        image_source,
        fold,
        model_name,
    )

    num_classes = len(classes_dict)
    precision, recall, fscore = compute_metrics(predict_list, ground_truth_list, num_classes=num_classes)

    return mAP.item(), mAP50.item(), mAP75.item(), mae.item(), rmse.item(), precision, recall, fscore, r.item()

def create_csv(selected_model, fold, root, model_path, save_imgs, tiling_mode="auto"):
    """Cria um arquivo CSV com os resultados das métricas."""
    results_path = RESULTS_CSV_PATH
    try:
        mAP, mAP50, mAP75, MAE, RMSE, precision, recall, fscore, r = generate_results(
            root, fold, model_path, selected_model, save_imgs, tiling_mode=tiling_mode
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = results_path.exists()
        with results_path.open(mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["ml", "fold", "mAP", "mAP50", "mAP75", "MAE", "RMSE", "accuracy", "precision", "recall", "fscore"])
            writer.writerow([selected_model, fold, mAP, mAP50, mAP75, MAE, RMSE, r, precision, recall, fscore])
        print(f"[INFO] Resultados salvos com sucesso em {results_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar resultados em {results_path}: {e}")
