import os
import json
import numpy as np
import cv2
from Detectors.YOLOV8.DetectionsYolov8 import resultYOLO
from Detectors.FasterRCNN.inference import resultFaster
from Detectors.Detr.inference_image_detect import resultDetr
from Detectors.Retinanet.inference import resultRetinanet
import torch
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score,MulticlassAccuracy
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score,BinaryAccuracy

LIMIAR_IOU = 0.50
RESULTADOS_PATH = "Teste123"

os.makedirs(RESULTADOS_PATH, exist_ok=True)  # Garante que a pasta existe

def get_classes(path_json):
    with open(path_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    categories_dict = {category["id"]: category["name"] for category in data["categories"]}

    return categories_dict

def pegaDataset(fold):
  # Carrega o arquivo JSON
  with open(fold, 'r') as f:
      data = json.load(f)

  # Lista para armazenar as informações de cada imagem
  image_info_list = []

  # Itera sobre as imagens e suas anotações
  for image in data['images']:
      image_id = image['id']
      file_name = image['file_name']
      
      # Procura pelas anotações correspondentes à imagem atual
      annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]
      
      # Lista para armazenar as informações de anotação da imagem atual
      annotations_info = []
      category_id = []
      bbox = []
      imageid = []
      for annotation in annotations:
          category_id.append(annotation['category_id'])
          bbox.append(annotation['bbox'])
          # Cria o dicionário de informações da anotação
      annotation_info = {
          'bboxes': bbox,
          'labels': category_id,
          'bboxes_ignore': np.array([]),
          'masks': [[]],
          'seg_map': file_name
      }
          
      # Adiciona as informações da anotação à lista de informações de anotação
      
      # Cria o dicionário de informações da imagem
      image_info = {
          'image_id' : image_id,
          'file_name': file_name,
          'annotations': annotation_info
      }
      
      # Adiciona as informações da imagem à lista de informações de imagem
      image_info_list.append(image_info)

  # Imprime a lista de informações de imagem
  return image_info_list

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox[0],bbox[1],bbox[2],bbox[3]
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    return [x_min,y_min,x_max,y_max]

def calculate_iou(box1, box2):

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Converter para (x_min, y_min, x_max, y_max)
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # Calcular coordenadas da interseção
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calcular área da interseção
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Calcular áreas das bboxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calcular IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def get_list_classes(ground_truth, predict, classes):
    ground_truth_list = []
    predict_list = []

    for key in predict:
        img_path = f"/home/pedroeduardo/Documentos/compara_detectores_torch/dataset/all/train/{key}"
        image = cv2.imread(img_path)

        if image is None:
            print(f"Erro ao carregar a imagem: {img_path}")
            continue

        overlay = image.copy()  # Cópia para sobrepor caixas com transparência
        alpha = 0.6  # Grau de transparência

        ground_truth_values = len(ground_truth[key])
        predict_values = len(predict[key])

        cv2.putText(image, f"GT: {ground_truth_values}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
        cv2.putText(image, f"PRED: {predict_values}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

        cont_TP = 0
        cont_FP = 0
        matched_gt = set()  # Guarda os GTs já correspondidos para evitar contagem duplicada

        for bbox_pred in predict[key]:  
            x1_max, y1_max = int(bbox_pred[0] + bbox_pred[2]), int(bbox_pred[1] + bbox_pred[3])
            best_iou = 0
            best_gt = None

            for i, bbox_gt in enumerate(ground_truth[key]):  
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                iou = calculate_iou(bbox_pred[:4], bbox_gt[:4])

                if iou >= LIMIAR_IOU and iou > best_iou and i not in matched_gt:
                    best_iou = iou
                    best_gt = i

            if best_gt is not None:  # Predição correspondente a um GT
                matched_gt.add(best_gt)
                gt_class = ground_truth[key][best_gt][-1]

                ground_truth_list.append(gt_class)
                predict_list.append(bbox_pred[4])

                color = (0, 255, 0) if gt_class == bbox_pred[4] else (0, 0, 255)
                cv2.rectangle(overlay, (int(bbox_pred[0]), int(bbox_pred[1])), (x1_max, y1_max), color, thickness=2)
                cv2.putText(overlay, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), y1_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if gt_class == bbox_pred[4]:
                    cont_TP += 1
                else:
                    cont_FP += 1
            else:  # Nenhum GT compatível -> Falso Positivo
                cv2.rectangle(overlay, (int(bbox_pred[0]), int(bbox_pred[1])), (x1_max, y1_max), (0, 0, 255), thickness=2)
                cv2.putText(overlay, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), y1_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                ground_truth_list.append(0)  # Falso Positivo
                predict_list.append(bbox_pred[4])
                cont_FP += 1

        # Adiciona os Ground Truths não detectados (Falsos Negativos)
        for i, bbox_gt in enumerate(ground_truth[key]):
            if i not in matched_gt:
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                cv2.rectangle(overlay, (int(bbox_gt[0]), int(bbox_gt[1])), (x2_max, y2_max), (255, 0, 0), thickness=2)
                cv2.putText(overlay, str(classes[bbox_gt[-1]]), (x2_max, y2_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                ground_truth_list.append(bbox_gt[-1])
                predict_list.append(0)  # Falso Negativo

        # Aplica transparência às anotações
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        precision = round(cont_TP / (cont_TP + cont_FP), 3) if (cont_TP + cont_FP) > 0 else 0
        recall = round(cont_TP / ground_truth_values, 3) if ground_truth_values > 0 else 0

        cv2.putText(image, f"P: {precision}", (5, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
        cv2.putText(image, f"R: {recall}", (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)

        # Salva a imagem processada
        save_path = os.path.join(RESULTADOS_PATH, key)
        cv2.imwrite(save_path, image)
        #print(f"Imagem salva em: {save_path}")

    return ground_truth_list, predict_list

def compute_metrics(preds, targets, num_classes=1):
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    
    if num_classes <= 2:
        precision = BinaryPrecision()(preds, targets)
        recall = BinaryRecall()(preds, targets)
        fscore = BinaryF1Score()(preds, targets)
        r = BinaryAccuracy()(preds, targets)
    else:
        precision = MulticlassPrecision(num_classes=num_classes, average='macro')(preds, targets)
        recall = MulticlassRecall(num_classes=num_classes, average='macro')(preds, targets)
        fscore = MulticlassF1Score(num_classes=num_classes, average='macro')(preds, targets)
        r = MulticlassAccuracy(num_classes=num_classes, average='macro')(preds, targets)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'fscore': fscore.item(),
        'r': r.item()
    }

def geraResults(root,fold,model,nameModel,save_imgs):

    path_test_json = os.path.join(root,'filesJSON',fold+str('_test.json'))
    path_all_anotations = os.path.join(root,'train','_annotations.coco.json')

    classes_dict = get_classes(path_all_anotations)
    coco_test = pegaDataset(path_test_json)
    predict_values = {}
    ground_truth_values = {}
    controll = 0
    number_of_detections = 0
    real_number = 0
    for images in coco_test:
        controll+=1 
        ground_truth_list = []
        for i,bbox in enumerate(images['annotations']['bboxes']):
            x1,y1,width,height = bbox[0],bbox[1],bbox[2],bbox[3]
            label = images["annotations"]['labels'][i]
            ground_truth_list.append([x1,y1,width,height,label])
        ground_truth_values[images['file_name']] = ground_truth_list
        image = os.path.join(root,'train',images['file_name'])

        frame = cv2.imread(image)

        if nameModel == "YOLOV8":
            result = resultYOLO.result(frame,model)
        predict_values[images['file_name']] = result
        if controll > 3:
            break

    ground_truth_values_map = []
    predict_values_map = []

    for key in ground_truth_values:
        bbox_list1 = []
        label_list1 = []
        for values in ground_truth_values[key]:
            real_number+=1
            bbox = [values[0],values[1],values[2],values[3]]
            bbox = xywh_to_xyxy(bbox)
            bbox_list1.append(bbox)
            label_list1.append(values[-1])
        ground_truth_values_map.append({"boxes":torch.tensor(bbox_list1),"labels":torch.tensor(label_list1)})
        
    for key in predict_values:
        bbox_list = []
        label_list = []
        score_list = []
        for values in predict_values[key]:
            number_of_detections+=1
            bbox = [values[0],values[1],values[2],values[3]]
            bbox = xywh_to_xyxy(bbox)
            bbox_list.append(bbox)
            label_list.append(values[4])
            score_list.append(values[5])
        predict_values_map.append({"boxes":torch.tensor(bbox_list),"scores":torch.tensor(score_list),"labels":torch.tensor(label_list)})

    metric = MeanAveragePrecision()

    # predict_values_map = [
    #     {'boxes': torch.tensor([[252, 315, 286, 389]]), 'scores': torch.tensor([0.99]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[202, 127, 355, 231]]), 'scores': torch.tensor([0.98]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[130, 280, 207, 307]]), 'scores': torch.tensor([0.95]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[327, 341, 333, 356]]), 'scores': torch.tensor([0.97]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[336, 291, 343, 304]]), 'scores': torch.tensor([0.96]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[354, 364, 359, 378]]), 'scores': torch.tensor([0.99]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[298, 244, 304, 255]]), 'scores': torch.tensor([0.97]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[296, 171, 303, 182]]), 'scores': torch.tensor([0.96]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[150, 250, 200, 300]]), 'scores': torch.tensor([0.98]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[100, 200, 150, 250]]), 'scores': torch.tensor([0.99]), 'labels': torch.tensor([7])}
    # ]

    # ground_truth_values_map = [
    #     {'boxes': torch.tensor([[252, 315, 286, 389]]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[202, 127, 355, 231]]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[130, 280, 207, 307]]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[327, 341, 333, 356]]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[336, 291, 343, 304]]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[354, 364, 359, 378]]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[298, 244, 304, 255]]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[296, 171, 303, 182]]), 'labels': torch.tensor([7])},
    #     {'boxes': torch.tensor([[150, 250, 200, 300]]), 'labels': torch.tensor([6])},
    #     {'boxes': torch.tensor([[100, 200, 150, 250]]), 'labels': torch.tensor([7])}
    # ]

    metric.update(predict_values_map, ground_truth_values_map)

    result_map = metric.compute()
   
    ground_truth_cout_values = []
  
    for key in ground_truth_values:
        count_classes = [0] * len(classes_dict)
        for classe in classes_dict:
            for bound_box in ground_truth_values[key]:
                if classe == bound_box[-1]:
                    count_classes[classe] += 1 
        ground_truth_cout_values.append(count_classes)
    ground_truth_cout_values = torch.tensor(ground_truth_cout_values)
    predict_count_values = []
  
    for key in predict_values:
        count_classes = [0] * len(classes_dict)
        for classe in classes_dict:
            for bound_box in predict_values[key]:

                if classe == bound_box[4]:
                    count_classes[classe] += 1 
        predict_count_values.append(count_classes)
    predict_count_values = torch.tensor(predict_count_values)


    # predict_count_values = torch.tensor([
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # ])

    # ground_truth_cout_values = torch.tensor([
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # ])


    mae_metric = MeanAbsoluteError()
    rmse_metric = MeanSquaredError(squared=False) 

    mae = mae_metric(predict_count_values, ground_truth_cout_values)
    rmse = rmse_metric(predict_count_values, ground_truth_cout_values)

    mAP = result_map["map"]
    mAP50 = result_map["map_50"]
    mAP75 = result_map["map_75"]

    print(f"mAP: {mAP:.4f}")
    print(f"mAP50: {mAP50:.4f}")
    print(f"mAP75: {mAP75:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    ground_truth_list, predict_list = get_list_classes(ground_truth_values,predict_values,classes_dict)

    num_classes = len(classes_dict)

    # predict_list = [7, 6, 7, 6, 7, 6, 7, 7, 6, 7]
    # ground_truth_list = [7, 6, 7, 6, 7, 6, 7, 7, 6, 7]

    metrics = compute_metrics(predict_list, ground_truth_list, num_classes=num_classes)
    print(metrics)
    for i in range(num_classes):
        print(f'Ground {i}:',ground_truth_list.count(i))
        print(f'pred {i}:',predict_list.count(i))
    
model_path = "/home/pedroeduardo/Documentos/compara_detectores_torch/src/model_checkpoints/fold_1/YOLOV8/train/weights/best.pt"

geraResults(os.path.join('..', 'dataset','all'),"fold_1",model_path,"YOLOV8",False)