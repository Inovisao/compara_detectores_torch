import json
import numpy as np
import cv2
import os
from Detectors.YOLOV8.DetectionsYolov8 import resultYOLO
from Detectors.FasterRCNN.inference import resultFaster
from Detectors.Detr.inference_image_detect import resultDetr
import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import csv

LIMIAR_IOU = 0.5
LIMIAR_CLASSIFICADOR=0.5

def convert_to_coco_format(results,idImagens):
    coco_results = []
    image_id = 0
    for j in range(len(results)):
        for boxes in results[j]:
            if boxes.size == 0:
                continue
            for bb in boxes:
                bbox_coco = [float(bb[0]), float(bb[1]), 
                            float(bb[2] - bb[0]), 
                            float(bb[3] - bb[1])]
                coco_results.append({
                        'image_id': idImagens[image_id],
                        'category_id': 1,  # assuming there is only one class
                        'bbox': bbox_coco,
                        'score': float(bb[4])
                    })
        image_id += 1
    return coco_results

def calculate_map(results, idImagens,dataset):
    if not results:
        print("Nenhum resultado fornecido.")
        return None, None, None
    coco_gt = COCO(dataset)
    coco_dt = coco_gt.loadRes(convert_to_coco_format(results,idImagens))

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0] if coco_eval.stats is not None else None  # mAP@0.5
    mAP50 = coco_eval.stats[1] if coco_eval.stats is not None else None  # mAP@0.75
    mAP75 = coco_eval.stats[2] if coco_eval.stats is not None else None  # mAP (mAP@0.5:0.95)

    return mAP, mAP50, mAP75

def printToFile(linha='',arquivo='../results/results.csv',modo='a'):
  original_stdout = sys.stdout # Save a reference to the original standard output
  with open(arquivo, modo) as f:
    sys.stdout = f # Change the standard output to the file we created.
    print(linha)
    sys.stdout = original_stdout # Reset the standard output to its original value

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    if bb1['x1'] >= bb1['x2'] or bb1['y1'] >= bb1['y2'] or bb2['x1'] >= bb2['x2'] or bb2['y1'] >= bb2['y2']:
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou

def is_max_score_thr(bb1, pred_array):
  """
    Compares if given bounding box is the one with the highest score_thr inside the array of predicted bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    pred_array : array of predicted objects
        Keys of dicts: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    boolean
  """
  is_max = True
  for cls in pred_array:
    for bb2 in cls:
      bbd={'x1':int(bb2[0]),'x2':int(bb2[2]),'y1':int(bb2[1]),'y2':int(bb2[3])}
      if is_max and bb2[4] > bb1['score_thr'] and get_iou(bb1,bbd) > LIMIAR_IOU:
        is_max = False
  return is_max

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

def convert_detections(detections):
    converted = []
    for detection in detections:
        # Extraia os valores na ordem x1, y1, x2, y2, score_thr
        x1 = detection['x1']
        y1 = detection['y1']
        x2 = detection['x2']
        y2 = detection['y2']
        score = detection['score_thr']
        
        # Forme um array numpy para cada detecção
        array_format = np.array([[x1, y1, x2, y2, score]], dtype=np.float32)
        converted.append(array_format)
    return converted

def gerar_csv(dados):
    # Definindo o nome do arquivo
    nome_arquivo = '../results/counting.csv'
    
    # Definindo os cabeçalhos do CSV (apenas para garantir que a estrutura dos dados seja consistente)
    cabecalhos = ['ml', 'fold', 'groundtruth', 'predicted', 'TP', 'FP', 'dif', 'fileName']
    
    # Escrevendo os dados no arquivo CSV sem cabeçalho
    with open(nome_arquivo, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=cabecalhos)
        
        # Escreve apenas as linhas de dados, sem o cabeçalho
        for linha in dados:
            writer.writerow(linha)

def geraResult(root,fold,model,nameModel,save_imgs):
    arquivoJson = os.path.join(root,'filesJSON',fold+str('_test.json'))
    cocodataset = pegaDataset(arquivoJson)
    MAX_BOX=1000
    results=[]
    medidos=[]
    preditos=[]
    all_TP = 0
    all_FP = 0
    all_GT=0
    idImagens = []
    dados = []
    for images in cocodataset:
        image = os.path.join(root,'train',images['file_name'])

        frame = cv2.imread(image)

        if nameModel == 'YOLOV8':
            imageSaveModel = 'YOLOV8'
            result = resultYOLO.result(frame,model)
        elif nameModel == 'FasterRCNN':
            imageSaveModel = 'Faster'
            result = resultFaster(fold,frame)
        else:
            imageSaveModel = 'Detr'
            result = resultDetr(fold,frame)
        
        if images['annotations']['bboxes'] == []:
            continue

        idImagens.append(images['image_id'])
        bboxes = np.insert(images['annotations']['bboxes'],4,0.91,axis=1)
        ground_thruth = []
        bboxes2 = []
        for i,bbox in enumerate(images['annotations']['bboxes']):
            classes = (images['annotations']['labels'][i] - 1)
            x1,x2 = int(bbox[0]),int(bbox[0]+bbox[2])
            y1,y2 = int(bbox[1]),int(bbox[1]+bbox[3])
            ground_thruth.append({'x1':x1,'x2':x2,'y1':y1,'y2':y2,'class':classes})
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), thickness=1)
        objetos_medidos = len(ground_thruth)

        for j in range(len(result)):
            for bb in result[j]:
                obj = {'x1':int(bb[0]),'x2':int(bb[2]),'y1':int(bb[1]),'y2':int(bb[3]),'score_thr':bb[4],'class':j}

                if is_max_score_thr(obj,result):
                    bboxes2.append(obj)
                    
        bboxes2 = np.array(bboxes2)
        objetos_preditos=0
        cont_TP=0
        cont_FP=0

        for j in range(min(MAX_BOX, bboxes2.shape[0])): 
            if bboxes2[j]['score_thr'] >= LIMIAR_CLASSIFICADOR: #score_thr  ou seja, a confiança
                objetos_preditos=objetos_preditos+1  # Total de objetos preditos automaticamente (usando IoU > 0.5)
                left_top = (bboxes2[j]['x1'],bboxes2[j]['y1'])
                left_top_text = (bboxes2[j]['x1'],bboxes2[j]['y1']-10)
                right_bottom = (bboxes2[j]['x2'],bboxes2[j]['y2'])
                tp = False
                for box in ground_thruth:
                    if get_iou(box,bboxes2[j]) > LIMIAR_IOU: # IOU > 0.3
                        if(bboxes2[j]['class'] == box['class']):
                            tp = True
                if tp == True:
                    cont_TP+=1
                    frame=cv2.rectangle(frame, left_top, right_bottom, (0,255,0), thickness=1) 
                else:
                    cont_FP+=1
                    frame=cv2.rectangle(frame, left_top, right_bottom, (0,0,255), thickness=1)    
        dados.append({'ml': nameModel+'x', 'fold': fold, 'groundtruth': objetos_medidos, 'predicted': objetos_preditos, 'TP': cont_TP, 'FP': cont_FP, 'dif': int(objetos_medidos-objetos_preditos), 'fileName': images['file_name']})
        all_TP+=cont_TP    
        all_FP+=cont_FP
        all_GT+=len(images['annotations']['bboxes'])

        medidos.append(len(ground_thruth))
        preditos.append(objetos_preditos)

        frame=cv2.putText(frame, str(objetos_medidos),(5,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 1)
        frame=cv2.putText(frame, str(objetos_preditos),(5,60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 1)

        try:
            precision = round(cont_TP/(cont_TP+cont_FP),3)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = round(cont_TP/len(bboxes),3)
        except ZeroDivisionError:
            recall = 0

        frame=cv2.putText(frame, "P:"+str(precision),(5,90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255), 1)
        frame=cv2.putText(frame, "R:"+str(recall),(5,120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255), 1)

        if save_imgs:
            save_path = '/prediction_'+imageSaveModel
            save_path = os.path.join('..','results','prediction',imageSaveModel)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            img_path = os.path.join(save_path ,images['file_name'])
            cv2.imwrite(img_path,frame)
        result2 = convert_detections(bboxes2)

        results.append(result)
        diferenca=objetos_preditos-objetos_medidos
    try:
        mAP, mAP50, mAP75 = calculate_map(results,idImagens,arquivoJson)
    except:
        mAP, mAP50, mAP75 = 0
    try:  
        MAE=mean_absolute_error(medidos,preditos)
        RMSE=math.sqrt(mean_squared_error(medidos,preditos))
    except:
        MAE=0
        RMSE=0    
    try:
        r=np.corrcoef(medidos,preditos)[0,1]
    except ZeroDivisionError:
        r = 0
    try:
        precision_fold = round(all_TP/(all_TP+all_FP),3)
    except ZeroDivisionError:
        precision_fold = 0
    try:
        recall_fold = round(all_TP/all_GT,3)
    except ZeroDivisionError:
        recall_fold = 0
    try: 
        fscore=round((2*precision_fold*recall_fold)/(precision_fold+recall_fold),3)
    except ZeroDivisionError:
        fscore=0
    string_results = str(mAP)+','+str(mAP50)+','+str(mAP75)+','+str(MAE)+','+str(RMSE)+','+str(r)+','+str(precision_fold)+','+str(recall_fold)+','+str(fscore)
    gerar_csv(dados)
    return string_results

def criaCSV(num_dobra,selected_model,fold,root,model_path,save_imgs):
    save_results = os.path.join('..','results','results.csv')
    resAP50 = geraResult(root,fold,nameModel=selected_model,model=model_path,save_imgs=save_imgs)
    printToFile(selected_model + ','+fold+','+resAP50,save_results,'a')