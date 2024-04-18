import os
import os
import torch
import os.path as osp
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmdet.datasets import CocoDataset
from mmcv.visualization import color_val
import mmcv, torch
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import sys
from mmdet.datasets import CocoDataset

LIMIAR_CLASSIFICADOR = 0
LIMIAR_IOU = 0
MOSTRA_NOME_CLASSE = 0
CLASSES = 0

def printToFile(linha='',arquivo='dataset/results.csv',modo='a'):
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
    # print(bb1)
    # print(bb2)
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
#   print("iou:",str(iou))
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


def testingModel(cfg=None,typeN='test',models_path=None,show_imgs=True,save_imgs=False,num_model=1,fold='fold_1'):

  classes = ('Peixes',)
  pasta_dataset=os.path.join('..','dataset','all')
  # build the model from a config file and a checkpoint file
  torch.backends.cudnn.benchmark = True


  modelx = 'init_detector(cfg, models_path)'
  
  if typeN=='test':
    ann_file = os.path.join(pasta_dataset,'train')
    img_prefix = os.path.join(pasta_dataset,'filesJSON',fold)


  coco_dataset = CocoDataset(ann_file=ann_file, classes=classes,data_root=pasta_dataset,img_prefix=img_prefix,filter_empty_gt=False)
  print(coco_dataset)
  input()

  # Vai desenhar os retângulos na imagem:
  # AZUL: Anotações feitas por humanos
  # VERDE: Detecção feita automaticamente e que tem intersecção com uma caixa dos humanos (VP = Verdadeiro Positivo)
  # VERMELHO: Detecção feita automaticamente mas sem intersecção com anotação (FP = Falso Positivo)
  MAX_BOX=1000
  results=[]
  medidos=[]
  preditos=[]
  all_TP = 0
  all_FP = 0
  all_GT=0
  for i,dt in enumerate(coco_dataset.data_infos):
    print('Processando Imagem de Teste:',dt['file_name'])

    imagex=None
    imagex=mmcv.imread(os.path.join(coco_dataset.img_prefix,dt['file_name']))
    resultx = inference_detector(modelx, imagex)
    #modelx.show_result(imagex, resultx, score_thr=0.3, out_file=models_path + dt['file_name'])

    ann = coco_dataset.get_ann_info(i)
    labels = ann['labels']
    bboxes = np.insert(ann['bboxes'],4,0.91,axis=1)

    #vis.imshow_gt_det_bboxes(imagex,dict(gt_bboxes=bboxes, gt_labels=np.repeat(1, len(bboxes))), resultx,det_bbox_color=(0,100,0), show=True,score_thr=0.5)
    ground_thruth = []
    objetos_medidos=bboxes.shape[0] # Total de objetos marcados manualmente (groundtruth)
    for j in range(min(MAX_BOX, bboxes.shape[0])): 
      left_top = (int(bboxes[j, 0]), int(bboxes[j, 1]))
      right_bottom = (int(bboxes[j, 2]), int(bboxes[j, 3]))
      ground_thruth.append({'x1':left_top[0],'x2':right_bottom[0],'y1':left_top[1],'y2':right_bottom[1],'class':labels[j]})
      imagex=cv2.rectangle(imagex, left_top, right_bottom, color_val('blue'), thickness=1)
    
    bboxes2 = []
    for j in range(len(resultx)):
      for bb in resultx[j]:
        obj = {'x1':int(bb[0]),'x2':int(bb[2]),'y1':int(bb[1]),'y2':int(bb[3]),'score_thr':bb[4],'class':j}
        if is_max_score_thr(obj,resultx):
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
        TP = False
        for box in ground_thruth:          
          if get_iou(box,bboxes2[j]) > LIMIAR_IOU: # IOU > 0.3
            if(bboxes2[j]['class'] == box['class']):
              TP = True

        if TP == True:
          cont_TP+=1
          imagex=cv2.rectangle(imagex, left_top, right_bottom, color_val('green'), thickness=1) 
          if MOSTRA_NOME_CLASSE:
            cv2.putText(imagex, CLASSES[bboxes2[j]['class']], left_top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_val('green'), thickness=2) 

        else:
          cont_FP+=1
          imagex=cv2.rectangle(imagex, left_top, right_bottom, color_val('red'), thickness=1)    
          if MOSTRA_NOME_CLASSE:
            cv2.putText(imagex, CLASSES[bboxes2[j]['class']], left_top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_val('red'), thickness=2)

#    print("TP:"+ str(cont_TP))
    all_TP+=cont_TP    
#    print("FP:"+ str(cont_FP)) 
    all_FP+=cont_FP
    all_GT+=len(bboxes)
              
        
        
        
    # Guarda todas as contagens, manuais e preditas, de cada imagem em uma lista
    medidos.append(objetos_medidos)
    preditos.append(objetos_preditos)

    # Mostra as contagens na imagem que será salva
    imagex=cv2.putText(imagex, str(objetos_medidos),(5,30), cv2.FONT_HERSHEY_TRIPLEX, 1, color_val('blue'), 1)
    imagex=cv2.putText(imagex, str(objetos_preditos),(5,60), cv2.FONT_HERSHEY_TRIPLEX, 1, color_val('green'), 1)
    try:
        precision = round(cont_TP/(cont_TP+cont_FP),3)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = round(cont_TP/len(bboxes),3)
    except ZeroDivisionError:
        recall = 0

    imagex=cv2.putText(imagex, "P:"+str(precision),(5,90), cv2.FONT_HERSHEY_TRIPLEX, 1, color_val('yellow'), 1)
    imagex=cv2.putText(imagex, "R:"+str(recall),(5,120), cv2.FONT_HERSHEY_TRIPLEX, 1, color_val('yellow'), 1)

    if show_imgs and i<10:  ## VAI MOSTRAR APENAS 10 IMAGENS PARA NÃO FICAR LENTO!
      cv2.imshow(imagex)
    elif save_imgs:
      #save_path = cfg.data_root+'/prediction_'+selected_model
      save_path = os.path.join(cfg.data_root,save_path)
      if not os.path.exists(save_path):
        os.makedirs(save_path)
      img_path = os.path.join(save_path ,dt['file_name'])
      cv2.imwrite(img_path,imagex)



    results.append(resultx)
    diferenca=objetos_preditos-objetos_medidos
    #printToFile(str(num_model)+'_'+selected_model + ','+fold+','+str(objetos_medidos)+','+str(objetos_preditos)+','+str(cont_TP)+','+str(cont_FP)+','+str(diferenca)+','+dt['file_name'],'dataset/counting.csv','a')  


  print("preditos:")  
  print(preditos) 
  eval_results = coco_dataset.evaluate(results, classwise=True)
  eval_results2 = coco_dataset.evaluate(results, classwise=True, metric='proposal') 
  #recall = coco_dataset.fast_eval_recall(results,proposal_nums=(100), iou_thrs  = 0.5)
  coco_dataset.results2json(results, pasta_dataset)
  print('Resultados do comando coco_dataset.evaluate:')
  print(eval_results)
  print(eval_results2)
  # print(results)
  #print(selected_model,'\t',eval_results['bbox_mAP_50'])
  #string_results = selected_model+'\t'+str(eval_results['bbox_mAP_50'])

  string_results = '0,0,0,0,0,0,0,0,0'

  try:
    mAP=eval_results['bbox_mAP']
    mAP50=eval_results['bbox_mAP_50']
    mAP75=eval_results['bbox_mAP_75']
  except:
    mAP=0
    mAP50=0
    mAP75=0
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

  return string_results

testingModel(cfg=None,typeN='test',models_path=None,show_imgs=True,save_imgs=False,num_model=1,fold='fold_1')