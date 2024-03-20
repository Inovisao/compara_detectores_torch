from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
ROOT_DATA_DIR = os.path.join('..','dataset')

def convert_coco_to_voc(fold):
    diretorio = os.path.join(ROOT_DATA_DIR,'faster')
    if os.path.exists(diretorio):  
        shutil.rmtree(diretorio)
    os.makedirs(diretorio)  
    caminho_train = os.path.join(ROOT_DATA_DIR,'faster','train')
    caminho_test = os.path.join(ROOT_DATA_DIR,'faster','test')
    caminho_valid = os.path.join(ROOT_DATA_DIR,'faster','valid')
    os.makedirs(caminho_train,exist_ok=True)
    os.makedirs(caminho_test,exist_ok=True)
    os.makedirs(caminho_valid,exist_ok=True)


    caminhos = (os.listdir(os.path.join(ROOT_DATA_DIR,'filesJSON')))
    foldsUsadas = []
    #Pega o caminho do arquivo coco que esta sendo usada
    for caminho in caminhos:
        if str(caminho[0:6]) == str(fold):
            foldsUsadas.append(caminho)
    for Caminho in foldsUsadas:
        coco_annotation_file = os.path.join(ROOT_DATA_DIR,'filesJSON',Caminho)
        path = Caminho[7:-5]
        if path == 'val':
            output_dir = caminho_valid
        elif path == 'test':
            output_dir = caminho_test
        else:
            output_dir = caminho_train
        coco = COCO(coco_annotation_file)
        
        categories = coco.loadCats(coco.getCatIds())
        category_id_to_name = {category['id']: category['name'] for category in categories}
        
        image_ids = coco.getImgIds()
        
        for image_id in tqdm(image_ids, desc="Converting images"):
            image_data = coco.loadImgs(image_id)[0]
            file_name = image_data['file_name']
            
            annotations_ids = coco.getAnnIds(imgIds=image_data['id'])
            annotations = coco.loadAnns(annotations_ids)
            
            with open(os.path.join(output_dir, os.path.splitext(file_name)[0] + '.xml'), 'w') as f:
                f.write('<annotation>\n')
                f.write('\t<folder>' + 'JPEGImages' + '</folder>\n')
                f.write('\t<filename>' + file_name + '</filename>\n')
                f.write('\t<source>\n')
                f.write('\t\t<database>Unknown</database>\n')
                f.write('\t</source>\n')
                f.write('\t<size>\n')
                f.write('\t\t<width>' + str(image_data['width']) + '</width>\n')
                f.write('\t\t<height>' + str(image_data['height']) + '</height>\n')
                f.write('\t\t<depth>3</depth>\n')
                f.write('\t</size>\n')
                f.write('\t<segmented>0</segmented>\n')

                for annotation in annotations:
                    f.write('\t<object>\n')
                    f.write('\t\t<name>' + category_id_to_name[annotation['category_id']] + '</name>\n')
                    f.write('\t\t<pose>Unspecified</pose>\n')
                    f.write('\t\t<truncated>0</truncated>\n')
                    f.write('\t\t<difficult>0</difficult>\n')
                    f.write('\t\t<bndbox>\n')
                    f.write('\t\t\t<xmin>' + str(int(annotation['bbox'][0])) + '</xmin>\n')
                    f.write('\t\t\t<ymin>' + str(int(annotation['bbox'][1])) + '</ymin>\n')
                    f.write('\t\t\t<xmax>' + str(int(annotation['bbox'][0] + annotation['bbox'][2])) + '</xmax>\n')
                    f.write('\t\t\t<ymax>' + str(int(annotation['bbox'][1] + annotation['bbox'][3])) + '</ymax>\n')
                    f.write('\t\t</bndbox>\n')
                    f.write('\t</object>\n')
                    
                f.write('</annotation>')
            image = os.path.join(ROOT_DATA_DIR,'all','train',file_name)
            shutil.copy(image,output_dir)

