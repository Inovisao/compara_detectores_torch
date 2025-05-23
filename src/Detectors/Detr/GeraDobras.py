from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import json
import yaml

ROOT_DATA_DIR = os.path.join('..','dataset','all')
# Função que Ira gerar as labels para o Treino da faster
def convert_coco_to_voc(fold):

    with open(os.path.join(ROOT_DATA_DIR, 'train', '_annotations.coco.json'), 'r') as f:
        data = json.load(f)
    
    ann_ids = []
    for anotation in data["annotations"]:
        if anotation["category_id"] not in ann_ids:
            ann_ids.append(anotation["category_id"])
    Classe = ['__background__']
    for category in data["categories"]:
        if category["id"] in ann_ids:
            Classe.append(category["name"],)
    caminho_arquivo_yaml = os.path.join(ROOT_DATA_DIR, 'dataDetr.yaml')
    
    # Carregue o conteúdo do arquivo YAML
    with open(caminho_arquivo_yaml, 'r') as arquivo:
        conteudo = yaml.safe_load(arquivo)

    # Altere o conteúdo conforme necessário
    conteudo['NC'] = len(Classe)
    conteudo['CLASSES'] = Classe

    # Salve o conteúdo alterado de volta no arquivo YAML
    with open(caminho_arquivo_yaml, 'w') as arquivo:
        yaml.dump(conteudo, arquivo)

    diretorio = os.path.join(ROOT_DATA_DIR,'detr')
    if os.path.exists(diretorio):  
        shutil.rmtree(diretorio)
    os.makedirs(diretorio)  
    caminho_train = os.path.join(ROOT_DATA_DIR,'detr','train')
    caminho_test = os.path.join(ROOT_DATA_DIR,'detr','test')
    caminho_valid = os.path.join(ROOT_DATA_DIR,'detr','valid')
    os.makedirs(caminho_train,exist_ok=True)
    os.makedirs(caminho_test,exist_ok=True)
    os.makedirs(caminho_valid,exist_ok=True)

    caminhos = (os.listdir(os.path.join(ROOT_DATA_DIR,'filesJSON')))
    foldsUsadas = []
    #Pega o caminho do arquivo coco que esta sendo usada
    for caminho in caminhos:
        if str(caminho.split('_')[1]) == str(fold.split('_')[1]):
            foldsUsadas.append(caminho)
    # Loop para separar e criar as labels e imagens
    for Caminho in foldsUsadas:
        coco_annotation_file = os.path.join(ROOT_DATA_DIR,'filesJSON',Caminho) # Le o arquivo Json
        path = Caminho.split('_')[2].split('.')[0]
        # Ira selecionar o diretorio
        if path == 'val':
            output_dir = caminho_valid
        elif path == 'test':
            output_dir = caminho_test
        else:
            output_dir = caminho_train

        coco = COCO(coco_annotation_file) # Convert coco para json
        
        categories = coco.loadCats(coco.getCatIds())
        category_id_to_name = {category['id']: category['name'] for category in categories}
        
        image_ids = coco.getImgIds()
        # Loop onde escreve a labels referente as imagens
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
            image = os.path.join(ROOT_DATA_DIR,'train',file_name)
            shutil.copy(image,output_dir)