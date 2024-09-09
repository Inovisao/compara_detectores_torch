from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm

ROOT_DATA_DIR = os.path.join('..', 'dataset', 'all')

def convert_coco_to_custom_xml(fold):
    diretorio = os.path.join(ROOT_DATA_DIR, 'retinanet')
    if os.path.exists(diretorio):  
        shutil.rmtree(diretorio)
    os.makedirs(diretorio)
    
    caminho_train = os.path.join(ROOT_DATA_DIR, 'retinanet', 'train')
    caminho_test = os.path.join(ROOT_DATA_DIR, 'retinanet', 'test')
    caminho_valid = os.path.join(ROOT_DATA_DIR, 'retinanet', 'valid')
    os.makedirs(caminho_train, exist_ok=True)
    os.makedirs(caminho_test, exist_ok=True)
    os.makedirs(caminho_valid, exist_ok=True)

    caminhos = os.listdir(os.path.join(ROOT_DATA_DIR, 'filesJSON'))
    foldsUsadas = []
    
    for caminho in caminhos:
        if str(caminho.split('_')[1]) == str(fold).split('_')[1]:
            foldsUsadas.append(caminho)
    
    for Caminho in foldsUsadas:
        coco_annotation_file = os.path.join(ROOT_DATA_DIR, 'filesJSON', Caminho)
        path = Caminho.split('_')[2].split('.')[0]
        
        if path == 'val':
            output_dir = caminho_valid
        elif path == 'test':
            output_dir = caminho_test
        else:
            output_dir = caminho_train
            
        output_dirimg = os.path.join(output_dir,'image')
        output_dirannotations = os.path.join(output_dir,'annotations')

        os.makedirs(output_dirimg, exist_ok=True)
        os.makedirs(output_dirannotations, exist_ok=True)


        coco = COCO(coco_annotation_file)
        categories = coco.loadCats(coco.getCatIds())
        category_id_to_name = {category['id']: category['name'] for category in categories}
        
        image_ids = coco.getImgIds()
        
        for image_id in tqdm(image_ids, desc="Converting images"):
            image_data = coco.loadImgs(image_id)[0]
            file_name = image_data['file_name']
            
            annotations_ids = coco.getAnnIds(imgIds=image_data['id'])
            annotations = coco.loadAnns(annotations_ids)
            
            with open(os.path.join(output_dirannotations, os.path.splitext(file_name)[0] + '.xml'), 'w') as f:
                f.write('<?xml version="1.0" ?>\n')
                f.write('<annotation verified="yes">\n')
                f.write(f'\t<folder>{fold}</folder>\n')  # Adicione o nome da pasta, se aplicável
                f.write(f'\t<filename>{file_name}</filename>\n')
                f.write(f'\t<path>{output_dirimg}</path>\n')  # Adicione o caminho completo da imagem
                f.write('\t<source>\n')
                f.write('\t\t<database>Unknown</database>\n')
                f.write('\t</source>\n')
                f.write(f'\t<size>\n')
                f.write(f'\t\t<width>{image_data["width"]}</width>\n')
                f.write(f'\t\t<height>{image_data["height"]}</height>\n')
                f.write(f'\t\t<depth>3</depth>\n')  # Supondo imagens RGB
                f.write(f'\t</size>\n')
                f.write(f'\t<segmented>0</segmented>\n')
                
                for annotation in annotations:
                    f.write('\t<object>\n')
                    f.write(f'\t\t<name>{category_id_to_name[annotation["category_id"]]}</name>\n')
                    f.write('\t\t<pose>Unspecified</pose>\n')
                    f.write(f'\t\t<truncated>{int(annotation.get("truncated", 0))}</truncated>\n')  # Supondo truncado 0 por padrão
                    f.write(f'\t\t<difficult>{int(annotation.get("difficult", 0))}</difficult>\n')  # Supondo difficult 0 por padrão
                    f.write(f'\t\t<bndbox>\n')
                    f.write(f'\t\t\t<xmin>{int(annotation["bbox"][0])}</xmin>\n')
                    f.write(f'\t\t\t<ymin>{int(annotation["bbox"][1])}</ymin>\n')
                    f.write(f'\t\t\t<xmax>{int(annotation["bbox"][0] + annotation["bbox"][2])}</xmax>\n')
                    f.write(f'\t\t\t<ymax>{int(annotation["bbox"][1] + annotation["bbox"][3])}</ymax>\n')
                    f.write(f'\t\t</bndbox>\n')
                    f.write('\t</object>\n')

                f.write('</annotation>\n')
            
            image = os.path.join(ROOT_DATA_DIR, 'train', file_name)
            shutil.copy(image, output_dirimg)

# Exemplo de uso
convert_coco_to_custom_xml('fold_1')
