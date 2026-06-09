from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from utils.augmentation import build_augmentation_pipeline

_PIPELINE = build_augmentation_pipeline("pascal_voc")

ROOT_DATA_DIR = os.path.join('..','dataset','all')


def _read_voc_xml(xml_path: Path) -> tuple[list[str], list[list[int]]]:
    root = ET.parse(xml_path).getroot()
    class_names, bboxes = [], []
    for obj in root.findall("object"):
        class_names.append(obj.findtext("name", ""))
        bb = obj.find("bndbox")
        bboxes.append([
            int(float(bb.findtext("xmin", 0))),
            int(float(bb.findtext("ymin", 0))),
            int(float(bb.findtext("xmax", 0))),
            int(float(bb.findtext("ymax", 0))),
        ])
    return class_names, bboxes


def _write_voc_xml(
    xml_path: Path,
    file_name: str,
    width: int,
    height: int,
    class_names: list[str],
    bboxes: list[list[int]],
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = file_name
    src = ET.SubElement(root, "source")
    ET.SubElement(src, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    for cls, bbox in zip(class_names, bboxes):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(bbox[0])
        ET.SubElement(bb, "ymin").text = str(bbox[1])
        ET.SubElement(bb, "xmax").text = str(bbox[2])
        ET.SubElement(bb, "ymax").text = str(bbox[3])
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=False)


def _augment_detr_train(train_dir: str, copies: int = 2) -> None:
    train_path = Path(train_dir)
    image_paths = sorted(train_path.glob("*.jpg")) + sorted(train_path.glob("*.png"))

    for image_path in image_paths:
        xml_path = train_path / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue

        class_names, bboxes = _read_voc_xml(xml_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for i in range(copies):
            result = _PIPELINE(image=image_rgb, bboxes=bboxes, class_labels=class_names)
            aug_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_h, aug_w = result["image"].shape[:2]
            aug_bboxes = [list(map(int, b)) for b in result["bboxes"]]
            aug_classes = list(result["class_labels"])

            suffix = f"_aug{i + 1}"
            aug_stem = f"{image_path.stem}{suffix}"
            aug_file_name = f"{aug_stem}{image_path.suffix}"

            cv2.imwrite(str(train_path / aug_file_name), aug_bgr)
            _write_voc_xml(
                train_path / f"{aug_stem}.xml",
                aug_file_name,
                aug_w,
                aug_h,
                aug_classes,
                aug_bboxes,
            )


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
    
##

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

    _augment_detr_train(caminho_train)