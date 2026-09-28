import json
import os
import shutil
import yaml
from collections import defaultdict

ROOT_DATA_DIR = os.path.join('..', 'dataset','all')


# Função para criar o dataset para treinamento da YOLOV8
def CriarLabelsYOLOV8(fold):
    with open(os.path.join(ROOT_DATA_DIR, 'train', '_annotations.coco.json'), 'r') as f:
        data = json.load(f)

    category_ids = {annotation['category_id'] for annotation in data['annotations']}
    categories = [category for category in data['categories'] if category['id'] in category_ids]
    class_id_by_category = {
        category['id']: class_id for class_id, category in enumerate(categories)
    }
    caminho_arquivo_yaml = os.path.join(ROOT_DATA_DIR, 'data.yaml')
    with open(caminho_arquivo_yaml, 'r') as arquivo:
        conteudo = yaml.safe_load(arquivo)

    conteudo['nc'] = len(categories)
    conteudo['names'] = [category['name'] for category in categories]

    with open(caminho_arquivo_yaml, 'w') as arquivo:
        yaml.safe_dump(conteudo, arquivo, sort_keys=False)

    yolo_root = os.path.join(ROOT_DATA_DIR, 'YOLO')
    if os.path.exists(yolo_root):
        shutil.rmtree(yolo_root)

    for split in ('train', 'test', 'valid'):
        for kind in ('labels', 'images'):
            os.makedirs(os.path.join(yolo_root, split, kind), exist_ok=True)

    files_json = os.path.join(ROOT_DATA_DIR, 'filesJSON')
    for filename in sorted(os.listdir(files_json)):
        parts = filename.split('_')
        if len(parts) < 3 or '_'.join(parts[:2]) != fold or not filename.endswith('.json'):
            continue

        split = os.path.splitext(parts[-1])[0]
        if split == 'val':
            split = 'valid'
        if split not in ('train', 'test', 'valid'):
            continue

        with open(os.path.join(files_json, filename), 'r') as f:
            fold_data = json.load(f)

        annotations_by_image = defaultdict(list)
        for annotation in fold_data['annotations']:
            annotations_by_image[annotation['image_id']].append(annotation)

        for image in fold_data['images']:
            image_path = os.path.join(ROOT_DATA_DIR, 'train', image['file_name'])
            image_width = float(image['width'])
            image_height = float(image['height'])
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"Dimensões inválidas na imagem {image['file_name']}")

            label_lines = []
            for annotation in annotations_by_image[image['id']]:
                category_id = annotation['category_id']
                if category_id not in class_id_by_category:
                    raise ValueError(f"Categoria COCO desconhecida: {category_id}")

                x, y, width, height = map(float, annotation['bbox'])
                x_center = (x + width / 2) / image_width
                y_center = (y + height / 2) / image_height
                label_lines.append(
                    f"{class_id_by_category[category_id]} {x_center:.8f} {y_center:.8f} "
                    f"{width / image_width:.8f} {height / image_height:.8f}\n"
                )

            label_name = os.path.splitext(os.path.basename(image['file_name']))[0] + '.txt'
            label_path = os.path.join(yolo_root, split, 'labels', label_name)
            with open(label_path, 'w') as arquivo:
                arquivo.writelines(label_lines)
            shutil.copy2(image_path, os.path.join(yolo_root, split, 'images'))