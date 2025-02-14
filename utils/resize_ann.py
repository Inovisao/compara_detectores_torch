import json
import os

def resize_annotations(coco_json_path, output_json_path, new_size=(640, 640)):
    # Ler o arquivo JSON COCO
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Criar um dicionário para mapear ID da imagem para suas dimensões
    image_sizes = {image['id']: (image['width'], image['height']) for image in coco_data['images']}

    # Modificar as anotações
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id in image_sizes:
            original_width, original_height = image_sizes[image_id]
            # Calcular a proporção de redimensionamento
            scale_x = new_size[0] / original_width
            scale_y = new_size[1] / original_height

            # Ajustar as coordenadas da bounding box
            annotation['bbox'][0] *= scale_x  # x
            annotation['bbox'][1] *= scale_y  # y
            annotation['bbox'][2] *= scale_x  # largura
            annotation['bbox'][3] *= scale_y  # altura

    # Salvar o arquivo JSON modificado
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

# Exemplo de uso
coco_json_file = 'train/_annotations.coco.json'  # Caminho para o arquivo JSON original
output_json_file = '../dataset/all/train/_annotations.coco.json'  # Caminho para o novo arquivo JSON

resize_annotations(coco_json_file, output_json_file, new_size=(640, 640))


# Passo 1: Abrir o arquivo em modo de leitura e carregar os dados
with open(output_json_file, 'r') as f:
    coco_data = json.load(f)

# Passo 2: Modificar os dados
for image in coco_data['images']:
    image['width'] = 640
    image['height'] = 640

# Passo 3: Abrir o arquivo novamente em modo de escrita e salvar as alterações
with open(output_json_file, 'w') as f:
    json.dump(coco_data, f, indent=4) 