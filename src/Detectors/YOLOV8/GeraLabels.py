import json
import os
import shutil
import yaml

ROOT_DATA_DIR = os.path.join('..', 'dataset','all')

def _get_use_tiled_dataset():
    """Check if tiled dataset mode is enabled"""
    return os.getenv('USE_TILED_DATASET', 'true').lower() == 'true'

# Função para criar o dataset para treinamento da YOLOV8
def CriarLabelsYOLOV8(fold, root_data_dir=None):
    # Use provided root_data_dir or fall back to default
    if root_data_dir is None:
        root_data_dir = ROOT_DATA_DIR

    use_tiled = _get_use_tiled_dataset()

    # Determine where to read categories from
    if use_tiled:
        # For tiled datasets, read from train split annotations
        annotations_file = os.path.join(root_data_dir, 'train', '_annotations.coco.json')
    else:
        # For original dataset, read from train folder
        annotations_file = os.path.join(root_data_dir, 'train', '_annotations.coco.json')

    # Abre o arquivo Json onde temos as anotações de cada imagem
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    ann_ids = []
    for anotation in data["annotations"]:
        if anotation["category_id"] not in ann_ids:
            ann_ids.append(anotation["category_id"])
    Classe = []
    for category in data["categories"]:
        if category["id"] in ann_ids:
            Classe.append(category["name"],)

    caminho_arquivo_yaml = os.path.join(root_data_dir, 'data.yaml')

    # For tiled datasets, we need to create data.yaml if it doesn't exist
    if not os.path.exists(caminho_arquivo_yaml):
        # Create a new data.yaml
        conteudo = {
            'train': os.path.join(root_data_dir, 'YOLO', 'train', 'images'),
            'val': os.path.join(root_data_dir, 'YOLO', 'valid', 'images'),
            'test': os.path.join(root_data_dir, 'YOLO', 'test', 'images'),
            'nc': 0,  # Will be updated below
            'names': []  # Will be updated below
        }
    else:
        # Carregue o conteúdo do arquivo YAML
        with open(caminho_arquivo_yaml, 'r') as arquivo:
            conteudo = yaml.safe_load(arquivo)

    # Altere o conteúdo conforme necessário
    conteudo['nc'] = len(Classe)
    conteudo['names'] = Classe

    # Salve o conteúdo alterado de volta no arquivo YAML
    with open(caminho_arquivo_yaml, 'w') as arquivo:
        yaml.dump(conteudo, arquivo)

    yolo_output_dir = os.path.join(root_data_dir,'YOLO')
    if os.path.exists(yolo_output_dir):
        shutil.rmtree(yolo_output_dir)

    for c1 in ['train', 'test', 'valid']:
        for c2 in ['labels', 'images']:
            os.makedirs(os.path.join(yolo_output_dir, c1, c2), exist_ok=True)

    # Handle both tiled and original dataset structures
    if use_tiled:
        # For tiled datasets, read directly from train/val/test folders
        foldsUsadas = []
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(root_data_dir, split)
            annotations_path = os.path.join(split_dir, '_annotations.coco.json')
            if os.path.exists(annotations_path):
                foldsUsadas.append((split, annotations_path))
    else:
        # Original logic: read from filesJSON
        caminhos = (os.listdir(os.path.join(root_data_dir,'filesJSON')))
        foldsUsadas = []
        #Pega o caminho do arquivo coco que esta sendo usada
        for caminho in caminhos:
            fold_check = caminho.split("_")[0] + "_" +caminho.split("_")[1]
            if str(fold_check) == str(fold):
                foldsUsadas.append(caminho)

    for item in foldsUsadas:
        if use_tiled:
            # item is (split_name, annotations_path)
            split_name, caminho = item
            source_split = split_name  # Original split name for image source
            path = split_name  # Will be converted to 'valid' if needed for YOLO format
        else:
            # item is a filename string
            Caminho = item
            path = Caminho.split('_')[-1][0:-5]
            source_split = None  # Not used for non-tiled
            caminho = os.path.join(root_data_dir,'filesJSON',Caminho)

        # Lendo o arquivo JSON
        with open(caminho, 'r') as f:
            anotacaoDobras = json.load(f)
        # Exibindo os dados lidos
        NameFile = []
        imageID = []
        anotacao = {}
        idAnotcao = {}
        #Salva a lista de id das imagens
        for i in range(len(anotacaoDobras['annotations'])):
            imageID.append(anotacaoDobras['annotations'][i]['image_id'])

        #Faz a lista sem repetir os IDs
        imageID = (list(set(imageID)))

        #Cria um dicionario Com ID das imagens
        for i in imageID:
            anotacao[i] = []
            idAnotcao[i] = [] 
        #Ira pegar as anotações es classes de cada imagens e salvar em seus dicionarios
        for id in imageID:
            for i in range(len(anotacaoDobras['annotations'])):
                if anotacaoDobras['annotations'][i]['image_id'] == id:
                    anotacao[id].append(anotacaoDobras['annotations'][i]['bbox'])
                    idAnotcao[id].append(anotacaoDobras['annotations'][i]['category_id']-1)
        #Salva o nome de cada imagem
        for i in imageID:
            for j in range(len(anotacaoDobras['images'])):
                if (anotacaoDobras['images'][j]['id']) == i:
                    NameFile.append(anotacaoDobras['images'][j]['file_name'])
        
        slectImage = 0
        #Converte as anotações para o formato da YOLOV8
        for id in imageID:
            linhas = []
            for i in range (len(anotacao[id])):
                x1 = int(anotacao[id][i][0])
                y1 = int(anotacao[id][i][1])

                x2 = int(x1 + int(anotacao[id][i][2]))
                y2 = int(y1 + int(anotacao[id][i][3]))

                x_center = abs((x1+ x2)/2)
                y_center = abs((y1 + y2) / 2)

                width = abs(anotacao[id][i][2])
                height = abs(anotacao[id][i][3])
                linhas.append(str(abs(idAnotcao[id][i]))+' '+str(x_center/640)+' '+str(y_center/640)+' '+str(width/640)+' '+str(height/640)+"\n")

            # Determine source image location
            if use_tiled:
                # Images are in the same directory as annotations (use source_split which hasn't been converted yet)
                image = os.path.join(root_data_dir, source_split, NameFile[slectImage])
            else:
                # Images are in the train folder
                image = os.path.join(root_data_dir,'train',NameFile[slectImage])

            arq = NameFile[slectImage][0:-4]+'.txt'
            with open(arq, 'w') as arquivo:
            # Escrevendo múltiplas linhas no arquivo
                arquivo.writelines(linhas)
            slectImage+=1

            # Convert 'val' to 'valid' for YOLO directory naming convention
            output_path = 'valid' if path == 'val' else path

            shutil.move(arq, os.path.join(yolo_output_dir, output_path, 'labels'))
            shutil.copy(image, os.path.join(yolo_output_dir, output_path, 'images'))