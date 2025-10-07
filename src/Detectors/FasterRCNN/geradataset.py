import os
import shutil
import json

# Caminhos
ROOT_DATA_DIR = os.path.join('..','dataset','all')

def _get_use_tiled_dataset():
    """Check if tiled dataset mode is enabled"""
    return os.getenv('USE_TILED_DATASET', 'true').lower() == 'true'

# Criar a pasta de destino se não existir
def geredata(fold, root_data_dir=None):
    # Use provided root_data_dir or fall back to default
    if root_data_dir is None:
        root_data_dir = ROOT_DATA_DIR

    use_tiled = _get_use_tiled_dataset()

    destination_folder = os.path.join(root_data_dir,'Faster')
    os.makedirs(destination_folder, exist_ok=True)

    if use_tiled:
        # For tiled datasets, read directly from train/val/test folders
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(root_data_dir, split)
            annotations_path = os.path.join(split_dir, '_annotations.coco.json')

            if not os.path.exists(annotations_path):
                continue

            dest_split_dir = os.path.join(destination_folder, split)
            os.makedirs(dest_split_dir, exist_ok=True)

            # Copy annotation file
            path_new_json = os.path.join(dest_split_dir, '_annotations.coco.json')
            shutil.copy(annotations_path, path_new_json)

            # Load JSON and copy images
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for imgs in data['images']:
                img_name = imgs['file_name']
                img_path = os.path.join(split_dir, img_name)
                if os.path.exists(img_path):
                    shutil.copy(img_path, dest_split_dir)
    else:
        # Original logic for non-tiled datasets
        foldsUsadas = []
        caminhos = (os.listdir(os.path.join(root_data_dir,'filesJSON')))
        #Pega o caminho do arquivo coco que esta sendo usada
        for caminho in caminhos:
            fold_check = caminho.split("_")[0] + "_" +caminho.split("_")[1]
            if str(fold_check) == str(fold):
                foldsUsadas.append(caminho)

        # Carregar o JSON
        for fold_file in foldsUsadas:
            path = os.path.join(destination_folder, fold_file.split('_')[-1][0:-5])
            os.makedirs(path, exist_ok=True)
            json_path = os.path.join(root_data_dir,'filesJSON', fold_file)
            path_new_json = os.path.join(path,'_annotations.coco.json')
            shutil.copy(json_path, path_new_json)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for imgs in data['images']:
                img_name = imgs['file_name']
                img_path = os.path.join(root_data_dir,'train',img_name)
                shutil.copy(img_path, path)
