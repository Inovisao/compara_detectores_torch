import os
import shutil
import subprocess
from Detectors.Detr.GeraDobras import convert_coco_to_voc

# Função para Rodar a rede 
def runDetr(fold,fold_dir,ROOT_DATA_DIR):
    print(f"[runDetr] fold={fold} fold_dir={fold_dir} ROOT_DATA_DIR={ROOT_DATA_DIR}", flush=True)

    training_dir = os.path.abspath(os.path.join(fold_dir, 'Detr', 'training'))
    os.makedirs(training_dir, exist_ok=True)
    print(f"[runDetr] DETR_TRAINING_DIR={training_dir}", flush=True)

    print(f"[runDetr] Convertendo COCO → VOC para fold={fold}", flush=True)
    convert_coco_to_voc(fold, root_data_dir=ROOT_DATA_DIR)
    print(f"[runDetr] Conversão concluída", flush=True)

    treino = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TreinoDetr.sh')
    treino_abs = treino
    print(f"[runDetr] script={treino_abs} exists={os.path.exists(treino_abs)}", flush=True)
    print(f"[runDetr] Executando subprocess...", flush=True)
    env = os.environ.copy()
    env['DETR_TRAINING_DIR'] = training_dir
    env['DATASET_ROOT'] = str(ROOT_DATA_DIR)
    result = subprocess.run([treino], check=True, env=env)
    print(f"[runDetr] Concluído, código={result.returncode}, checkpoint em {training_dir}", flush=True)

    detr_data = os.path.join(ROOT_DATA_DIR, 'detr')
    if os.path.exists(detr_data):
        print(f"[runDetr] Removendo dados VOC temporários: {detr_data}", flush=True)
        shutil.rmtree(detr_data)
