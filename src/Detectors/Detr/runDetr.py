import os
import shutil
import subprocess
from Detectors.Detr.GeraDobras import convert_coco_to_voc

# Função para Rodar a rede 
def runDetr(fold,fold_dir,ROOT_DATA_DIR):
    print(f"[runDetr] fold={fold} fold_dir={fold_dir} ROOT_DATA_DIR={ROOT_DATA_DIR}", flush=True)

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    print(f"[runDetr] Convertendo COCO → VOC para fold={fold}", flush=True)
    convert_coco_to_voc(fold)
    print(f"[runDetr] Conversão concluída", flush=True)

    treino = os.path.join('Detectors','Detr','TreinoDetr.sh')
    treino_abs = os.path.abspath(treino)
    print(f"[runDetr] script={treino_abs} exists={os.path.exists(treino_abs)}", flush=True)
    print(f"[runDetr] Executando subprocess...", flush=True)
    result = subprocess.run([treino], check=True)
    print(f"[runDetr] Subprocess retornou código: {result.returncode}", flush=True)

    src = os.path.abspath('./Detr')
    target = os.path.join(fold_dir, "Detr")
    print(f"[runDetr] Renomeando {src} → {target}, src_exists={os.path.exists(src)}", flush=True)
    os.rename("./Detr", target)

    detr_data = os.path.join(ROOT_DATA_DIR, 'detr')
    print(f"[runDetr] Removendo dados VOC temporários: {detr_data}", flush=True)
    shutil.rmtree(detr_data)
    print(f"[runDetr] Concluído", flush=True)
