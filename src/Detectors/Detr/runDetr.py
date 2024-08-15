import os
import shutil
import subprocess
from Detectors.Detr.GeraDobras import convert_coco_to_voc

# Função para Rodar a rede 
def runDetr(fold,fold_dir,ROOT_DATA_DIR):

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    convert_coco_to_voc(fold)
    treino = os.path.join('Detectors','Detr','TreinoDetr.sh')
    subprocess.run([treino]) # Roda o bash para treino
    os.rename("./Detr", os.path.join(fold_dir,"Detr"))
    shutil.rmtree(os.path.join(ROOT_DATA_DIR,'detr'))
