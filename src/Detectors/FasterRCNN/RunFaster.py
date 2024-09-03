import os
import shutil
import subprocess
from Detectors.FasterRCNN.GeraDobras import convert_coco_to_voc

# Função para Rodar a rede 
def runFaster(fold,fold_dir,ROOT_DATA_DIR):

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    if not os.path.exists('FasterRCNN'):
        os.makedirs('FasterRCNN')
    if os.path.exists(os.path.join(fold_dir,"FasterRCNN")):  
        shutil.rmtree(os.path.join(fold_dir,"FasterRCNN")) 
    convert_coco_to_voc(fold)
    treino = os.path.join('Detectors','FasterRCNN','TreinoFaster.sh')
    subprocess.run([treino]) # Roda o bash para treino
    os.rename("./FasterRCNN", os.path.join(fold_dir,"FasterRCNN"))
    shutil.rmtree(os.path.join(ROOT_DATA_DIR,'faster'))
