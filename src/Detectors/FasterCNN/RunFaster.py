import os
import shutil
import subprocess
from Detectors.FasterCNN.GeraDobras import convert_coco_to_voc

# Função para Rodar a rede 
def runFaster(fold,fold_dir,ROOT_DATA_DIR):

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    if not os.path.exists('FasterCNN'):
        os.makedirs('FasterCNN')
    if os.path.exists(os.path.join(fold_dir,"FasterCNN")):  
        shutil.rmtree(os.path.join(fold_dir,"FasterCNN")) 
    convert_coco_to_voc(fold)
    treino = os.path.join('Detectors','FasterCNN','TreinoFaster.sh')
    subprocess.run([treino]) # Roda o bash para treino
    os.rename("./FasterCNN", os.path.join(fold_dir,"FasterCNN"))
    shutil.rmtree(os.path.join(ROOT_DATA_DIR,'faster'))
    shutil.rmtree('runs')