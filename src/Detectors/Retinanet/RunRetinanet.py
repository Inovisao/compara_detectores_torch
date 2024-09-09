import os
import shutil
import subprocess
from Detectors.Retinanet.geraDobras import convert_coco_to_custom_xml

# Função para Rodar a rede 
def runRetinanet(fold,fold_dir,ROOT_DATA_DIR):

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    if os.path.exists(os.path.join(fold_dir,"Retinanet")):  
        shutil.rmtree(os.path.join(fold_dir,"Retinanet")) 
    convert_coco_to_custom_xml(fold)
    treino = os.path.join('Detectors','Retinanet','TreinoRetinanet.sh')
    subprocess.run([treino]) # Roda o bash para treino
    os.rename("./outputs", os.path.join(fold_dir,"Retinanet"))
    shutil.rmtree(os.path.join(ROOT_DATA_DIR,'retinanet'))
