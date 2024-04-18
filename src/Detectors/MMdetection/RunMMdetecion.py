import os
import subprocess

def runMMdetection(model,fold_dir):
    model = model.split('/')[1]
    path = model.split('-')[0]
    os.system(f'mim download mmdet --config {model} --dest Detectors/MMdetection/checkpoints')
    from Detectors.MMdetection.ConfigTrain import configTrain
    from Detectors.MMdetection.CheckPoint import Checkpoint
    checkpoint = Checkpoint(path)
    configTrain(model=model,checkpoint=checkpoint,path=path)
    save_model = os.path.join(fold_dir,model)
    run_model = os.path.join('Detectors','MMdetection','mmdetection','configs',path,'train.py')
    treino = os.path.join('Detectors','MMdetection','Treinommdetection.sh')
    args = [run_model,save_model]
    subprocess.run([treino]+args)