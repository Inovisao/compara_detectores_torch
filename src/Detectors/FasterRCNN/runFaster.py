import os
import shutil
import subprocess

from Detectors.FasterRCNN.geradataset import geredata


def runFaster(fold, fold_dir, ROOT_DATA_DIR):
    geredata(fold)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    treino = os.path.join(script_dir, 'TreinoFaster.sh')
    output_dir = os.path.join(fold_dir, 'Faster')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cwd = os.path.abspath(os.path.join(script_dir, '..', '..'))
    subprocess.run(['bash', treino], cwd=cwd, check=True)

    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir, exist_ok=True)

    source_output = os.path.join(cwd, 'Faster')
    if not os.path.exists(source_output):
        raise FileNotFoundError(f'Checkpoint do FasterRCNN não encontrado em {source_output}')

    os.rename(source_output, output_dir)

    if os.path.exists(os.path.join(ROOT_DATA_DIR, 'Faster')):
        shutil.rmtree(os.path.join(ROOT_DATA_DIR, 'Faster'))
