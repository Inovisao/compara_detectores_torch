import os
from pathlib import Path
from Detectors.FasterRCNN.geradataset import geredata
from Detectors.FasterRCNN import config
import shutil
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_TILE_ROOT = PROJECT_ROOT / 'dataset' / 'tiles'


def runFaster(fold, fold_dir, ROOT_DATA_DIR):
    geredata(fold)
    config.init_dataset(fold)
    treino = os.path.join('Detectors', 'FasterRCNN', 'TreinoFaster.sh')
    if os.path.exists(os.path.join(fold_dir, 'Faster')):
        shutil.rmtree(os.path.join(fold_dir, "Faster"))
    env = os.environ.copy()
    env['FASTER_FOLD'] = fold
    subprocess.run([treino], check=True, env=env)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    os.rename('Faster', os.path.join(fold_dir, 'Faster'))
    shutil.rmtree(os.path.join(DATASET_TILE_ROOT, fold, 'Faster'), ignore_errors=True)
