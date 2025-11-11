import os
import shutil
import subprocess

from Detectors.FasterRCNN import config
from Detectors.FasterRCNN.geradataset import FasterDatasetConfig, geredata


def _prepare_environment(dataset: FasterDatasetConfig, fold: str) -> dict:
    env = os.environ.copy()
    env['FASTER_TRAIN_DIR'] = str(dataset.train_dir)
    env['FASTER_TRAIN_ANN'] = str(dataset.train_annotations)
    env['FASTER_VAL_DIR'] = str(dataset.val_dir)
    env['FASTER_VAL_ANN'] = str(dataset.val_annotations)
    env['FASTER_FOLD'] = fold
    return env


def runFaster(fold, fold_dir, root_data_dir):
    dataset_config = geredata(fold, root_data_dir)
    # Validate dataset availability in the current process for early feedback
    config.configure_dataset(
        dataset_config.train_dir,
        dataset_config.train_annotations,
        dataset_config.val_dir,
        dataset_config.val_annotations,
    )

    treino = os.path.join('Detectors', 'FasterRCNN', 'TreinoFaster.sh')
    target_dir = os.path.join(fold_dir, 'Faster')
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    env = _prepare_environment(dataset_config, fold)
    subprocess.run([treino], check=True, env=env)

    os.makedirs(fold_dir, exist_ok=True)
    os.rename('Faster', target_dir)
