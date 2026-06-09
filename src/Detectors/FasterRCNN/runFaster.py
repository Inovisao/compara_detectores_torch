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
    print(f"[runFaster] fold={fold} fold_dir={fold_dir} root_data_dir={root_data_dir}", flush=True)
    dataset_config = geredata(fold, root_data_dir)
    print(f"[runFaster] train_dir={dataset_config.train_dir}", flush=True)
    print(f"[runFaster] train_ann={dataset_config.train_annotations}", flush=True)
    print(f"[runFaster] val_dir={dataset_config.val_dir}", flush=True)
    print(f"[runFaster] val_ann={dataset_config.val_annotations}", flush=True)

    config.configure_dataset(
        dataset_config.train_dir,
        dataset_config.train_annotations,
        dataset_config.val_dir,
        dataset_config.val_annotations,
    )

    target_dir = os.path.abspath(os.path.join(fold_dir, 'Faster'))
    os.makedirs(target_dir, exist_ok=True)
    print(f"[runFaster] OUT_DIR={target_dir}", flush=True)

    treino = os.path.join('Detectors', 'FasterRCNN', 'TreinoFaster.sh')
    treino_abs = os.path.abspath(treino)
    print(f"[runFaster] script={treino_abs} exists={os.path.exists(treino_abs)}", flush=True)

    env = _prepare_environment(dataset_config, fold)
    env['FASTER_OUT_DIR'] = target_dir
    print(f"[runFaster] Executando subprocess: {treino}", flush=True)
    result = subprocess.run([treino], check=True, env=env)
    print(f"[runFaster] Concluído, código={result.returncode}, checkpoint em {target_dir}", flush=True)
