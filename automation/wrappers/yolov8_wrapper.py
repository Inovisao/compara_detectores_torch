import sys
import os
from pathlib import Path
from typing import Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from automation.wrappers.base_wrapper import BaseWrapper
from automation.experiment_grid import Experiment

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from Detectors.YOLOV8.GeraLabels import CriarLabelsYOLOV8
except ImportError:
    CriarLabelsYOLOV8 = None

try:
    from Detectors.YOLOV8.TrocaSettings import Settings
except ImportError:
    Settings = None

class YOLOV8Wrapper(BaseWrapper):
    def train(self, experiment: Experiment) -> Tuple[str, Dict[str, float]]:
        """Train YOLOV8 model with experiment configuration."""
        Settings()
        CriarLabelsYOLOV8(experiment.fold)
        
        model = YOLO(f"{experiment.architecture}.pt")
        
        model.train(
            data='../dataset/all/data.yaml',
            lr0=experiment.learning_rate,
            optimizer=experiment.optimizer,
            batch=experiment.batch_size,
            epochs=experiment.epochs,
            patience=experiment.patience,
            weight_decay=experiment.weight_decay,
            cos_lr=(experiment.scheduler == 'cosine'),
            mosaic=experiment.augmentations.get('mosaic', 0.0),
            flipud=experiment.augmentations.get('flipud', 0.0),
            degrees=experiment.augmentations.get('degrees', 0.0),
            seed=experiment.seed,
            project='YOLOV8',
            name=experiment.experiment_id,
            exist_ok=True,
            plots=True,
            single_cls=False,
            rect=False,
            imgsz=640
        )
        
        best_model_path = 'YOLOV8/train/weights/best.pt'
        
        return best_model_path, {}
