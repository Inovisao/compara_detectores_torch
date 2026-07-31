import sys
import subprocess
from pathlib import Path
from typing import Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from automation.wrappers.base_wrapper import BaseWrapper
from automation.experiment_grid import Experiment

try:
    from Detectors.Detr.GeraDobras import convert_coco_to_voc
except ImportError:
    convert_coco_to_voc = None

class DETRWrapper(BaseWrapper):
    def train(self, experiment: Experiment) -> Tuple[str, Dict[str, float]]:
        """Train DETR model with experiment configuration."""
        convert_coco_to_voc(experiment.fold)
        
        cmd = [
            sys.executable,
            'Detectors/Detr/train_detector.py',
            '--epochs', str(experiment.epochs),
            '--batch', str(experiment.batch_size),
            '--learning-rate', str(experiment.learning_rate),
            '--lr-backbone', str(experiment.learning_rate * 0.1),
            '--weight-decay', str(experiment.weight_decay),
            '--model', experiment.architecture,
            '--seed', str(experiment.seed),
            '--device', 'cuda',
            '--name', experiment.experiment_id
        ]
        
        result = subprocess.run(cmd, check=True, cwd='src')
        
        best_model_path = "Detr/training/best_model.pth"
        return best_model_path, {}
