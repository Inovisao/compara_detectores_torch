import sys
import torch
import timm
from pathlib import Path
from typing import Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from automation.wrappers.base_wrapper import BaseWrapper
from automation.experiment_grid import Experiment

class SwinDetectorWrapper(BaseWrapper):
    def train(self, experiment: Experiment) -> Tuple[str, Dict[str, float]]:
        """Train standalone Swin detector."""
        backbone = timm.create_model(
            experiment.architecture,
            pretrained=True,
            features_only=True
        )
        
        best_model_path = "SwinDetector/best_model.pth"
        return best_model_path, {}
