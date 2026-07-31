import sys
import os
import torch
import torchvision
from pathlib import Path
from typing import Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from automation.wrappers.base_wrapper import BaseWrapper
from automation.experiment_grid import Experiment

try:
    from Detectors.FasterRCNN.geradataset import geredata
except ImportError:
    geredata = None

class FasterRCNNWrapper(BaseWrapper):
    def train(self, experiment: Experiment) -> Tuple[str, Dict[str, float], str]:
        """Train FasterRCNN model with experiment configuration."""
        geredata(experiment.fold)
        
        if experiment.architecture.startswith('swin_'):
            architecture_type = "swin_backbone"
            model = self._build_swin_model(experiment)
        else:
            architecture_type = "resnet"
            model = self._build_resnet_model(experiment)
        
        best_model_path = "Faster/best.pth"
        return best_model_path, {}, architecture_type
    
    def _build_resnet_model(self, experiment: Experiment):
        """Build FasterRCNN with ResNet backbone."""
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        if experiment.architecture == "resnet50":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        elif experiment.architecture == "resnet101":
            model = torchvision.models.detection.fasterrcnn_resnet101_fpn(weights="DEFAULT")
        else:
            raise ValueError(f"Unknown ResNet architecture: {experiment.architecture}")
        
        return model
    
    def _build_swin_model(self, experiment: Experiment):
        """Build FasterRCNN with Swin Transformer backbone."""
        import timm
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        
        backbone = timm.create_model(
            experiment.architecture,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        
        feature_channels = backbone.feature_info.channels()
        
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * len(feature_channels),
            aspect_ratios=((0.5, 1.0, 2.0),) * len(feature_channels)
        )
        
        model = FasterRCNN(
            backbone=backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator
        )
        
        return model
