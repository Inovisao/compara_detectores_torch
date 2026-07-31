from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Detector(ABC):
    @abstractmethod
    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        config: dict,
        output_dir: Path,
    ) -> Path:
        """Run training. Return path to best checkpoint.

        config dict contains:
            architecture: str    # e.g. "yolov8s"
            num_classes: int
            + all keys from default_hparams() overridden by YAML/CLI
        """

    @abstractmethod
    def predict(self, images: list) -> list:
        """images: list of torch.Tensor [C,H,W]. Returns list of [{boxes, scores, labels}] where:
        - boxes: Tensor [N, 4] in xyxy format
        - scores: Tensor [N]
        - labels: Tensor [N] int64 class ids
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model weights from checkpoint path."""

    @classmethod
    @abstractmethod
    def architectures(cls) -> list[str]:
        """Supported architecture variant names (e.g. ['yolov8n', 'yolov8s'])."""

    @classmethod
    @abstractmethod
    def default_hparams(cls) -> dict:
        """Per-family hyperparameter defaults with type-annotated descriptions.

        Each value should be a dict with 'default' and 'help' keys.
        Example: {'lr': {'default': 0.001, 'help': '(float) initial learning rate'}, ...}
        """
