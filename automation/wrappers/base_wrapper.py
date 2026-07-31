from abc import ABC, abstractmethod
from typing import Tuple, Dict
from automation.experiment_grid import Experiment

class BaseWrapper(ABC):
    """Abstract base class for model training wrappers."""
    
    @abstractmethod
    def train(self, experiment: Experiment) -> Tuple[str, Dict[str, float]]:
        """
        Train model with given experiment configuration.
        
        Args:
            experiment: Experiment configuration
            
        Returns:
            Tuple of (best_model_paths, metrics_dict)
        """
        pass
