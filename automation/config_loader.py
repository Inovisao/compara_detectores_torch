import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

@dataclass
class DatasetConfig:
    coco_json: str
    images_dir: str
    output_dir: str

@dataclass
class FoldsConfig:
    n_folds: int = 5
    val_percentage: float = 0.3

@dataclass
class ExecutionConfig:
    dry_run: bool = False
    continue_on_error: bool = True
    retry_failed: bool = False
    log_level: str = "INFO"

@dataclass
class ModelConfig:
    architectures: List[str]
    learning_rates: List[float]
    optimizers: List[str]
    batch_sizes: List[int]
    weight_decays: List[float]
    schedulers: List[str]
    epochs: List[int]
    patience: List[int]
    augmentations: Dict[str, List[Any]] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    dataset: DatasetConfig
    folds: FoldsConfig
    models: Dict[str, ModelConfig]
    execution: ExecutionConfig

def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['experiment_name', 'seed', 'dataset', 'folds', 'models']
    for field_name in required_fields:
        if field_name not in raw_config:
            raise ValueError(f"Missing required field: {field_name}")
    
    # Parse dataset config
    dataset = DatasetConfig(**raw_config['dataset'])
    
    # Parse folds config
    folds = FoldsConfig(**raw_config.get('folds', {}))
    
    # Parse execution config
    execution = ExecutionConfig(**raw_config.get('execution', {}))
    
    # Parse models config
    models = {}
    for model_name, model_config in raw_config['models'].items():
        models[model_name] = ModelConfig(**model_config)
    
    return ExperimentConfig(
        experiment_name=raw_config['experiment_name'],
        seed=raw_config['seed'],
        dataset=dataset,
        folds=folds,
        models=models,
        execution=execution
    )
