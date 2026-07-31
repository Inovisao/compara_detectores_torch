from dataclasses import dataclass
from typing import List, Dict, Any
import itertools
from automation.config_loader import ExperimentConfig

@dataclass
class Experiment:
    experiment_id: str
    model: str
    architecture: str
    learning_rate: float
    optimizer: str
    batch_size: int
    weight_decay: float
    scheduler: str
    epochs: int
    patience: int
    augmentations: Dict[str, Any]
    fold: int
    seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'experiment_id': self.experiment_id,
            'model': self.model,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'scheduler': self.scheduler,
            'epochs': self.epochs,
            'patience': self.patience,
            'fold': self.fold,
            'seed': self.seed
        }
        for aug_name, aug_value in self.augmentations.items():
            result[f'augmentation_{aug_name}'] = aug_value
        return result

def generate_experiment_id(model: str, architecture: str, lr: float, optimizer: str,
                          batch_size: int, scheduler: str, fold: int, seed: int,
                          augmentations: Dict[str, Any]) -> str:
    aug_str = "_".join(f"{k}{v}" for k, v in sorted(augmentations.items()))
    return f"{model}_{architecture}_lr{lr}_{optimizer}_bs{batch_size}_{scheduler}_{aug_str}_fold{fold}_seed{seed}"

def generate_experiments(config: ExperimentConfig) -> List[Experiment]:
    experiments = []
    
    for model_name, model_config in config.models.items():
        hyperparam_combos = list(itertools.product(
            model_config.architectures,
            model_config.learning_rates,
            model_config.optimizers,
            model_config.batch_sizes,
            model_config.weight_decays,
            model_config.schedulers,
            model_config.epochs,
            model_config.patience
        ))
        
        if model_config.augmentations:
            aug_keys = list(model_config.augmentations.keys())
            aug_values = list(itertools.product(*[model_config.augmentations[k] for k in aug_keys]))
            aug_combos = [dict(zip(aug_keys, v)) for v in aug_values]
        else:
            aug_combos = [{}]
        
        for arch, lr, opt, bs, wd, sched, ep, pat in hyperparam_combos:
            for aug_combo in aug_combos:
                for fold in range(1, config.folds.n_folds + 1):
                    exp_id = generate_experiment_id(
                        model_name, arch, lr, opt, bs, sched, fold, config.seed, aug_combo
                    )
                    
                    experiments.append(Experiment(
                        experiment_id=exp_id,
                        model=model_name,
                        architecture=arch,
                        learning_rate=lr,
                        optimizer=opt,
                        batch_size=bs,
                        weight_decay=wd,
                        scheduler=sched,
                        epochs=ep,
                        patience=pat,
                        augmentations=aug_combo,
                        fold=fold,
                        seed=config.seed
                    ))
    
    return experiments
