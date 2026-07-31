import csv
from pathlib import Path
from typing import Dict, Any
from automation.experiment_grid import Experiment

class ResultsCollector:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.columns = [
            'experiment_id', 'model', 'architecture', 'architecture_type',
            'learning_rate', 'optimizer', 'batch_size', 'weight_decay',
            'scheduler', 'epochs', 'patience', 'fold', 'seed',
            'mAP', 'mAP50', 'mAP75', 'mAP50_95',
            'MAE', 'RMSE', 'r', 'precision', 'recall', 'f1_score',
            'train_loss_final', 'training_time_s', 'num_parameters',
            'best_model_path', 'status', 'timestamp'
        ]
        
        if not self.csv_path.exists():
            self._write_header()
    
    def _write_header(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
    
    def write_result(self, experiment: Experiment, best_model_path: str,
                    metrics: Dict[str, Any], architecture_type: str = "resnet",
                    training_time: float = 0.0, num_parameters: int = 0):
        from datetime import datetime
        
        row = experiment.to_dict()
        row['architecture_type'] = architecture_type
        row['best_model_path'] = best_model_path
        row['status'] = 'completed'
        row['timestamp'] = datetime.now().isoformat()
        row['training_time_s'] = training_time
        row['num_parameters'] = num_parameters
        
        for key, value in metrics.items():
            row[key] = value
        
        for col in self.columns:
            if col not in row:
                row[col] = 'N/A'
        
        filtered_row = {col: row[col] for col in self.columns}
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(filtered_row)
