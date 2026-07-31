import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class ProgressTracker:
    def __init__(self, progress_file: str, experiment_name: str):
        self.progress_file = Path(progress_file)
        self.experiment_name = experiment_name
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load progress from file or create new."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "experiment_name": self.experiment_name,
                "total_experiments": 0,
                "completed": [],
                "failed": []
            }
    
    def _save(self):
        """Save progress to file atomically."""
        temp_file = self.progress_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        temp_file.replace(self.progress_file)
    
    def is_completed(self, experiment_id: str) -> bool:
        """Check if experiment is already completed."""
        return any(exp['id'] == experiment_id for exp in self.data['completed'])
    
    def is_failed(self, experiment_id: str) -> bool:
        """Check if experiment has failed."""
        return any(exp['id'] == experiment_id for exp in self.data['failed'])
    
    def mark_completed(self, experiment_id: str, best_model_path: str, metrics: Dict[str, Any]):
        """Mark experiment as completed."""
        entry = {
            "id": experiment_id,
            "status": "completed",
            "best_model_path": best_model_path,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        self.data['completed'].append(entry)
        self._save()
    
    def mark_failed(self, experiment_id: str, error: str):
        """Mark experiment as failed."""
        entry = {
            "id": experiment_id,
            "status": "failed",
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.data['failed'].append(entry)
        self._save()
    
    def get_completed_count(self) -> int:
        """Get number of completed experiments."""
        return len(self.data['completed'])
    
    def get_failed_count(self) -> int:
        """Get number of failed experiments."""
        return len(self.data['failed'])
    
    def set_total_experiments(self, total: int):
        """Set total number of experiments."""
        self.data['total_experiments'] = total
        self._save()
