#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

from automation.config_loader import load_config
from automation.experiment_grid import generate_experiments
from automation.progress_tracker import ProgressTracker
from automation.results_collector import ResultsCollector
from automation.wrappers.yolov8_wrapper import YOLOV8Wrapper
from automation.wrappers.fasterrcnn_wrapper import FasterRCNNWrapper
from automation.wrappers.detr_wrapper import DETRWrapper
from automation.wrappers.swin_detector_wrapper import SwinDetectorWrapper

def get_wrapper(model_name: str):
    """Get appropriate wrapper for model."""
    if model_name == "YOLOV8":
        return YOLOV8Wrapper()
    elif model_name == "Faster":
        return FasterRCNNWrapper()
    elif model_name == "Detr":
        return DETRWrapper()
    elif model_name == "SwinDetector":
        return SwinDetectorWrapper()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_experiments(config_path: str, dry_run: bool = False, retry_failed: bool = False):
    """Main experiment runner."""
    config = load_config(config_path)
    
    experiments = generate_experiments(config)
    print(f"Generated {len(experiments)} experiments")
    
    progress_file = Path(__file__).parent / 'state' / 'progress.json'
    tracker = ProgressTracker(str(progress_file), config.experiment_name)
    tracker.set_total_experiments(len(experiments))
    
    results_file = Path(__file__).parent.parent / 'results' / 'experiments.csv'
    collector = ResultsCollector(str(results_file))
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Total experiments: {len(experiments)}")
        for model_name in config.models.keys():
            count = sum(1 for e in experiments if e.model == model_name)
            print(f"  {model_name}: {count} experiments")
        print(f"\nExperiment plan saved to: {progress_file.parent / 'experiment_plan.json'}")
        
        import json
        plan = [e.to_dict() for e in experiments]
        with open(progress_file.parent / 'experiment_plan.json', 'w') as f:
            json.dump(plan, f, indent=2)
        
        return
    
    print(f"\nStarting experiments...")
    print(f"Completed: {tracker.get_completed_count()}/{len(experiments)}")
    print(f"Failed: {tracker.get_failed_count()}/{len(experiments)}")
    
    for i, experiment in enumerate(experiments, 1):
        if tracker.is_completed(experiment.experiment_id):
            print(f"[{i}/{len(experiments)}] Skipping {experiment.experiment_id} (already completed)")
            continue
        
        if tracker.is_failed(experiment.experiment_id) and not retry_failed:
            print(f"[{i}/{len(experiments)}] Skipping {experiment.experiment_id} (failed)")
            continue
        
        print(f"\n[{i}/{len(experiments)}] Running {experiment.experiment_id}")
        
        try:
            wrapper = get_wrapper(experiment.model)
            
            start_time = time.time()
            
            if experiment.model == "Faster":
                model_path, metrics, arch_type = wrapper.train(experiment)
            else:
                model_path, metrics = wrapper.train(experiment)
                arch_type = "resnet"
            
            training_time = time.time() - start_time
            
            collector.write_result(
                experiment, model_path, metrics,
                architecture_type=arch_type,
                training_time=training_time
            )
            
            tracker.mark_completed(experiment.experiment_id, model_path, metrics)
            
            print(f"✓ Completed in {training_time:.1f}s")
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed: {error_msg}")
            
            if not config.execution.continue_on_error:
                raise
            
            tracker.mark_failed(experiment.experiment_id, error_msg)
    
    print(f"\n=== EXPERIMENTS COMPLETE ===")
    print(f"Completed: {tracker.get_completed_count()}/{len(experiments)}")
    print(f"Failed: {tracker.get_failed_count()}/{len(experiments)}")
    print(f"Results saved to: {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Run detection experiments")
    parser.add_argument('--config', default='automation/config/experiment.yaml',
                       help='Path to config file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate config and show experiment plan without running')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry failed experiments')
    
    args = parser.parse_args()
    
    run_experiments(args.config, args.dry_run, args.retry_failed)

if __name__ == '__main__':
    main()
