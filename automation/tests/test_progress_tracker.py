import pytest
from automation.progress_tracker import ProgressTracker

def test_progress_tracker_new_file(tmp_path):
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(str(progress_file), "test_exp")
    
    assert not tracker.is_completed("exp1")
    assert not tracker.is_failed("exp1")
    assert tracker.get_completed_count() == 0

def test_progress_tracker_mark_completed(tmp_path):
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(str(progress_file), "test_exp")
    
    tracker.mark_completed("exp1", "model_path.pt", {"mAP": 0.85})
    
    assert tracker.is_completed("exp1")
    assert tracker.get_completed_count() == 1
    assert not tracker.is_failed("exp1")

def test_progress_tracker_mark_failed(tmp_path):
    progress_file = tmp_path / "progress.json"
    tracker = ProgressTracker(str(progress_file), "test_exp")
    
    tracker.mark_failed("exp1", "CUDA OOM")
    
    assert tracker.is_failed("exp1")
    assert not tracker.is_completed("exp1")

def test_progress_tracker_resume(tmp_path):
    progress_file = tmp_path / "progress.json"
    
    # First run
    tracker1 = ProgressTracker(str(progress_file), "test_exp")
    tracker1.mark_completed("exp1", "path1.pt", {"mAP": 0.85})
    
    # Second run (resume)
    tracker2 = ProgressTracker(str(progress_file), "test_exp")
    assert tracker2.is_completed("exp1")
