from pathlib import Path

from engine.sweeper import experiment_output_dir, is_experiment_complete


def test_experiment_output_dir_separates_hyperparameter_combinations(tmp_path):
    experiment = {
        "experiment": "run",
        "detector": "faster_rcnn",
        "architecture": "resnet50",
        "fold": 0,
        "combo_name": "faster_rcnn_resnet50_lr=0.001_optimizer=SGD",
    }

    output = experiment_output_dir(tmp_path, experiment)

    assert output == tmp_path / "run" / "fold_1" / "resnet50" / experiment["combo_name"]


def test_experiment_is_complete_requires_checkpoint_and_metrics(tmp_path):
    assert not is_experiment_complete(tmp_path)
    (tmp_path / "best.pth").touch()
    assert not is_experiment_complete(tmp_path)
    (tmp_path / "metrics.json").touch()
    assert is_experiment_complete(tmp_path)
