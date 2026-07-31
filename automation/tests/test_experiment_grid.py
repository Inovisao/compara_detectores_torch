from automation.config_loader import ExperimentConfig, DatasetConfig, FoldsConfig, ModelConfig, ExecutionConfig
from automation.experiment_grid import generate_experiments, Experiment

def test_generate_single_experiment():
    config = ExperimentConfig(
        experiment_name="test",
        seed=42,
        dataset=DatasetConfig(coco_json="test.json", images_dir="imgs", output_dir="out"),
        folds=FoldsConfig(n_folds=1, val_percentage=0.3),
        models={
            "YOLOV8": ModelConfig(
                architectures=["yolov8s"],
                learning_rates=[0.001],
                optimizers=["AdamW"],
                batch_sizes=[32],
                weight_decays=[0.0005],
                schedulers=["cosine"],
                epochs=[10],
                patience=[5],
                augmentations={}
            )
        },
        execution=ExecutionConfig()
    )
    
    experiments = generate_experiments(config)
    
    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.model == "YOLOV8"
    assert exp.architecture == "yolov8s"
    assert exp.learning_rate == 0.001
    assert exp.fold == 1
    assert exp.seed == 42

def test_generate_multiple_experiments():
    config = ExperimentConfig(
        experiment_name="test",
        seed=42,
        dataset=DatasetConfig(coco_json="test.json", images_dir="imgs", output_dir="out"),
        folds=FoldsConfig(n_folds=2, val_percentage=0.3),
        models={
            "YOLOV8": ModelConfig(
                architectures=["yolov8s", "yolov8m"],
                learning_rates=[0.001, 0.0001],
                optimizers=["AdamW"],
                batch_sizes=[32],
                weight_decays=[0.0005],
                schedulers=["cosine"],
                epochs=[10],
                patience=[5],
                augmentations={}
            )
        },
        execution=ExecutionConfig()
    )
    
    experiments = generate_experiments(config)
    
    # 2 architectures × 2 LRs × 2 folds = 8 experiments
    assert len(experiments) == 8
