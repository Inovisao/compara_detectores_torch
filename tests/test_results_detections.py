import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Stub optional torchmetrics imports so the module can be imported in test environments.
torchmetrics = types.ModuleType("torchmetrics")
detection = types.ModuleType("torchmetrics.detection")
mean_ap = types.ModuleType("torchmetrics.detection.mean_ap")
regression = types.ModuleType("torchmetrics.regression")
classification = types.ModuleType("torchmetrics.classification")

class DummyMetric:
    def __call__(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return None

    def compute(self):
        return {}

class MeanAveragePrecision(DummyMetric):
    pass

class MeanAbsoluteError(DummyMetric):
    pass

class MeanSquaredError(DummyMetric):
    def __init__(self, squared=True):
        self.squared = squared

class PearsonCorrCoef(DummyMetric):
    pass

class MulticlassPrecision(DummyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__()

class MulticlassRecall(DummyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__()

class MulticlassF1Score(DummyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__()

class MulticlassAccuracy(DummyMetric):
    def __init__(self, *args, **kwargs):
        super().__init__()

class BinaryPrecision(DummyMetric):
    pass

class BinaryRecall(DummyMetric):
    pass

class BinaryF1Score(DummyMetric):
    pass

class BinaryAccuracy(DummyMetric):
    pass

mean_ap.MeanAveragePrecision = MeanAveragePrecision
regression.MeanAbsoluteError = MeanAbsoluteError
regression.MeanSquaredError = MeanSquaredError
regression.PearsonCorrCoef = PearsonCorrCoef
classification.MulticlassPrecision = MulticlassPrecision
classification.MulticlassRecall = MulticlassRecall
classification.MulticlassF1Score = MulticlassF1Score
classification.MulticlassAccuracy = MulticlassAccuracy
classification.BinaryPrecision = BinaryPrecision
classification.BinaryRecall = BinaryRecall
classification.BinaryF1Score = BinaryF1Score
classification.BinaryAccuracy = BinaryAccuracy

torchmetrics.detection = detection
torchmetrics.regression = regression
torchmetrics.classification = classification
sys.modules["torchmetrics"] = torchmetrics
sys.modules["torchmetrics.detection"] = detection
sys.modules["torchmetrics.detection.mean_ap"] = mean_ap
sys.modules["torchmetrics.regression"] = regression
sys.modules["torchmetrics.classification"] = classification

detection.mean_ap = mean_ap
mean_ap.MeanAveragePrecision = MeanAveragePrecision

# Stub detector modules that are not required for this CSV regression path.
module_specs = {
    "Detectors.YOLOV8.DetectionsYolov8": {"resultYOLO": types.SimpleNamespace(result=lambda *args, **kwargs: [])},
    "Detectors.FasterRCNN.inference": {"ResultFaster": types.SimpleNamespace(resultFaster=lambda *args, **kwargs: [])},
    "Detectors.Detr.inference_image_detect": {"resultDetr": lambda *args, **kwargs: []},
    "Detectors.mminference.inference": {"runMMdetection": lambda *args, **kwargs: []},
}
for module_name, attrs in module_specs.items():
    module = types.ModuleType(module_name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    sys.modules[module_name] = module

supervision = types.ModuleType("supervision")
supervision_draw = types.ModuleType("supervision.draw")
supervision_color = types.ModuleType("supervision.draw.color")
class ColorPalette:
    pass
supervision_color.ColorPalette = ColorPalette
sys.modules["supervision"] = supervision
sys.modules["supervision.draw"] = supervision_draw
sys.modules["supervision.draw.color"] = supervision_color

import ResultsDetections as rd


def test_create_csv_handles_generate_results_failure(monkeypatch, capsys):
    def fail_generate_results(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rd, "generate_results", fail_generate_results)

    rd.create_csv(
        selected_model="YOLOV8",
        fold="fold_1",
        root="dummy",
        model_path="dummy",
        save_imgs=False,
    )

    captured = capsys.readouterr()
    assert "[ERRO] Falha ao salvar resultados em" in captured.out
    assert "results.csv" in captured.out
    assert "boom" in captured.out
