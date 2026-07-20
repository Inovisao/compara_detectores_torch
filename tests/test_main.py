from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

_REAL_OPEN = builtins.open


# Deixa leituras passarem; silencia escritas (evita FileNotFoundError
# nos dirs que main.py tenta criar durante o import do módulo).
def _write_safe_open(file, mode="r", **kwargs):
    if isinstance(mode, str) and "w" in mode:
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=MagicMock())
        mock_cm.__exit__ = MagicMock(return_value=False)
        return mock_cm
    return _REAL_OPEN(file, mode, **kwargs)


def _make_fake_folds(root: Path, folds: list[str] = ("fold_1", "fold_2")) -> None:
    d = root / "filesJSON"
    d.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        for split in ("train", "val", "test"):
            (d / f"{fold}_{split}.json").touch()


# ── fixture: importa main uma vez com todos os efeitos colaterais bloqueados ──

@pytest.fixture(scope="module")
def main_mod(tmp_path_factory):
    fake_root = tmp_path_factory.mktemp("dataset")
    _make_fake_folds(fake_root)

    sys.modules.pop("main", None)

    rd_mock = MagicMock()
    rd_mock.RESULTS_CSV_PATH = fake_root / "results.csv"
    rd_mock.COUNTING_CSV_PATH = fake_root / "counting.csv"
    rd_mock.RESULTS_PATH = fake_root / "results"

    rdc_mock = MagicMock()
    rdc_mock.RESULTS_BY_CLASS_CSV_PATH = fake_root / "rbc.csv"

    detector_mocks = {
        "Detectors.YOLOV8.RunYOLOV8":       MagicMock(),
        "Detectors.FasterRCNN.runFaster":    MagicMock(),
        "Detectors.Detr.runDetr":            MagicMock(),
        "Detectors.YOLOV8.config":           MagicMock(),
        "Detectors.FasterRCNN.config":       MagicMock(),
        "Detectors.Detr.config":             MagicMock(),
        "Detectors.YOLOV11.config":          MagicMock(),
        "Detectors.YOLO26.config":           MagicMock(),
        "Detectors.RetinaNet.config":        MagicMock(),
        "Detectors.SSDLite.config":          MagicMock(),
        "Detectors.YOLOV5_TPH.config":       MagicMock(),
    }

    with (
        patch.dict("sys.modules", {
            "ResultsDetections":        rd_mock,
            "ResultsDetectionsbyclass": rdc_mock,
            **detector_mocks,
        }),
        patch.dict(os.environ, {
            "DATASET_ROOT":  str(fake_root),
            "MODELS_TO_RUN": "YOLOV8,Faster,Detr",
        }),
        patch("shutil.rmtree"),
        patch("os.makedirs"),
        patch("builtins.open", side_effect=_write_safe_open),
    ):
        import main as m

    yield m
    sys.modules.pop("main", None)


# ── TestNormalizeModelName ─────────────────────────────────────────────────────

class TestNormalizeModelName:
    def test_canonical_names_are_unchanged(self, main_mod):
        assert main_mod.normalize_model_name("YOLOV8") == "YOLOV8"
        assert main_mod.normalize_model_name("Faster") == "Faster"
        assert main_mod.normalize_model_name("Detr")   == "Detr"

    def test_aliases_resolve_correctly(self, main_mod):
        assert main_mod.normalize_model_name("FasterRCNN") == "Faster"
        assert main_mod.normalize_model_name("DETR")       == "Detr"

    def test_case_insensitive(self, main_mod):
        assert main_mod.normalize_model_name("yolov8") == "YOLOV8"
        assert main_mod.normalize_model_name("faster") == "Faster"

    def test_unknown_model_raises(self, main_mod):
        with pytest.raises(ValueError, match="não suportado"):
            main_mod.normalize_model_name("ResNet50")

    def test_normalize_models_deduplicates(self, main_mod):
        assert main_mod.normalize_models(["YOLOV8", "yolov8", "YOLOV8"]) == ["YOLOV8"]

    def test_normalize_models_preserves_order(self, main_mod):
        assert main_mod.normalize_models(["Detr", "YOLOV8", "Faster"]) == ["Detr", "YOLOV8", "Faster"]


# ── TestCollectFoldNames ───────────────────────────────────────────────────────

class TestCollectFoldNames:
    def test_returns_sorted_by_number(self, main_mod, tmp_path):
        _make_fake_folds(tmp_path, ["fold_3", "fold_1", "fold_2"])
        result = main_mod._collect_fold_names(tmp_path / "filesJSON")
        assert result == ["fold_1", "fold_2", "fold_3"]

    def test_deduplicates_splits_within_fold(self, main_mod, tmp_path):
        d = tmp_path / "filesJSON"
        d.mkdir()
        for split in ("train", "val", "test"):
            (d / f"fold_1_{split}.json").touch()
        assert main_mod._collect_fold_names(d) == ["fold_1"]

    def test_empty_dir_raises(self, main_mod, tmp_path):
        empty = tmp_path / "filesJSON"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            main_mod._collect_fold_names(empty)


# ── TestTestModelPaths ─────────────────────────────────────────────────────────

class TestTestModelPaths:
    @pytest.mark.parametrize("model,expected_suffix", [
        ("YOLOV8", "best.pt"),
        ("Faster", "best.pth"),
        ("Detr",   "best_model.pth"),
    ])
    def test_path_ends_with_expected_file(self, main_mod, model, expected_suffix):
        path = main_mod.test_model(model, "model_checkpoints/fold_1")
        assert path.endswith(expected_suffix)

    @pytest.mark.parametrize("model", ["YOLOV8", "Faster", "Detr"])
    def test_path_contains_model_name(self, main_mod, model):
        path = main_mod.test_model(model, "model_checkpoints/fold_1")
        assert model in path


# ── TestTrainModelDispatch ─────────────────────────────────────────────────────

class TestTrainModelDispatch:
    @pytest.mark.parametrize("model,runner_module,runner_fn,expected_suffix", [
        ("YOLOV8", "Detectors.YOLOV8.RunYOLOV8",       "runYOLOV8",  "best.pt"),
        ("Faster", "Detectors.FasterRCNN.runFaster",    "runFaster",  "best.pth"),
        ("Detr",   "Detectors.Detr.runDetr",            "runDetr",    "best_model.pth"),
    ])
    def test_calls_correct_runner(
        self, main_mod, tmp_path, model, runner_module, runner_fn, expected_suffix
    ):
        mock_runner = MagicMock()
        fold      = "fold_1"
        fold_dir  = str(tmp_path)
        data_root = str(tmp_path / "dataset")

        with patch.dict("sys.modules", {runner_module: mock_runner}):
            result = main_mod.train_model(model, fold, fold_dir, data_root)

        getattr(mock_runner, runner_fn).assert_called_once_with(fold, fold_dir, data_root)
        assert result.endswith(expected_suffix)

    def test_existing_checkpoint_removed_before_retrain(self, main_mod, tmp_path):
        checkpoint_dir = tmp_path / "YOLOV8"
        checkpoint_dir.mkdir()

        mock_runner = MagicMock()
        with (
            patch.dict("sys.modules", {"Detectors.YOLOV8.RunYOLOV8": mock_runner}),
            patch("shutil.rmtree") as mock_rm,
        ):
            main_mod.train_model("YOLOV8", "fold_1", str(tmp_path), str(tmp_path))

        mock_rm.assert_called_once_with(str(checkpoint_dir))


class TestCheckpointManifest:
    def test_writes_local_and_evaluation_manifests(self, main_mod, tmp_path, monkeypatch):
        dataset_root = tmp_path / "dataset" / "asahi_rect"
        checkpoint = tmp_path / "model_checkpoints" / "fold_1" / "YOLOV8" / "train" / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"weights")
        eval_models = tmp_path / "models"

        monkeypatch.setenv("EVAL_MODELS_ROOT", str(eval_models))

        main_mod._write_checkpoint_manifest(
            model="YOLOV8",
            fold="fold_1",
            dataset_root=str(dataset_root),
            model_path=str(checkpoint),
            fold_dir=str(tmp_path / "model_checkpoints" / "fold_1"),
        )

        assert (tmp_path / "model_checkpoints" / "fold_1" / "YOLOV8" / "manifest.json").is_file()
        assert (eval_models / "asahi_rect" / "fold_1" / "yolo" / "manifest.json").is_file()
