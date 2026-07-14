"""
Testes para o pipeline de avaliação (ResultsDetections.py).

Foco: pontos de quebra silenciosa que, com CONTINUE=True / sem retreinamento,
causam perda permanente do resultado do fold.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# Importa no nível do módulo para evitar circular import do torchvision
# quando importado dentro de métodos de teste
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de fixture
# ──────────────────────────────────────────────────────────────────────────────

def _make_coco_json(path: Path, n_images: int = 3, n_ann_per_image: int = 2) -> None:
    images, annotations = [], []
    ann_id = 1
    for i in range(1, n_images + 1):
        images.append({"id": i, "file_name": f"img_{i:03d}.jpg", "width": 640, "height": 640})
        for _ in range(n_ann_per_image):
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": 1,
                "bbox": [10.0, 10.0, 50.0, 50.0], "area": 2500.0, "iscrowd": 0,
            })
            ann_id += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "boi"}],
        "images": images,
        "annotations": annotations,
    }), encoding="utf-8")


def _make_fake_images(directory: Path, file_names: list[str]) -> None:
    import numpy as np, cv2
    directory.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    for name in file_names:
        cv2.imwrite(str(directory / name), blank)


# ──────────────────────────────────────────────────────────────────────────────
# Importa funções alvo mockando dependências pesadas
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rd():
    """Importa ResultsDetections com todos os detectors mockados."""
    heavy = [
        "Detectors.YOLOV5_TPH.DetectionsYOLOV5TPH",
        "Detectors.YOLOV8.DetectionsYolov8",
        "Detectors.YOLOV11.DetectionsYOLOV11",
        "Detectors.YOLO26.DetectionsYOLO26",
        "Detectors.RetinaNet.DetectionsRetinaNet",
        "Detectors.SSDLite.DetectionsSSDLite",
        "Detectors.FasterRCNN.inference",
        "Detectors.FasterRCNN.geradataset",
        "Detectors.FasterRCNN.config",
        "Detectors.Detr.DetectionsDetr",
        "Detectors.ViT.DetectionsViT",
        "sage",
    ]
    mocks = {m: MagicMock() for m in heavy}
    mocks["sage"].detect_sage_dataset = MagicMock(return_value=False)
    mocks["sage"].SageAggregator = MagicMock()

    sys.modules.pop("ResultsDetections", None)
    with patch.dict("sys.modules", mocks):
        import ResultsDetections as _rd
    yield _rd
    sys.modules.pop("ResultsDetections", None)


# ──────────────────────────────────────────────────────────────────────────────
# 1. _resolve_test_split — estrutura filesJSON (novo dataset SAHI)
# ──────────────────────────────────────────────────────────────────────────────

class TestResolveTestSplit:
    def test_finds_test_json_when_filesjson_exists(self, rd, tmp_path):
        fj = tmp_path / "filesJSON"
        fj.mkdir()
        (fj / "fold_1_test.json").touch()
        (tmp_path / "fold_1" / "test" / "images").mkdir(parents=True)

        json_path, images_dir = rd._resolve_test_split(str(tmp_path), "fold_1")

        assert Path(json_path) == fj / "fold_1_test.json"
        assert Path(images_dir) == tmp_path / "fold_1" / "test" / "images"

    def test_raises_when_test_json_missing(self, rd, tmp_path):
        (tmp_path / "filesJSON").mkdir()
        (tmp_path / "test").mkdir()

        with pytest.raises(FileNotFoundError, match="fold_1_test.json"):
            rd._resolve_test_split(str(tmp_path), "fold_1")

    def test_raises_when_no_filesjson_and_no_split_dirs(self, rd, tmp_path):
        with pytest.raises(FileNotFoundError):
            rd._resolve_test_split(str(tmp_path), "fold_1")

    def test_falls_back_to_test_dir_without_filesjson(self, rd, tmp_path):
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        ann = test_dir / "_annotations.coco.json"
        ann.touch()

        json_path, images_dir = rd._resolve_test_split(str(tmp_path), "fold_1")

        assert Path(json_path) == ann
        assert Path(images_dir) == test_dir


# ──────────────────────────────────────────────────────────────────────────────
# 2. Checkpoint: deve existir antes da inferência começar
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointExists:
    def test_missing_checkpoint_is_detected(self, tmp_path):
        checkpoint = tmp_path / "best.pth"
        assert not checkpoint.exists(), "Checkpoint não deveria existir antes do treino"

    def test_checkpoint_exists_after_training(self, tmp_path):
        checkpoint = tmp_path / "best.pth"
        checkpoint.write_bytes(b"fake")
        assert checkpoint.exists()

    def test_torch_load_raises_on_missing_file(self):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            torch.load("/nonexistent/path/best.pth", map_location="cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 3. PearsonCorrCoef — NaN com predições constantes escreve lixo no CSV
# ──────────────────────────────────────────────────────────────────────────────

class TestPearsonNaN:
    """Confirma o comportamento de NaN e verifica que o CSV não recebe NaN."""

    def test_constant_predictions_produce_nan(self):
        pearson = PearsonCorrCoef()
        preds = torch.zeros(5)
        gt = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0])
        result = pearson(preds, gt)
        assert torch.isnan(result), "Esperado NaN com predições constantes"

    def test_both_zero_produce_nan(self):
        pearson = PearsonCorrCoef()
        result = pearson(torch.zeros(4), torch.zeros(4))
        assert torch.isnan(result)

    def test_nan_not_written_to_csv(self, rd, tmp_path):
        """create_csv deve substituir NaN por 0.0 antes de escrever."""
        results_csv = tmp_path / "results.csv"

        fake_metrics = (0.5, 0.5, 0.3, 1.2, 1.5, 0.6, 0.5, 0.55, float("nan"))

        with (
            patch.object(rd, "generate_results", return_value=fake_metrics),
            patch.object(rd, "RESULTS_CSV_PATH", results_csv),
        ):
            rd.create_csv("YOLOV8", "fold_1", str(tmp_path), "model.pt", False)

        content = results_csv.read_text()
        assert "nan" not in content.lower(), (
            "NaN escrito no CSV — corrija create_csv para substituir por 0.0"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. create_csv — não deve engolir exceções silenciosamente
# ──────────────────────────────────────────────────────────────────────────────

class TestCreateCsvDoesNotSwallowErrors:
    def test_exception_in_generate_results_is_not_silent(self, rd, tmp_path):
        """
        Se generate_results falhar, create_csv não pode simplesmente imprimir
        e continuar — o fold seria perdido sem reprocessamento possível.
        O CSV não deve conter nenhuma linha para esse fold.
        """
        results_csv = tmp_path / "results.csv"

        with (
            patch.object(rd, "generate_results", side_effect=RuntimeError("model load failed")),
            patch.object(rd, "RESULTS_CSV_PATH", results_csv),
        ):
            rd.create_csv("YOLOV8", "fold_1", str(tmp_path), "missing.pt", False)

        # O CSV não deve existir ou deve estar vazio (sem linha de dados)
        if results_csv.exists():
            lines = [l for l in results_csv.read_text().splitlines() if l.strip()]
            data_lines = [l for l in lines if not l.startswith("ml,")]
            assert len(data_lines) == 0, (
                "create_csv escreveu linha de resultado mesmo com erro — "
                "resultado inválido persistido no CSV"
            )

    def test_missing_image_file_leaves_no_partial_csv_row(self, rd, tmp_path):
        """FileNotFoundError de imagem ausente não deve gerar linha parcial no CSV."""
        results_csv = tmp_path / "results.csv"

        with (
            patch.object(rd, "generate_results", side_effect=FileNotFoundError("img_001.jpg não encontrada")),
            patch.object(rd, "RESULTS_CSV_PATH", results_csv),
        ):
            rd.create_csv("Detr", "fold_2", str(tmp_path), "best_model.pth", False)

        if results_csv.exists():
            lines = [l for l in results_csv.read_text().splitlines() if l.strip()]
            data_lines = [l for l in lines if not l.startswith("ml,")]
            assert len(data_lines) == 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. compute_metrics — edge cases que crasham ou retornam lixo
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_empty_lists_return_zeros(self, rd):
        p, r, f = rd.compute_metrics([], [], num_classes=1)
        assert p == 0.0 and r == 0.0 and f == 0.0

    def test_all_correct_binary(self, rd):
        preds   = [1, 1, 1, 0, 0]
        targets = [1, 1, 1, 0, 0]
        p, r, f = rd.compute_metrics(preds, targets, num_classes=2)
        assert p == pytest.approx(1.0, abs=1e-4)
        assert r == pytest.approx(1.0, abs=1e-4)
        assert f == pytest.approx(1.0, abs=1e-4)

    def test_all_wrong_binary_does_not_crash(self, rd):
        preds   = [1, 1, 1]
        targets = [0, 0, 0]
        p, r, f = rd.compute_metrics(preds, targets, num_classes=2)
        assert isinstance(p, float)
        assert isinstance(r, float)
        assert isinstance(f, float)

    def test_single_prediction_does_not_crash(self, rd):
        p, r, f = rd.compute_metrics([1], [1], num_classes=2)
        assert isinstance(p, float)


# ──────────────────────────────────────────────────────────────────────────────
# 6. process_predictions — divisão por zero quando gt_count == 0
# ──────────────────────────────────────────────────────────────────────────────

class TestProcessPredictions:
    def test_no_crash_when_gt_is_empty(self, rd, tmp_path):
        ground_truth = {"img_001.jpg": []}
        predictions  = {"img_001.jpg": [[10.0, 10.0, 50.0, 50.0, 1, 0.9]]}
        classes      = {1: "boi"}

        with patch.object(rd, "RESULTS_PATH", tmp_path / "prediction"):
            gt_list, pred_list, r = rd.process_predictions(
                ground_truth, predictions, classes,
                save_img=False, images_source=str(tmp_path),
                fold="fold_1", model_name="YOLOV8",
            )

        assert isinstance(r, torch.Tensor)

    def test_no_crash_when_predictions_empty(self, rd, tmp_path):
        ground_truth = {"img_001.jpg": [[10.0, 10.0, 50.0, 50.0, 1]]}
        predictions  = {"img_001.jpg": []}
        classes      = {1: "boi"}

        with patch.object(rd, "RESULTS_PATH", tmp_path / "prediction"):
            gt_list, pred_list, r = rd.process_predictions(
                ground_truth, predictions, classes,
                save_img=False, images_source=str(tmp_path),
                fold="fold_1", model_name="YOLOV8",
            )

        assert isinstance(r, torch.Tensor)

    def test_save_img_with_missing_image_does_not_crash(self, rd, tmp_path):
        """cv2.imread retorna None para imagem ausente — não deve travar."""
        ground_truth = {"inexistente.jpg": [[5.0, 5.0, 20.0, 20.0, 1]]}
        predictions  = {"inexistente.jpg": [[5.0, 5.0, 20.0, 20.0, 1, 0.8]]}
        classes      = {1: "boi"}

        with patch.object(rd, "RESULTS_PATH", tmp_path / "prediction"):
            gt_list, pred_list, r = rd.process_predictions(
                ground_truth, predictions, classes,
                save_img=True, images_source=str(tmp_path),
                fold="fold_1", model_name="YOLOV8",
            )

        assert isinstance(r, torch.Tensor)


# ──────────────────────────────────────────────────────────────────────────────
# 7. MeanAveragePrecision — tensores vazios não travam
# ──────────────────────────────────────────────────────────────────────────────

class TestMeanAveragePrecision:
    def test_empty_predictions_does_not_crash(self):
        metric = MeanAveragePrecision()
        preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.long)}]
        gt    = [{"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "labels": torch.tensor([1])}]
        metric.update(preds, gt)
        result = metric.compute()
        assert "map" in result

    def test_empty_ground_truth_does_not_crash(self):
        metric = MeanAveragePrecision()
        preds = [{"boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}]
        gt    = [{"boxes": torch.zeros((0, 4)), "labels": torch.tensor([], dtype=torch.long)}]
        metric.update(preds, gt)
        result = metric.compute()
        assert "map" in result


# ──────────────────────────────────────────────────────────────────────────────
# 8. load_dataset — integridade do JSON
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadDataset:
    def test_loads_correctly(self, rd, tmp_path):
        json_path = tmp_path / "fold_1_test.json"
        _make_coco_json(json_path, n_images=2, n_ann_per_image=3)

        result = rd.load_dataset(str(json_path))

        assert len(result) == 2
        assert len(result[0]["annotations"]["bboxes"]) == 3

    def test_image_with_no_annotations(self, rd, tmp_path):
        json_path = tmp_path / "fold_1_test.json"
        _make_coco_json(json_path, n_images=1, n_ann_per_image=0)

        result = rd.load_dataset(str(json_path))

        assert result[0]["annotations"]["bboxes"] == []
        assert result[0]["annotations"]["labels"] == []

    def test_missing_json_raises(self, rd, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            rd.load_dataset(str(tmp_path / "nonexistent.json"))
