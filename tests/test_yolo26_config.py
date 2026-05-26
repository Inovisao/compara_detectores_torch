from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from Detectors.YOLO26.config import get_finetune_params


def test_finetune_patience_uses_shared_default(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO26_FT_PATIENCE", "25")

    params = get_finetune_params(tmp_path / "data.yaml", "yolo26n.pt", fold="fold_1")

    assert params["phase_a"]["patience"] == 25
    assert params["phase_b"]["patience"] == 25


def test_finetune_patience_can_be_overridden_per_phase(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO26_FT_PATIENCE", "25")
    monkeypatch.setenv("YOLO26_FT_A_PATIENCE", "40")
    monkeypatch.setenv("YOLO26_FT_B_PATIENCE", "8")

    params = get_finetune_params(tmp_path / "data.yaml", "yolo26n.pt")

    assert params["phase_a"]["patience"] == 40
    assert params["phase_b"]["patience"] == 8
