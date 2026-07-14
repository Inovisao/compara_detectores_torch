"""
Testes para os fixes aplicados ao pipeline DETR:
1. Clamping de boxes com w/h negativos no matcher e no loss
2. valid_loader criado com num_workers=0
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── helpers ───────────────────────────────────────────────────────────────────

def _detr_path(*parts):
    return SRC_DIR / "Detectors" / "Detr" / "utils" / "detection" / "detr" / Path(*parts)


# ── box_ops ───────────────────────────────────────────────────────────────────

class TestBoxOps:
    def _import(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "box_ops", _detr_path("box_ops.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_valid_boxes_pass_assert(self):
        box_ops = self._import()
        boxes1 = torch.tensor([[0.1, 0.1, 0.5, 0.5],
                                [0.2, 0.2, 0.8, 0.9]])
        boxes2 = torch.tensor([[0.0, 0.0, 0.4, 0.4]])
        result = box_ops.generalized_box_iou(boxes1, boxes2)
        assert result.shape == (2, 1)

    def test_cxcywh_with_positive_wh(self):
        box_ops = self._import()
        # cx=0.5, cy=0.5, w=0.4, h=0.4  →  [0.3, 0.3, 0.7, 0.7]
        inp = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        out = box_ops.box_cxcywh_to_xyxy(inp)
        assert (out[:, 2:] >= out[:, :2]).all(), "x2 should be >= x1 for positive w/h"

    def test_cxcywh_with_negative_wh_produces_invalid_box(self):
        """Documenta que box_cxcywh_to_xyxy sozinha não corrige w/h negativo."""
        box_ops = self._import()
        inp = torch.tensor([[0.5, 0.5, -0.2, -0.2]])
        out = box_ops.box_cxcywh_to_xyxy(inp)
        # sem clamp: x2 < x1
        assert not (out[:, 2:] >= out[:, :2]).all()


# ── matcher clamp fix ─────────────────────────────────────────────────────────

class TestMatcherClamp:
    def _import_matcher(self):
        import importlib.util, types

        # stub scipy
        scipy_stub = types.ModuleType("scipy")
        scipy_opt = types.ModuleType("scipy.optimize")
        scipy_opt.linear_sum_assignment = lambda c: ([], [])
        scipy_stub.optimize = scipy_opt
        sys.modules.setdefault("scipy", scipy_stub)
        sys.modules.setdefault("scipy.optimize", scipy_opt)

        # box_ops must be on sys.path
        box_ops_dir = str(_detr_path("box_ops.py").parent)
        if box_ops_dir not in sys.path:
            sys.path.insert(0, box_ops_dir)

        spec = importlib.util.spec_from_file_location(
            "matcher_test", _detr_path("matcher.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_matcher_survives_negative_wh_predictions(self):
        """HungarianMatcher não deve lançar AssertionError com w/h negativos."""
        matcher_mod = self._import_matcher()
        matcher = matcher_mod.HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)

        bs, nq, nc = 2, 10, 2
        # predições com w/h negativos (situação real nas primeiras épocas)
        pred_boxes = torch.randn(bs, nq, 4)
        pred_boxes[..., 2:] = -torch.abs(pred_boxes[..., 2:])  # force negative w/h

        outputs = {
            "pred_logits": torch.randn(bs, nq, nc),
            "pred_boxes": pred_boxes,
        }
        targets = [
            {"labels": torch.tensor([0, 1]), "boxes": torch.tensor([[0.3, 0.3, 0.5, 0.5],
                                                                      [0.6, 0.6, 0.8, 0.8]])},
            {"labels": torch.tensor([1]),    "boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]])},
        ]

        # deve rodar sem AssertionError
        try:
            matcher(outputs, targets)
        except AssertionError as e:
            pytest.fail(f"AssertionError após clamp fix: {e}")

    def test_matcher_normal_predictions(self):
        """Matcher funciona normalmente com predições válidas."""
        matcher_mod = self._import_matcher()
        matcher = matcher_mod.HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)

        bs, nq, nc = 1, 5, 2
        pred_boxes = torch.zeros(bs, nq, 4)
        pred_boxes[..., 2:] = 0.3  # w=0.3, h=0.3 (positive)

        outputs = {
            "pred_logits": torch.randn(bs, nq, nc),
            "pred_boxes": pred_boxes,
        }
        targets = [{"labels": torch.tensor([0]),
                    "boxes": torch.tensor([[0.4, 0.4, 0.6, 0.6]])}]

        result = matcher(outputs, targets)
        assert len(result) == bs


# ── valid_loader num_workers ──────────────────────────────────────────────────

class TestValidLoaderWorkers:
    def test_valid_loader_uses_zero_workers(self):
        """create_valid_loader deve ser chamado com num_workers=0 em safe_create_loaders."""
        import ast

        src = (SRC_DIR / "Detectors" / "Detr" / "train_detector.py").read_text()
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "safe_create_loaders":
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func_name = ""
                if isinstance(call.func, ast.Name):
                    func_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    func_name = call.func.attr
                if func_name == "create_valid_loader":
                    # third positional arg (index 2) should be 0
                    if len(call.args) >= 3:
                        arg = call.args[2]
                        assert isinstance(arg, ast.Constant) and arg.value == 0, \
                            f"create_valid_loader terceiro arg deveria ser 0, got {ast.dump(arg)}"
                    else:
                        # check keyword 'num_workers'
                        kw = {k.arg: k.value for k in call.keywords}
                        assert "num_workers" in kw, "num_workers não passado para create_valid_loader"
                        val = kw["num_workers"]
                        assert isinstance(val, ast.Constant) and val.value == 0, \
                            f"num_workers deveria ser 0, got {ast.dump(val)}"
                    return  # found and checked

        pytest.fail("safe_create_loaders ou create_valid_loader não encontrado em train_detector.py")
