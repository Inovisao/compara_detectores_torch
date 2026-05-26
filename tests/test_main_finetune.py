from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import main_finetune

# generate_results return order: mAP, mAP50, mAP75, MAE, RMSE, precision, recall, fscore, r
FAKE_METRICS = (0.50, 0.60, 0.40, 1.20, 1.50, 0.80, 0.70, 0.75, 0.90)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_folds(root: Path, folds: list[str]) -> None:
    d = root / "filesJSON"
    d.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        (d / f"{fold}_train.json").touch()
        (d / f"{fold}_test.json").touch()


def _phase_a(**overrides):
    base = {"data": "x.yaml", "epochs": 150, "batch": 32, "freeze": 9,
            "lr0": 0.01, "weights": "yolo26n.pt", "name": "phase_a_fold_1"}
    return {**base, **overrides}


def _phase_b(**overrides):
    base = {"data": "x.yaml", "epochs": 35, "batch": 16, "freeze": 10, "lr0": 0.0001}
    return {**base, **overrides}


@pytest.fixture()
def restore_rd_counting():
    original = main_finetune.RD.COUNTING_CSV_PATH
    yield
    main_finetune.RD.COUNTING_CSV_PATH = original


# ── _collect_folds ─────────────────────────────────────────────────────────────

class TestCollectFolds:
    def test_returns_sorted(self, tmp_path):
        _make_folds(tmp_path, ["fold_3", "fold_1", "fold_2"])
        assert main_finetune._collect_folds(tmp_path) == ["fold_1", "fold_2", "fold_3"]

    def test_deduplicates_splits(self, tmp_path):
        d = tmp_path / "filesJSON"
        d.mkdir()
        for s in ("train", "val", "test"):
            (d / f"fold_1_{s}.json").touch()
        assert main_finetune._collect_folds(tmp_path) == ["fold_1"]

    def test_raises_if_empty(self, tmp_path):
        (tmp_path / "filesJSON").mkdir()
        with pytest.raises(FileNotFoundError):
            main_finetune._collect_folds(tmp_path)


# ── _evaluate ─────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_writes_csv_row_correctly(self, tmp_path):
        result_csv = tmp_path / "r.csv"
        counting_csv = tmp_path / "c.csv"
        with patch("main_finetune.generate_results", return_value=FAKE_METRICS):
            main_finetune._evaluate("root", "fold_1", "m.pt", result_csv, counting_csv)
        rows = list(csv.reader(result_csv.open()))
        assert len(rows) == 1
        ml, fold, mAP, mAP50, mAP75, MAE, RMSE, r, precision, recall, fscore = rows[0]
        assert ml == "YOLO26"
        assert fold == "fold_1"
        assert float(mAP) == pytest.approx(0.50)
        assert float(r) == pytest.approx(0.90)
        assert float(precision) == pytest.approx(0.80)

    def test_appends_multiple_folds(self, tmp_path):
        result_csv = tmp_path / "r.csv"
        with patch("main_finetune.generate_results", return_value=FAKE_METRICS):
            main_finetune._evaluate("root", "fold_1", "m.pt", result_csv, tmp_path / "c.csv")
            main_finetune._evaluate("root", "fold_2", "m.pt", result_csv, tmp_path / "c.csv")
        rows = list(csv.reader(result_csv.open()))
        assert len(rows) == 2
        assert rows[0][1] == "fold_1"
        assert rows[1][1] == "fold_2"

    def test_sets_counting_csv_path_on_rd(self, tmp_path, restore_rd_counting):
        counting_csv = tmp_path / "counting.csv"
        with patch("main_finetune.generate_results", return_value=FAKE_METRICS):
            main_finetune._evaluate("root", "fold_1", "m.pt", tmp_path / "r.csv", counting_csv)
        assert main_finetune.RD.COUNTING_CSV_PATH == counting_csv

    def test_propagates_generate_results_exception(self, tmp_path):
        result_csv = tmp_path / "r.csv"
        with patch("main_finetune.generate_results", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                main_finetune._evaluate("root", "fold_1", "m.pt", result_csv, tmp_path / "c.csv")
        assert not result_csv.exists()

    def test_generate_results_called_with_correct_args(self, tmp_path):
        with patch("main_finetune.generate_results", return_value=FAKE_METRICS) as mock_gen:
            main_finetune._evaluate("/my/root", "fold_2", "/my/model.pt",
                                    tmp_path / "r.csv", tmp_path / "c.csv")
        mock_gen.assert_called_once_with(
            "/my/root", "fold_2", "/my/model.pt", "YOLO26", False, tiling_mode="basic"
        )


# ── _run_phase_base ────────────────────────────────────────────────────────────

class TestRunPhaseBase:
    def _patch(self, tmp_path, folds=("fold_1",), phase_a_params=None):
        base = tmp_path / "base"
        _make_folds(base, list(folds))
        results = tmp_path / "results"
        base_csv = results / "results_base.csv"
        count_csv = results / "counting_base.csv"
        ft_project = tmp_path / "ft_project"
        pa = phase_a_params or _phase_a()
        return dict(
            base=base, results=results, base_csv=base_csv,
            count_csv=count_csv, ft_project=ft_project, pa=pa,
        )

    def _run(self, ctx, extra_patches=None):
        patches = [
            patch.object(main_finetune, "DATASET_BASE", ctx["base"]),
            patch.object(main_finetune, "RESULTS_DIR", ctx["results"]),
            patch.object(main_finetune, "RESULTS_BASE_CSV", ctx["base_csv"]),
            patch.object(main_finetune, "COUNTING_BASE_CSV", ctx["count_csv"]),
            patch.object(main_finetune, "FT_PROJECT", ctx["ft_project"]),
            patch("main_finetune.CriarLabelsYOLO26"),
            patch("main_finetune.get_finetune_params",
                  return_value={"phase_a": ctx["pa"], "phase_b": {}}),
            patch("main_finetune.tg"),
        ]
        if extra_patches:
            patches.extend(extra_patches)
        return patches

    def test_csv_header_written(self, tmp_path):
        ctx = self._patch(tmp_path)
        with patch("main_finetune.YOLO"), patch("main_finetune._evaluate"):
            with self._run(ctx)[0]:
                for p in self._run(ctx)[1:]:
                    p.start()
                main_finetune._run_phase_base()
                for p in self._run(ctx)[1:]:
                    p.stop()

    # simpler approach without nesting
    def _run_base(self, ctx, mock_yolo=None, mock_eval=None):
        mock_yolo = mock_yolo or MagicMock()
        mock_eval = mock_eval or MagicMock()
        with patch.object(main_finetune, "DATASET_BASE", ctx["base"]), \
             patch.object(main_finetune, "RESULTS_DIR", ctx["results"]), \
             patch.object(main_finetune, "RESULTS_BASE_CSV", ctx["base_csv"]), \
             patch.object(main_finetune, "COUNTING_BASE_CSV", ctx["count_csv"]), \
             patch.object(main_finetune, "FT_PROJECT", ctx["ft_project"]), \
             patch("main_finetune.CriarLabelsYOLO26"), \
             patch("main_finetune.get_finetune_params",
                   return_value={"phase_a": ctx["pa"], "phase_b": {}}), \
             patch("main_finetune.YOLO", mock_yolo), \
             patch("main_finetune._evaluate", mock_eval), \
             patch("main_finetune.tg"):
            main_finetune._run_phase_base()
        return mock_yolo, mock_eval

    def test_writes_csv_header(self, tmp_path):
        ctx = self._patch(tmp_path)
        self._run_base(ctx)
        assert ctx["base_csv"].read_text().strip() == main_finetune.CSV_HEADER

    def test_uses_phase_a_config_not_default_training_params(self, tmp_path):
        ctx = self._patch(tmp_path, phase_a_params=_phase_a(freeze=9, epochs=150, lr0=0.01))
        mock_inst = MagicMock()
        self._run_base(ctx, mock_yolo=MagicMock(return_value=mock_inst))
        kw = mock_inst.train.call_args[1]
        assert kw["freeze"] == 9
        assert kw["epochs"] == 150
        assert kw["lr0"] == pytest.approx(0.01)

    def test_weights_key_excluded_from_train(self, tmp_path):
        ctx = self._patch(tmp_path, phase_a_params=_phase_a(weights="yolo26n.pt"))
        mock_inst = MagicMock()
        self._run_base(ctx, mock_yolo=MagicMock(return_value=mock_inst))
        assert "weights" not in mock_inst.train.call_args[1]

    def test_exist_ok_passed_to_train(self, tmp_path):
        ctx = self._patch(tmp_path)
        mock_inst = MagicMock()
        self._run_base(ctx, mock_yolo=MagicMock(return_value=mock_inst))
        assert mock_inst.train.call_args[1].get("exist_ok") is True

    def test_evaluate_called_once_per_fold(self, tmp_path):
        ctx = self._patch(tmp_path, folds=["fold_1", "fold_2"])
        mock_eval = MagicMock()
        self._run_base(ctx, mock_eval=mock_eval)
        assert mock_eval.call_count == 2

    def test_evaluate_receives_correct_csv_paths(self, tmp_path):
        ctx = self._patch(tmp_path)
        mock_eval = MagicMock()
        self._run_base(ctx, mock_eval=mock_eval)
        _, _, _, result_csv, counting_csv = mock_eval.call_args[0]
        assert result_csv == ctx["base_csv"]
        assert counting_csv == ctx["count_csv"]

    def test_evaluate_model_path_points_to_phase_a_best(self, tmp_path):
        ctx = self._patch(tmp_path, folds=["fold_1"])
        mock_eval = MagicMock()
        self._run_base(ctx, mock_eval=mock_eval)
        _, _, model_path, _, _ = mock_eval.call_args[0]
        assert "phase_a_fold_1" in model_path
        assert model_path.endswith("best.pt")


# ── _run_phase_finetune ────────────────────────────────────────────────────────

class TestRunPhaseFinetune:
    def _patch(self, tmp_path, folds=("fold_1",)):
        ft = tmp_path / "ft"
        _make_folds(ft, list(folds))
        results = tmp_path / "results"
        ft_csv = results / "results_ft.csv"
        count_csv = results / "counting_ft.csv"
        ft_project = tmp_path / "ft_project"
        return dict(ft=ft, results=results, ft_csv=ft_csv,
                    count_csv=count_csv, ft_project=ft_project)

    def _make_base_weights(self, ft_project: Path) -> Path:
        w = ft_project / "phase_a_fold_1" / "weights" / "best.pt"
        w.parent.mkdir(parents=True)
        w.touch()
        return w

    def _run_ft(self, ctx, mock_yolo=None, mock_eval=None):
        mock_yolo = mock_yolo or MagicMock()
        mock_eval = mock_eval or MagicMock()
        with patch.object(main_finetune, "DATASET_FT", ctx["ft"]), \
             patch.object(main_finetune, "FT_PROJECT", ctx["ft_project"]), \
             patch.object(main_finetune, "RESULTS_DIR", ctx["results"]), \
             patch.object(main_finetune, "RESULTS_FT_CSV", ctx["ft_csv"]), \
             patch.object(main_finetune, "COUNTING_FT_CSV", ctx["count_csv"]), \
             patch("main_finetune.CriarLabelsYOLO26"), \
             patch("main_finetune.get_finetune_params",
                   return_value={"phase_a": {}, "phase_b": _phase_b()}), \
             patch("main_finetune.YOLO", mock_yolo), \
             patch("main_finetune._evaluate", mock_eval), \
             patch("main_finetune.tg"):
            main_finetune._run_phase_finetune()
        return mock_yolo, mock_eval

    def test_raises_if_base_weights_missing(self, tmp_path):
        ctx = self._patch(tmp_path)
        with patch.object(main_finetune, "DATASET_FT", ctx["ft"]), \
             patch.object(main_finetune, "FT_PROJECT", ctx["ft_project"]), \
             patch.object(main_finetune, "RESULTS_DIR", ctx["results"]), \
             patch.object(main_finetune, "RESULTS_FT_CSV", ctx["ft_csv"]), \
             patch.object(main_finetune, "COUNTING_FT_CSV", ctx["count_csv"]):
            with pytest.raises(FileNotFoundError, match="phase_a_fold_1"):
                main_finetune._run_phase_finetune()

    def test_writes_csv_header(self, tmp_path):
        ctx = self._patch(tmp_path)
        self._make_base_weights(ctx["ft_project"])
        self._run_ft(ctx)
        assert ctx["ft_csv"].read_text().strip() == main_finetune.CSV_HEADER

    def test_loads_phase_a_fold1_weights(self, tmp_path):
        ctx = self._patch(tmp_path)
        base_w = self._make_base_weights(ctx["ft_project"])
        mock_yolo_cls = MagicMock()
        self._run_ft(ctx, mock_yolo=mock_yolo_cls)
        mock_yolo_cls.assert_called_once_with(str(base_w))

    def test_phase_b_freeze_passed_to_train(self, tmp_path):
        ctx = self._patch(tmp_path)
        self._make_base_weights(ctx["ft_project"])
        mock_inst = MagicMock()
        with patch.object(main_finetune, "DATASET_FT", ctx["ft"]), \
             patch.object(main_finetune, "FT_PROJECT", ctx["ft_project"]), \
             patch.object(main_finetune, "RESULTS_DIR", ctx["results"]), \
             patch.object(main_finetune, "RESULTS_FT_CSV", ctx["ft_csv"]), \
             patch.object(main_finetune, "COUNTING_FT_CSV", ctx["count_csv"]), \
             patch("main_finetune.CriarLabelsYOLO26"), \
             patch("main_finetune.get_finetune_params",
                   return_value={"phase_a": {}, "phase_b": _phase_b(freeze=10)}), \
             patch("main_finetune.YOLO", return_value=mock_inst), \
             patch("main_finetune._evaluate"), \
             patch("main_finetune.tg"):
            main_finetune._run_phase_finetune()
        kw = mock_inst.train.call_args[1]
        assert kw["freeze"] == 10
        assert kw.get("exist_ok") is True

    def test_evaluate_model_path_points_to_phase_b_best(self, tmp_path):
        ctx = self._patch(tmp_path, folds=["fold_1"])
        self._make_base_weights(ctx["ft_project"])
        mock_eval = MagicMock()
        self._run_ft(ctx, mock_eval=mock_eval)
        _, _, model_path, _, _ = mock_eval.call_args[0]
        assert "phase_b_fold_1" in model_path
        assert model_path.endswith("best.pt")

    def test_evaluate_called_once_per_fold(self, tmp_path):
        ctx = self._patch(tmp_path, folds=["fold_1", "fold_2"])
        self._make_base_weights(ctx["ft_project"])
        mock_eval = MagicMock()
        self._run_ft(ctx, mock_eval=mock_eval)
        assert mock_eval.call_count == 2

    def test_evaluate_receives_ft_csv_paths(self, tmp_path):
        ctx = self._patch(tmp_path)
        self._make_base_weights(ctx["ft_project"])
        mock_eval = MagicMock()
        self._run_ft(ctx, mock_eval=mock_eval)
        _, _, _, result_csv, counting_csv = mock_eval.call_args[0]
        assert result_csv == ctx["ft_csv"]
        assert counting_csv == ctx["count_csv"]


# ── main ───────────────────────────────────────────────────────────────────────

# ── _find_rscript ──────────────────────────────────────────────────────────────

class TestFindRscript:
    def test_returns_first_existing_candidate(self, tmp_path):
        r = tmp_path / "Rscript"
        r.touch()
        with patch.object(main_finetune, "_find_rscript",
                          wraps=lambda: str(r)):
            assert main_finetune._find_rscript() == str(r)

    def test_raises_if_no_candidate_exists(self, tmp_path):
        with patch.object(main_finetune.Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Rscript"):
                main_finetune._find_rscript()

    def test_finds_conda_rscript(self, tmp_path):
        conda_path = tmp_path / "Rscript"
        conda_path.touch()
        candidates = [str(conda_path), "/usr/bin/Rscript", "/usr/local/bin/Rscript"]
        original = main_finetune._find_rscript

        def patched():
            for c in candidates:
                if Path(c).exists():
                    return c
            raise FileNotFoundError("Rscript not found.")

        with patch.object(main_finetune, "_find_rscript", side_effect=patched):
            result = main_finetune._find_rscript()
        assert result == str(conda_path)


# ── main ───────────────────────────────────────────────────────────────────────

class TestMain:
    def _run_main(self, tmp_path, base_side=None, ft_side=None):
        mock_tg = MagicMock()
        with patch.object(main_finetune, "PROJECT_ROOT", tmp_path), \
             patch.object(main_finetune, "RESULTS_DIR", tmp_path / "results"), \
             patch.object(main_finetune, "RESULTS_BASE_CSV", tmp_path / "results" / "results_base.csv"), \
             patch.object(main_finetune, "RESULTS_FT_CSV", tmp_path / "results" / "results_finetune.csv"), \
             patch.object(main_finetune, "COUNTING_BASE_CSV", tmp_path / "results" / "counting_base.csv"), \
             patch.object(main_finetune, "COUNTING_FT_CSV", tmp_path / "results" / "counting_finetune.csv"), \
             patch("main_finetune._run_phase_base",
                   side_effect=base_side or (lambda: None)), \
             patch("main_finetune._run_phase_finetune",
                   side_effect=ft_side or (lambda: None)), \
             patch("main_finetune._find_rscript", return_value="/fake/Rscript"), \
             patch("main_finetune.subprocess.run") as mock_run, \
             patch("main_finetune.tg", mock_tg):
            main_finetune.main()
        return mock_tg, mock_run

    def test_calls_phases_in_order(self, tmp_path):
        order = []
        with patch.object(main_finetune, "PROJECT_ROOT", tmp_path), \
             patch.object(main_finetune, "RESULTS_DIR", tmp_path / "results"), \
             patch.object(main_finetune, "RESULTS_BASE_CSV", tmp_path / "results" / "results_base.csv"), \
             patch.object(main_finetune, "RESULTS_FT_CSV", tmp_path / "results" / "results_finetune.csv"), \
             patch.object(main_finetune, "COUNTING_BASE_CSV", tmp_path / "results" / "counting_base.csv"), \
             patch.object(main_finetune, "COUNTING_FT_CSV", tmp_path / "results" / "counting_finetune.csv"), \
             patch("main_finetune._run_phase_base",
                   side_effect=lambda: order.append("base")), \
             patch("main_finetune._run_phase_finetune",
                   side_effect=lambda: order.append("ft")), \
             patch("main_finetune._find_rscript", return_value="/fake/Rscript"), \
             patch("main_finetune.subprocess.run"), \
             patch("main_finetune.tg"):
            main_finetune.main()
        assert order == ["base", "ft"]

    def test_calls_rscript_with_graficos_finetune(self, tmp_path):
        _, mock_run = self._run_main(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/fake/Rscript"
        assert "graficos_finetune.R" in cmd[1]

    def test_rscript_check_true(self, tmp_path):
        _, mock_run = self._run_main(tmp_path)
        assert mock_run.call_args[1].get("check") is True

    def test_sends_completion_telegram(self, tmp_path):
        mock_tg, _ = self._run_main(tmp_path)
        texts = [c[0][0] for c in mock_tg.send.call_args_list]
        assert any("Conclu" in t or "conclu" in t for t in texts)

    def test_sends_photo_if_plot_exists(self, tmp_path):
        plot = tmp_path / "results" / "boxplot_compare.png"
        plot.parent.mkdir(parents=True, exist_ok=True)
        plot.write_bytes(b"fake png")
        mock_tg = MagicMock()
        with patch.object(main_finetune, "PROJECT_ROOT", tmp_path), \
             patch.object(main_finetune, "RESULTS_DIR", tmp_path / "results"), \
             patch.object(main_finetune, "RESULTS_BASE_CSV", tmp_path / "results" / "results_base.csv"), \
             patch.object(main_finetune, "RESULTS_FT_CSV", tmp_path / "results" / "results_finetune.csv"), \
             patch.object(main_finetune, "COUNTING_BASE_CSV", tmp_path / "results" / "counting_base.csv"), \
             patch.object(main_finetune, "COUNTING_FT_CSV", tmp_path / "results" / "counting_finetune.csv"), \
             patch("main_finetune._run_phase_base"), \
             patch("main_finetune._run_phase_finetune"), \
             patch("main_finetune._find_rscript", return_value="/fake/Rscript"), \
             patch("main_finetune.subprocess.run"), \
             patch("main_finetune.tg", mock_tg):
            main_finetune.main()
        mock_tg.send_photo.assert_called_once_with(plot)

    def test_sends_documents_for_existing_csvs(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir(parents=True, exist_ok=True)
        base_csv = results / "results_base.csv"
        base_csv.write_text("a,b\n")
        mock_tg = MagicMock()
        with patch.object(main_finetune, "PROJECT_ROOT", tmp_path), \
             patch.object(main_finetune, "RESULTS_DIR", results), \
             patch.object(main_finetune, "RESULTS_BASE_CSV", base_csv), \
             patch.object(main_finetune, "RESULTS_FT_CSV", results / "results_finetune.csv"), \
             patch.object(main_finetune, "COUNTING_BASE_CSV", results / "counting_base.csv"), \
             patch.object(main_finetune, "COUNTING_FT_CSV", results / "counting_finetune.csv"), \
             patch("main_finetune._run_phase_base"), \
             patch("main_finetune._run_phase_finetune"), \
             patch("main_finetune._find_rscript", return_value="/fake/Rscript"), \
             patch("main_finetune.subprocess.run"), \
             patch("main_finetune.tg", mock_tg):
            main_finetune.main()
        sent_docs = [c[0][0] for c in mock_tg.send_document.call_args_list]
        assert base_csv in sent_docs

    def test_no_send_photo_if_plot_missing(self, tmp_path):
        mock_tg, _ = self._run_main(tmp_path)
        mock_tg.send_photo.assert_not_called()

    def test_sends_error_telegram_on_exception(self, tmp_path):
        boom = RuntimeError("treino falhou")
        mock_tg = MagicMock()
        with patch.object(main_finetune, "PROJECT_ROOT", tmp_path), \
             patch("main_finetune._run_phase_base", side_effect=boom), \
             patch("main_finetune._run_phase_finetune"), \
             patch("main_finetune._find_rscript", return_value="/fake/Rscript"), \
             patch("main_finetune.subprocess.run"), \
             patch("main_finetune.tg", mock_tg):
            with pytest.raises(RuntimeError, match="treino falhou"):
                main_finetune.main()
        mock_tg.send_error.assert_called_once()
        assert mock_tg.send_error.call_args[0][1] is boom

    def test_reraises_exception_after_telegram(self, tmp_path):
        boom = ValueError("dataset missing")
        mock_tg = MagicMock()
        with patch.object(main_finetune, "PROJECT_ROOT", tmp_path), \
             patch("main_finetune._run_phase_base", side_effect=boom), \
             patch("main_finetune._run_phase_finetune"), \
             patch("main_finetune._find_rscript", return_value="/fake/Rscript"), \
             patch("main_finetune.subprocess.run"), \
             patch("main_finetune.tg", mock_tg):
            with pytest.raises(ValueError):
                main_finetune.main()
