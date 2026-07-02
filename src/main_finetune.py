from __future__ import annotations

import csv
import contextlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

os.environ.setdefault("WANDB_DISABLED", "true")

from ultralytics import YOLO
import ResultsDetections as RD
from ResultsDetections import generate_results, print_to_file
import telegram_notify as tg
from Detectors.YOLO26.GeraLabels import CriarLabelsYOLO26
from Detectors.YOLO26.config import get_finetune_params, DEFAULT_WEIGHTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent

DATASET_BASE = PROJECT_ROOT / "dataset" / "all"
DATASET_FT = PROJECT_ROOT / "dataset" / "fine_tuning"
RESULTS_DIR = PROJECT_ROOT / "results"
FT_PROJECT = SRC_ROOT / "runs" / "detect" / "YOLO26_finetune"

RESULTS_BASE_CSV = RESULTS_DIR / "results_base.csv"
RESULTS_FT_CSV = RESULTS_DIR / "results_finetune.csv"
COUNTING_BASE_CSV = RESULTS_DIR / "counting_base.csv"
COUNTING_FT_CSV = RESULTS_DIR / "counting_finetune.csv"
OUTPUT_LOG = RESULTS_DIR / "main_finetune_output.log"

CSV_HEADER = "ml,fold,backbone,loss_function,mAP,mAP50,mAP75,MAE,RMSE,r,precision,recall,fscore"
COUNTING_HEADER = "ml,fold,groundtruth,predicted,TP,FP,dif,fileName"


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.streams[0], "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.streams[0].fileno()


@contextlib.contextmanager
def _tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _format_float(value) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _loss_text(trainer) -> str:
    tloss = getattr(trainer, "tloss", None)
    names = getattr(trainer, "loss_names", None) or ("box", "cls", "dfl")

    if tloss is None:
        return ""
    try:
        values = tloss.detach().cpu().tolist()
    except AttributeError:
        values = tloss
    if not isinstance(values, (list, tuple)):
        values = [values]

    parts = []
    for name, value in zip(names, values):
        clean_name = str(name).replace("_loss", "")
        parts.append(f"{clean_name}={_format_float(value)}")
    return " | ".join(parts)


def _metrics_text(trainer) -> str:
    metrics = getattr(trainer, "metrics", None) or {}
    keys = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    labels = {
        "metrics/precision(B)": "P",
        "metrics/recall(B)": "R",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
    }
    parts = [
        f"{labels[key]}={_format_float(metrics[key])}"
        for key in keys
        if key in metrics
    ]
    return " | ".join(parts)


def _add_telegram_epoch_callback(model: YOLO, phase: str, fold: str, total_epochs: int) -> None:
    def _send_epoch_update(trainer) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        if epoch <= 0 or epoch % 10 != 0:
            return

        parts = [f"[{phase}] {fold} epoca {epoch}/{total_epochs}"]
        loss = _loss_text(trainer)
        metrics = _metrics_text(trainer)
        if loss:
            parts.append(loss)
        if metrics:
            parts.append(metrics)
        tg.send(" | ".join(parts))

    model.add_callback("on_fit_epoch_end", _send_epoch_update)


def _collect_folds(dataset_root: Path) -> list[str]:
    files_json_dir = dataset_root / "filesJSON"
    names = {
        "_".join(p.stem.split("_")[:2])
        for p in files_json_dir.glob("fold_*_*.json")
        if p.is_file()
    }
    if not names:
        raise FileNotFoundError(f"No fold files found in {files_json_dir}")
    return sorted(names, key=lambda n: int(n.split("_")[1]))


def _evaluate(
    root: str,
    fold: str,
    model_path: str,
    result_csv: Path,
    counting_csv: Path,
) -> None:
    RD.COUNTING_CSV_PATH = counting_csv
    mAP, mAP50, mAP75, MAE, RMSE, precision, recall, fscore, r = generate_results(
        root, fold, model_path, "YOLO26", False, tiling_mode="basic"
    )
    with result_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "YOLO26", fold, str(DEFAULT_WEIGHTS), "ultralytics-default",
            mAP, mAP50, mAP75, MAE, RMSE, r, precision, recall, fscore,
        ])


def _run_phase_base() -> None:
    folds = _collect_folds(DATASET_BASE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_to_file(CSV_HEADER, RESULTS_BASE_CSV, "w")
    print_to_file(COUNTING_HEADER, COUNTING_BASE_CSV, "w")

    tg.send(f"[base] Iniciando phase_a: {len(folds)} dobras — {DATASET_BASE.name}")

    for fold in folds:
        tg.send(f"[base] Iniciando fold {fold}")
        data_yaml = CriarLabelsYOLO26(fold, DATASET_BASE)
        phase_a = get_finetune_params(data_yaml, DEFAULT_WEIGHTS, fold=fold)["phase_a"]

        model = YOLO(DEFAULT_WEIGHTS)
        print(model.model.loss)
        model.info(verbose=True)
        for name, param in model.model.named_parameters():
            if "dfl" in name.lower():
                print(name, param.shape)
        _add_telegram_epoch_callback(model, "base", fold, phase_a["epochs"])
        model.train(**{k: v for k, v in phase_a.items() if k != "weights"}, exist_ok=True)

        model_path = FT_PROJECT / f"phase_a_{fold}" / "weights" / "best.pt"
        _evaluate(str(DATASET_BASE), fold, str(model_path), RESULTS_BASE_CSV, COUNTING_BASE_CSV)
        tg.send(f"[base] fold {fold} concluído")


def _run_phase_finetune() -> None:
    ft_folds = _collect_folds(DATASET_FT)
    base_weights = FT_PROJECT / "phase_a_fold_1" / "weights" / "best.pt"

    if not base_weights.exists():
        raise FileNotFoundError(f"Pesos phase_a fold_1 não encontrados: {base_weights}")

    print_to_file(CSV_HEADER, RESULTS_FT_CSV, "w")
    print_to_file(COUNTING_HEADER, COUNTING_FT_CSV, "w")

    tg.send(f"[finetune] Iniciando phase_b: {len(ft_folds)} dobras — {DATASET_FT.name}")

    for fold in ft_folds:
        tg.send(f"[finetune] Iniciando fold {fold}")
        data_yaml = CriarLabelsYOLO26(fold, DATASET_FT)
        phase_b = get_finetune_params(data_yaml, base_weights, fold=fold)["phase_b"]

        model = YOLO(str(base_weights))
        print(model.model.loss)
        model.info(verbose=True)
        for name, param in model.model.named_parameters():
            if "dfl" in name.lower():
                print(name, param.shape)
        _add_telegram_epoch_callback(model, "finetune", fold, phase_b["epochs"])
        model.train(**phase_b, exist_ok=True)

        ft_model_path = FT_PROJECT / f"phase_b_{fold}" / "weights" / "best.pt"
        _evaluate(str(DATASET_FT), fold, str(ft_model_path), RESULTS_FT_CSV, COUNTING_FT_CSV)
        tg.send(f"[finetune] fold {fold} concluído")


def _find_rscript() -> str:
    candidates = [
        "/home/neto/miniconda3/envs/detectores/bin/Rscript",
        "/usr/bin/Rscript",
        "/usr/local/bin/Rscript",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Rscript not found. Install R or add it to PATH.")


def _run_pipeline() -> None:
    _run_phase_base()
    _run_phase_finetune()

    rscript = _find_rscript()
    r_script = PROJECT_ROOT / "utils" / "graficos_finetune.R"
    subprocess.run([rscript, str(r_script)], check=True, cwd=str(PROJECT_ROOT / "utils"))

    plot = RESULTS_DIR / "boxplot_compare.png"
    if plot.exists():
        tg.send_photo(plot)
    for csv_path in (RESULTS_BASE_CSV, RESULTS_FT_CSV, COUNTING_BASE_CSV, COUNTING_FT_CSV):
        if csv_path.exists():
            tg.send_document(csv_path)


def main() -> None:
    try:
        with _tee_output(OUTPUT_LOG):
            _run_pipeline()
        if OUTPUT_LOG.exists():
            tg.send_document(OUTPUT_LOG)
        tg.send("[main_finetune] Concluído. Gráficos, CSVs e output enviados.")
    except Exception as exc:
        tg.send_error("main_finetune", exc)
        if OUTPUT_LOG.exists():
            tg.send_document(OUTPUT_LOG)
        raise


if __name__ == "__main__":
    main()
