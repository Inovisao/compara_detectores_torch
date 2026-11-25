"""Destinos de resultados compartilhados pelo treino e pela avaliação."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = Path(os.environ["DATASET_ROOT"]).resolve().name if os.getenv("DATASET_ROOT") else os.getenv("DATASET_NAME", "second")
if not DATASET_NAME or DATASET_NAME in {".", ".."} or Path(DATASET_NAME).name != DATASET_NAME:
    raise ValueError("DATASET_NAME deve ser um nome de pasta, como first ou second.")

RESULTS_DIR = PROJECT_ROOT / "results" / DATASET_NAME
RESULTS_PATH = RESULTS_DIR / "prediction"
RESULTS_CSV_PATH = RESULTS_DIR / "results.csv"
COUNTING_CSV_PATH = RESULTS_DIR / "counting.csv"
RESULTS_BY_CLASS_CSV_PATH = RESULTS_DIR / "resultsbyclass.csv"
