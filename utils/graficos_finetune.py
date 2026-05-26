from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
METRICS = ["mAP", "mAP50", "mAP75", "precision", "recall", "fscore", "MAE", "RMSE", "r"]


def generate(
    base_csv: Path = RESULTS_DIR / "results_base.csv",
    ft_csv: Path = RESULTS_DIR / "results_finetune.csv",
    output: Path = RESULTS_DIR / "boxplot_compare.png",
) -> Path:
    base = pd.read_csv(base_csv)
    ft = pd.read_csv(ft_csv)
    base["config"] = "Base"
    ft["config"] = "Finetune"
    df = pd.concat([base, ft], ignore_index=True)
    df["config"] = pd.Categorical(df["config"], categories=["Base", "Finetune"], ordered=True)

    available = [m for m in METRICS if m in df.columns]
    ncols = 3
    nrows = -(-len(available) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    palette = {"Base": "#4C72B0", "Finetune": "#DD8452"}

    for i, metric in enumerate(available):
        sns.boxplot(data=df, x="config", y=metric, hue="config",
                    palette=palette, ax=axes[i], legend=False)
        axes[i].set_title(metric)
        axes[i].set_xlabel("")

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Base vs Finetune", fontsize=14, y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    path = generate()
    print(f"Salvo em: {path}")
