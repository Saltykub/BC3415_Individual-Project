"""
evaluate.py
===========
Scores every trained model on the *same* held-out test set and writes a single
comparison table plus one confusion-matrix figure per model.

Metric choice
-------------
Accuracy alone is misleading here: predicting "positive" for everything already
scores ~77% on this dataset. Macro F1 averages the F1 of each class equally, so
the rare negative and neutral classes count as much as the common positive one.
Macro F1 is therefore the headline number and the comparison table is sorted
by it; per-class F1 is reported alongside so the trade-offs are visible.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence

import matplotlib

# Agg backend: no GUI needed, so the script runs the same way in a terminal,
# in CI, or inside Streamlit.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import config
import sentiment


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str],
                    model_key: str, display: str) -> Dict[str, object]:
    """Flat metrics dict for one model, ready to become a CSV row."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=config.LABELS)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted",
                           labels=config.LABELS)
    per_class = f1_score(y_true, y_pred, average=None, labels=config.LABELS)

    row: Dict[str, object] = {
        "model_key": model_key,
        "model": display,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
    }
    for label, score in zip(config.LABELS, per_class):
        row[f"f1_{label}"] = round(float(score), 4)
    return row


def plot_confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str],
                          title: str, filename: str,
                          normalise: bool = True):
    """
    Save a confusion matrix PNG.

    Row-normalised by default: with an imbalanced test set, raw counts make the
    positive row dominate visually and hide the interesting errors (negative
    reviews predicted as neutral).
    """
    matrix = confusion_matrix(y_true, y_pred, labels=config.LABELS)
    counts = matrix.copy()
    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalise else None)

    ax.set_xticks(range(len(config.LABELS)), config.LABELS, rotation=20)
    ax.set_yticks(range(len(config.LABELS)), config.LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)

    # Annotate with "share (count)" so both views are available at a glance.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = (f"{value:.0%}\n({counts[i, j]:,})" if normalise
                    else f"{counts[i, j]:,}")
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if value > 0.55 else "black")

    fig.colorbar(image, ax=ax, fraction=0.046, shrink=0.85)
    fig.tight_layout()

    path = config.FIGURE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Whole-suite evaluation
# ---------------------------------------------------------------------------
def evaluate_models(test: pd.DataFrame, model_keys: Sequence[str] | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate every available model on `test` and write all artefacts.

    Outputs:
      artifacts/reports/model_comparison.csv      one row per model
      artifacts/reports/classification_<key>.txt  full sklearn report
      artifacts/figures/confusion_matrix_<key>.png
    """
    config.ensure_dirs()
    on_disk = sentiment.available_models()
    model_keys = list(model_keys) if model_keys else list(on_disk)

    texts = test[config.TEXT_COLUMN].tolist()
    y_true = test[config.LABEL_COLUMN].tolist()

    rows: List[Dict[str, object]] = []
    for key in model_keys:
        if key not in on_disk:
            if verbose:
                print(f"  skipping {key}: not trained yet")
            continue

        display = config.MODEL_REGISTRY[key]["display"]
        if verbose:
            print(f"  scoring {display} on {len(texts):,} test reviews ...")

        predictor = sentiment.get_predictor(key)
        y_pred = predictor.predict(texts)

        row = compute_metrics(y_true, y_pred, key, display)
        rows.append(row)

        report = classification_report(y_true, y_pred, labels=config.LABELS,
                                       digits=4, zero_division=0)
        (config.REPORT_DIR / f"classification_{key}.txt").write_text(
            f"{display}\n{'=' * len(display)}\n\n{report}", encoding="utf-8"
        )
        plot_confusion_matrix(
            y_true, y_pred,
            title=f"{display}\nmacro F1 = {row['macro_f1']:.3f}",
            filename=f"confusion_matrix_{key}.png",
        )

        if verbose:
            print(f"    accuracy={row['accuracy']:.4f}  "
                  f"macro_f1={row['macro_f1']:.4f}  "
                  + "  ".join(f"F1_{lbl}={row[f'f1_{lbl}']:.3f}"
                              for lbl in config.LABELS))

    if not rows:
        raise RuntimeError("No trained models found. Run `python train.py` first.")

    comparison = (pd.DataFrame(rows)
                  .sort_values("macro_f1", ascending=False)
                  .reset_index(drop=True))
    comparison.to_csv(config.COMPARISON_CSV, index=False)

    # A tiny JSON side-file records which model won, so demo.py / app.py can
    # default to the best available model instead of a hard-coded choice.
    (config.REPORT_DIR / "best_model.json").write_text(
        json.dumps({
            "best_model_key": comparison.iloc[0]["model_key"],
            "macro_f1": float(comparison.iloc[0]["macro_f1"]),
            "test_rows": len(test),
        }, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print("\nModel comparison (sorted by macro F1)")
        print(comparison.to_string(index=False))
        print(f"\nSaved -> {config.COMPARISON_CSV}")

    return comparison


def load_comparison() -> pd.DataFrame | None:
    """Read the saved comparison table, or None if training has not run."""
    if config.COMPARISON_CSV.exists():
        return pd.read_csv(config.COMPARISON_CSV)
    return None


def best_model_key(default: str | None = None) -> str:
    """Best model by macro F1 if known, otherwise config.DEFAULT_MODEL."""
    path = config.REPORT_DIR / "best_model.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["best_model_key"]
        except (KeyError, ValueError):
            pass
    return default or config.DEFAULT_MODEL


if __name__ == "__main__":
    import preprocessing

    splits = preprocessing.load_splits()
    if "test" not in splits:
        splits, _ = preprocessing.prepare()
    evaluate_models(splits["test"])
