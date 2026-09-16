import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    return {"macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4)}


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    r = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels,
                              output_dict=True, zero_division=0)
    return pd.DataFrame(r).T.round(4)


def plot_confusion(y_true, y_pred, labels, out_path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))), normalize="true")
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > .55 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def rebuild_metrics_table(results_dir) -> pd.DataFrame:
    """Collect results/{task}/{run}/metrics.json -> results/metrics.csv (one row per run)."""
    results_dir = Path(results_dir)
    rows = []
    for f in sorted(results_dir.glob("*/*/metrics.json")):
        m = json.load(open(f))
        rows.append({"task": f.parent.parent.name, "run": f.parent.name,
                     **{k: m.get(k) for k in ("macro_f1", "accuracy", "n_eval",
                                              "model", "loss", "best_epoch", "has_test")}})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["task", "macro_f1"], ascending=[True, False])
    df.to_csv(results_dir / "metrics.csv", index=False)
    return df
