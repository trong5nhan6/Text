import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


def compute_metrics(y_true, y_pred) -> dict:
    return {"macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4)}


def full_metrics(y_true, labels, proba=None, pred=None) -> dict:
    """Every metric printed per epoch with data.split=official. Give `proba` (n x k) or only
    `pred` -- the running train scores have no probabilities, so their `ce` is None.

      macro_f1      ranking metric of the shared task: every class counts the same
      accuracy      reported next to it by the organisers
      macro_p/r     macro precision / recall: which side of macro-F1 is short
      weighted_f1   F1 weighted by class support (task b: dominated by Gender)
      ce            mean -log p(gold) -- plain cross-entropy, NOT the training loss (focal/wce)
      f1_<label>    per-class F1 (task b: the rare Geo-political / Violence decide macro-F1)
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(proba).argmax(1) if pred is None else np.asarray(pred)
    ids = list(range(len(labels)))
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=ids, zero_division=0)
    m = {"n": int(len(y_true)), "macro_f1": float(f.mean()), "accuracy": float((pred == y_true).mean()),
         "macro_p": float(p.mean()), "macro_r": float(r.mean()),
         "weighted_f1": float(f1_score(y_true, pred, labels=ids, average="weighted", zero_division=0)),
         "ce": None if proba is None else
         float(-np.log(np.clip(np.asarray(proba)[np.arange(len(y_true)), y_true], 1e-12, 1)).mean())}
    m.update({f"f1_{l}": float(x) for l, x in zip(labels, f)})
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def metrics_table(rows: dict, labels) -> str:
    """{split: full_metrics(...)} -> an aligned text table, one line per split."""
    short = [l[:9] for l in labels]
    cols = ["macro_f1", "accuracy", "macro_p", "macro_r", "weighted_f1", "ce"]
    head = (f"  {'split':6s} {'n':>5s} " + " ".join(f"{c:>11s}" for c in cols)
            + " | F1 " + " ".join(f"{s:>9s}" for s in short))
    lines = [head]
    for split, m in rows.items():
        cell = lambda v: f"{'-':>11s}" if v is None else f"{v:11.4f}"
        lines.append(f"  {split:6s} {m['n']:5d} " + " ".join(cell(m[c]) for c in cols)
                     + " |    " + " ".join(f"{m[f'f1_{l}']:9.4f}" for l in labels))
    return "\n".join(lines)


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
                     **{k: m.get(k) for k in ("macro_f1", "accuracy", "test_macro_f1",
                                              "test_accuracy", "n_eval", "model", "loss",
                                              "best_epoch", "has_test")}})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["task", "macro_f1"], ascending=[True, False])
    df.to_csv(results_dir / "metrics.csv", index=False)
    return df
