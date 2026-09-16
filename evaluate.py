#!/usr/bin/env python3
"""
Compare runs on their out-of-fold (OOF) predictions and evaluate a blend.

  python evaluate.py --task b                                   # all runs of task b
  python evaluate.py --task b --runs tfidf_lr muril_wce_s42     # selected runs + their average
  python evaluate.py --task b --runs tfidf_lr muril_wce_s42 --weights 0.3 0.7
  python evaluate.py --task b --runs tfidf_lr muril_wce_s42 --optimize   # search blend weights on OOF

Writes per-class reports + confusion matrices to results/{task}/{run}/ (and results/{task}/_blend/),
and refreshes results/metrics.csv.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import label_names, load_split
from src.evaluation.metrics import compute_metrics, per_class_report, plot_confusion, rebuild_metrics_table
from src.utils.config import load_config


def blend(probs, weights):
    w = np.asarray(weights, float); w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, probs))


def optimize_weights(probs, y, step=0.1):
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    best = (None, -1)
    for w in itertools.product(grid, repeat=len(probs)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        f1 = compute_metrics(y, blend(probs, w).argmax(1))["macro_f1"]
        if f1 > best[1]:
            best = (list(w), f1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--optimize", action="store_true", help="grid-search blend weights (step 0.1) on OOF")
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()

    cfg = load_config(a.config, task=a.task)
    res = Path(cfg["paths"]["results_dir"]) / a.task
    train = load_split(cfg, "train")
    labels = label_names(a.task)
    runs = a.runs or sorted(p.name for p in res.iterdir() if (p / "oof.npy").exists())
    if not runs:
        raise SystemExit(f"no finished runs in {res}")

    probs, rows = [], []
    for r in runs:
        p = np.load(res / r / "oof.npy"); probs.append(p)
        pred = p.argmax(1)
        m = compute_metrics(train.y, pred)
        rep = per_class_report(train.y, pred, labels)
        rep.to_csv(res / r / "per_class.csv")
        plot_confusion(train.y, pred, labels, res / r / "confusion.png", f"{a.task}/{r}  macro-F1 {m['macro_f1']:.4f}")
        rows.append({"run": r, **m, **{f"f1_{l}": rep.loc[l, "f1-score"] for l in labels}})

    if len(runs) > 1:
        if a.optimize:
            w, _ = optimize_weights(probs, train.y)
            print(f"optimized weights (OOF): {dict(zip(runs, w))}")
        else:
            w = a.weights or [1.0] * len(runs)
        pb = blend(probs, w); pred = pb.argmax(1)
        m = compute_metrics(train.y, pred)
        rep = per_class_report(train.y, pred, labels)
        out = res / "_blend"; out.mkdir(exist_ok=True)
        rep.to_csv(out / "per_class.csv")
        plot_confusion(train.y, pred, labels, out / "confusion.png", f"{a.task} blend  macro-F1 {m['macro_f1']:.4f}")
        json.dump({"runs": runs, "weights": [float(x) for x in w], **m}, open(out / "blend.json", "w"), indent=1)
        rows.append({"run": "BLEND(" + ", ".join(f"{r}:{x:.2f}" for r, x in zip(runs, np.asarray(w) / np.sum(w))) + ")",
                     **m, **{f"f1_{l}": rep.loc[l, "f1-score"] for l in labels}})

    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("run").round(4).to_string())
    rebuild_metrics_table(Path(cfg["paths"]["results_dir"]))


if __name__ == "__main__":
    main()
