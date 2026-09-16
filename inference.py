#!/usr/bin/env python3
"""
Create a CodaBench submission (predictions.csv + flat submission.zip).

Mode 1 — from saved probabilities of finished runs (single run or blend):
  python inference.py --task a --runs muril_ce_s42 --split val
  python inference.py --task b --runs tfidf_lr muril_wce_s42 roberta_wce_s42 --weights 1 2 2 --split test

Mode 2 — from checkpoints on any CSV (id + Comment), averaging all folds of each run.
Use this when the test file arrives after training (no retraining needed):
  python inference.py --task a --checkpoints checkpoints/a/muril_ce_s42 --input data/raw/binary_test_inputs.csv
  python inference.py --task b --checkpoints checkpoints/b/muril_wce_s42 checkpoints/b/roberta_wce_s42 \
                      --input data/raw/multiclass_test_inputs.csv --tfidf_runs tfidf_lr

Output: results/submissions/{task}_{split|input}_{tag}/predictions.csv + submission.zip
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import eval_targets, label_names, load_split
from src.data.preprocessing import read_inputs
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config


def predict_checkpoint_dir(run_ckpt: Path, texts, batch_size=64, precision="auto"):
    import torch
    from src.data.dataset import make_loader
    from src.models.factory import load_from_checkpoint
    from src.training.trainer import predict_proba, resolve_precision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = resolve_precision(precision, device)
    folds = sorted(p for p in run_ckpt.glob("fold*") if (p / "model.pt").exists())
    if not folds:
        raise SystemExit(f"no fold checkpoints in {run_ckpt}")
    out = 0
    for fd in folds:
        model, tok, meta = load_from_checkpoint(fd)
        cfg = {"data": {"max_len": meta.get("max_len", 96)},
               "training": {"batch_size": batch_size, "eval_batch_size": batch_size, "num_workers": 2}}
        dl = make_loader(list(texts), None, tok, cfg, train=False)
        out = out + predict_proba(model.to(device), dl, device, amp) / len(folds)
        print(f"  {fd}: done (fold macro-F1 {meta.get('macro_f1')})")
        del model
    return out


def write_submission(ids, probs, labels, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": ids, "label": np.array(labels)[probs.argmax(1)]})
    sub.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    np.save(out_dir / "probs.npy", probs)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "predictions.csv", arcname="predictions.csv")
    print(sub.label.value_counts().to_string())
    print(f"==> {out_dir / 'submission.zip'}  ({len(sub)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="finished runs in results/{task}/ (mode 1)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="checkpoints/{task}/{run} folders (mode 2)")
    ap.add_argument("--input", help="CSV to predict in mode 2")
    ap.add_argument("--tfidf_runs", nargs="*", default=[],
                    help="mode 2: add TF-IDF runs (their {split}.npy must match --input; use --split)")
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--tag")
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()

    cfg = load_config(a.config, task=a.task)
    labels = label_names(a.task)
    res = Path(cfg["paths"]["results_dir"])
    probs, names = [], []

    if a.checkpoints:                                            # ---- mode 2
        if not a.input:
            raise SystemExit("--input is required with --checkpoints")
        inp = read_inputs(a.input)
        for c in a.checkpoints:
            print(f"predicting with {c}")
            probs.append(predict_checkpoint_dir(Path(c), inp.text)); names.append(Path(c).name)
        for r in a.tfidf_runs:
            p = np.load(res / a.task / r / f"{a.split}.npy")
            assert len(p) == len(inp), f"{r}/{a.split}.npy has {len(p)} rows, input has {len(inp)}"
            probs.append(p); names.append(r)
        split_name = Path(a.input).stem
    else:                                                        # ---- mode 1
        if not a.runs:
            raise SystemExit("give --runs (mode 1) or --checkpoints + --input (mode 2)")
        inp = load_split(cfg, a.split)
        if inp is None:
            raise SystemExit(f"data/processed/{a.task}_{a.split}.csv not found — run src.data.preprocessing")
        train = load_split(cfg, "train")
        mask, y_eval = eval_targets(train)
        for r in a.runs:
            f = res / a.task / r / f"{a.split}.npy"
            if not f.exists():
                raise SystemExit(f"{f} missing (run trained before the {a.split} file existed?) -> use mode 2")
            probs.append(np.load(f)); names.append(r)
            oof = np.load(res / a.task / r / "oof.npy")
            print(f"  {r:35s} OOF {compute_metrics(y_eval, oof[mask].argmax(1))}")
        split_name = a.split

    w = np.asarray(a.weights or [1.0] * len(probs), float); w /= w.sum()
    if not a.checkpoints and len(a.runs) > 1:
        oof = sum(wi * np.load(res / a.task / r / "oof.npy") for wi, r in zip(w, a.runs))
        print(f"  {'BLEND':35s} OOF {compute_metrics(y_eval, oof[mask].argmax(1))}")
    p = sum(wi * pi for wi, pi in zip(w, probs))
    assert len(p) == len(inp)
    tag = a.tag or "+".join(names)[:80]
    write_submission(inp.id.values, p, labels, res / "submissions" / f"{a.task}_{split_name}_{tag}")


if __name__ == "__main__":
    main()
