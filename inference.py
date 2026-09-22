#!/usr/bin/env python3
"""
Create a CodaBench submission (predictions.csv + flat submission.zip).

Mode 1 — from saved probabilities of finished runs (single run or blend):
  python inference.py --task a --runs muril_ce_s42 --split val
  python inference.py --task b --runs tfidf_lr muril_wce_s42 roberta_wce_s42 --weights 1 2 2 --split test

Mode 2 — from checkpoints on any CSV (id + Comment).
Use this when the test file arrives after training (no retraining needed):
  python inference.py --task a --checkpoints checkpoints/a/muril_ce_s42 --input data/raw/binary_test_inputs.csv
  python inference.py --task b --checkpoints checkpoints/b/muril_wce_s42 checkpoints/b/roberta_wce_s42 \
                      --input data/raw/multiclass_test_inputs.csv --tfidf_runs tfidf_lr

Output: results/submissions/{task}_{split|input}_{tag}/predictions.csv + submission.zip
"""
import argparse
from pathlib import Path

import numpy as np

from src.data.dataset import eval_y, label_names, load_split
from src.data.preprocessing import read_inputs
from src.evaluation.metrics import compute_metrics
from src.evaluation.submission import write_submission
from src.utils.config import load_config


def predict_checkpoint_dir(run_ckpt: Path, texts, batch_size=64, precision="auto"):
    import torch
    from src.data.dataset import make_loader
    from src.models.factory import load_from_checkpoint
    from src.training.trainer import predict_proba, resolve_precision

    if not (run_ckpt / "model.pt").exists():
        raise SystemExit(f"no checkpoint in {run_ckpt} (expected model.pt)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, meta = load_from_checkpoint(run_ckpt)
    cfg = {"data": {"max_len": meta.get("max_len", 96)},
           "training": {"batch_size": batch_size, "eval_batch_size": batch_size, "num_workers": 2}}
    dl = make_loader(list(texts), None, tok, cfg, train=False)
    out = predict_proba(model.to(device), dl, device, resolve_precision(precision, device))
    print(f"  {run_ckpt}: held-out macro-F1 {meta.get('macro_f1')} @ epoch {meta.get('epoch')}")
    return out


def hard_vote(probs, weights):
    """Majority vote over each model's predicted label -> a one-hot-ish matrix of vote shares.

    Soft voting averages probabilities and is usually the better choice, because it keeps how
    sure each model was. It assumes the numbers are comparable across models, and in this repo
    they are not always: LinearSVC and RidgeClassifier have no predict_proba, so tfidf.py feeds
    their decision function through softmax(d * 2.0) -- monotone, so their own argmax is right,
    but on an arbitrary scale next to a real probability. Averaging those together lets whichever
    model happens to produce the largest numbers dominate. Hard voting throws the magnitudes away
    and keeps only the decision, which is exactly the right trade when the magnitudes are junk.

    Ties are broken by the weighted probability sum, scaled small enough that it can only ever
    separate equal vote counts, never outrank a model that won outright.
    """
    votes = np.zeros_like(probs[0], dtype=float)
    for wi, pi in zip(weights, probs):
        votes[np.arange(len(pi)), pi.argmax(1)] += wi
    soft = sum(wi * pi for wi, pi in zip(weights, probs))
    return votes + 1e-6 * soft


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
    ap.add_argument("--vote", choices=["soft", "hard"], default="soft",
                    help="soft: weighted average of probabilities (default). "
                         "hard: majority vote over each model's predicted label.")
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
        y = eval_y(load_split(cfg, "train"))
        for r in a.runs:
            f = res / a.task / r / f"{a.split}.npy"
            if not f.exists():
                raise SystemExit(f"{f} missing (run trained before the {a.split} file existed?) -> use mode 2")
            probs.append(np.load(f)); names.append(r)
            # A run trained with data.use_valdataset=false has no eval.npy, by design: it kept no
            # labelled rows to be scored on. It can still be blended here -- its val/test
            # probabilities are ordinary -- but the weights must come from somewhere else, and
            # there is no held-out number to print for it.
            ef = res / a.task / r / "eval.npy"
            if ef.exists():
                print(f"  {r:35s} held-out {compute_metrics(y, np.load(ef).argmax(1))}")
            else:
                print(f"  {r:35s} khong co eval.npy (use_valdataset=false) -> khong cham duoc")
        split_name = a.split

    w = np.asarray(a.weights or [1.0] * len(probs), float); w /= w.sum()
    if not a.checkpoints and len(a.runs) > 1:
        evs = [res / a.task / r / "eval.npy" for r in a.runs]
        if all(f.exists() for f in evs):
            e = sum(wi * np.load(f) for wi, f in zip(w, evs))
            print(f"  {'BLEND':35s} held-out {compute_metrics(y, e.argmax(1))}")
        else:
            n = sum(1 for f in evs if not f.exists())
            print(f"  {'BLEND':35s} khong cham duoc ({n}/{len(evs)} run khong co eval.npy). "
                  f"Trong so phai lay tu ban 90% tuong ung.")
    p = hard_vote(probs, w) if a.vote == "hard" else sum(wi * pi for wi, pi in zip(w, probs))
    if a.vote == "hard":
        soft = sum(wi * pi for wi, pi in zip(w, probs))
        n_diff = int((p.argmax(1) != soft.argmax(1)).sum())
        print(f"  vote=hard: {n_diff}/{len(p)} dong khac voi soft vote")
    assert len(p) == len(inp)
    tag = a.tag or "+".join(names)[:80]
    write_submission(inp.id.values, p, labels, res / "submissions" / f"{a.task}_{split_name}_{tag}")


if __name__ == "__main__":
    main()
