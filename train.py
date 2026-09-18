#!/usr/bin/env python3
"""
Train one model on the fit slice and score it on the held-out slice.

  python train.py --config configs/tfidf.yaml   --task a
  python train.py --config configs/muril.yaml   --task b
  python train.py --config configs/roberta.yaml --task b --set training.loss=focal --run_name xlmr_focal

Outputs
  results/{task}/{run}/  eval.npy val.npy [test.npy] metrics.json config.yaml
  checkpoints/{task}/{run}/          (best epoch, if checkpoint.save=best)
  logs/{task}_{run}.log ;  results/metrics.csv  (all runs)

A finished run is not retrained; changing hyper-parameters under the same run name
is refused. Use --run_name or --overwrite.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.dataset import (infer_columns, label_names, load_split, require_columns,
                              split_rows, text_columns)
from src.data.preprocessing import ensure_processed
from src.evaluation.metrics import compute_metrics, rebuild_metrics_table
from src.evaluation.submission import write_submission
from src.utils.config import dump, load_config, resolve_loss, run_name, training_signature
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", choices=["a", "b"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run_name")
    ap.add_argument("--run_suffix", help="appended to the default run name, e.g. _e6")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--overwrite", action="store_true", help="delete previous results of this run")
    return ap.parse_args()


def check_run_dir(run_dir: Path, cfg, overwrite, log) -> bool:
    """-> True if this run is already finished and should be skipped."""
    cfg_file = run_dir / "config.yaml"
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
        shutil.rmtree(Path(cfg["paths"]["checkpoint_dir"]) / cfg["task"] / run_dir.name, ignore_errors=True)
        log(f"removed previous results of {run_dir.name}")
    elif cfg_file.exists():
        old = yaml.safe_load(open(cfg_file, encoding="utf-8"))
        if training_signature(old) != training_signature(cfg):
            raise SystemExit(f"{run_dir} was trained with a different config. "
                             f"Use another --run_name or pass --overwrite.")
        if (run_dir / "eval.npy").exists():
            log(f"{run_dir.name} is already trained -> nothing to do (use --overwrite to redo it)")
            return True
    dump(cfg, cfg_file)
    return False


def train_transformer(cfg, train, val, test, n_labels, run_dir, log):
    """log: the logger object (Trainer calls log.info itself)."""
    import torch
    from src.models.factory import build_model, build_tokenizer
    from src.training.losses import build_loss
    from src.training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_name = resolve_loss(cfg)
    cols, infer_cols = text_columns(cfg), infer_columns(cfg)
    fit, ev = split_rows(require_columns(train, set(cols) | set(infer_cols), "train"))
    require_columns(val, infer_cols, "val")
    if test is not None:
        require_columns(test, infer_cols, "test")

    # An encoder shares its parameters across both scripts, so here "both" means extra rows,
    # not extra features. Only the fit slice is duplicated: a Kannada copy of an eval row is
    # the same comment, and putting it in training would leak.
    if cols == ["text", "text_kn"]:
        fit, train_col = pd.concat([fit, fit.assign(text=fit.text_kn)], ignore_index=True), "text"
    else:
        train_col = cols[0]
    counts = np.bincount(fit.y, minlength=n_labels)
    log.info(f"device={device} model={cfg['model']['name']} loss={loss_name} "
             f"precision={cfg['training']['precision']} | text_type={cfg['data']['text_type'] or 'latin'}"
             f"{' +tta' if cfg['data'].get('tta') else ''}"
             f" | {len(fit)} fit / {len(ev)} eval")

    set_seed(cfg["seed"])
    t0 = time.time()
    log.info(f"tai tokenizer + trong so {cfg['model']['name']} ...")   # quiet: no HF progress bars
    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg, n_labels)
    log.info(f"san sang sau {time.time() - t0:.0f}s")
    log.info(f"freeze: {model.freeze_summary}")
    trainer = Trainer(cfg, model, tokenizer, build_loss(loss_name, cfg["training"], counts), device, log)
    best = trainer.fit(fit.assign(text=fit[train_col]), ev.assign(text=ev[train_col]))

    def predict(df):
        """Average over the requested views. fit() leaves the model at its best epoch, so one
        view here reproduces best["pred"] exactly -- which is why it is reused when tta is off."""
        return sum(trainer.predict(df[c]) for c in infer_cols) / len(infer_cols)

    single = infer_cols == [train_col]
    out = {"eval": best["pred"] if single else predict(ev),
           "val": predict(val), "test": None if test is None else predict(test),
           "extra": {"model": cfg["model"]["name"], "loss": loss_name, "best_epoch": best["epoch"]}}
    json.dump(best["history"], open(run_dir / "history.json", "w"), indent=1)
    if cfg["checkpoint"]["save"] == "best":
        ck = Path(cfg["paths"]["checkpoint_dir"]) / cfg["task"] / run_dir.name
        model.save(ck, tokenizer, half=cfg["checkpoint"].get("half", True),
                   extra={"epoch": best["epoch"], "macro_f1": best["f1"], "task": cfg["task"],
                          "labels": label_names(cfg["task"]), "max_len": cfg["data"]["max_len"]})
        log.info(f"checkpoint -> {ck}")
    return out


def main():
    a = parse()
    cfg = load_config(a.config, a.set, task=a.task, seed=a.seed,
                      run_name=a.run_name, run_suffix=a.run_suffix)
    name = run_name(cfg)
    res_dir = Path(cfg["paths"]["results_dir"])
    run_dir = res_dir / cfg["task"] / name
    log = get_logger("hastika", Path(cfg["paths"]["log_dir"]) / f"{cfg['task']}_{name}.log")
    log.info(f"===== task {cfg['task']} | run {name} | config {a.config} =====")
    if check_run_dir(run_dir, cfg, a.overwrite, log.info):
        return

    ensure_processed(cfg, log.info)
    train, val, test = load_split(cfg, "train"), load_split(cfg, "val"), load_split(cfg, "test")
    labels = label_names(cfg["task"])
    set_seed(cfg["seed"])

    if cfg["model"]["type"] == "tfidf":
        from src.models.tfidf import fit_and_score
        best = fit_and_score(cfg, train, val, test, len(labels), log.info)
        out = {"eval": best["eval"], "val": best["val"], "test": best["test"],
               "extra": {"model": f"tfidf-{cfg['model'].get('clf', 'lr')} {best['unit']}={best['C']}",
                         "loss": "balanced" if best["balanced"] else "-", "best_C": best["C"]}}
    else:
        out = train_transformer(cfg, train, val, test, len(labels), run_dir, log)

    np.save(run_dir / "eval.npy", out["eval"]); np.save(run_dir / "val.npy", out["val"])
    if out["test"] is not None:
        np.save(run_dir / "test.npy", out["test"])
    _, ev = split_rows(train)
    metrics = {**compute_metrics(ev.y, out["eval"].argmax(1)), **out["extra"],
               "n_eval": len(ev), "has_test": out["test"] is not None, "labels": labels}
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=1)
    rebuild_metrics_table(res_dir)

    # ready-to-submit file for every split we have predictions for
    subs = res_dir / "submissions"
    write_submission(val.id.values, out["val"], labels, subs / f"{cfg['task']}_val_{name}", log.info)
    if out["test"] is not None:
        write_submission(test.id.values, out["test"], labels, subs / f"{cfg['task']}_test_{name}", log.info)
    log.info(f"==> {cfg['task']}/{name} [{len(ev)} eval rows]: "
             f"macro-F1 {metrics['macro_f1']:.4f} | acc {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
