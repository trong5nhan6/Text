#!/usr/bin/env python3
"""
Cross-validated training on the fixed folds.

  python train.py --config configs/tfidf.yaml   --task a
  python train.py --config configs/muril.yaml   --task b
  python train.py --config configs/roberta.yaml --task b --set training.loss=focal --run_name xlmr_focal
  python train.py --config configs/muril.yaml   --task a --folds 0 1        # only some folds (resume later)

Outputs
  results/{task}/{run}/  oof.npy val.npy [test.npy] metrics.json config.yaml per-fold cache in folds/
  checkpoints/{task}/{run}/fold{k}/   (best epoch per fold, if checkpoint.save=best)
  logs/{task}_{run}.log ;  results/metrics.csv  (all runs)
Finished folds are cached: rerunning the same command resumes. Changing hyper-parameters under the
same run name is refused (use --run_name or --overwrite).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.data.dataset import label_names, load_split
from src.data.preprocessing import ensure_processed
from src.evaluation.metrics import compute_metrics, rebuild_metrics_table, summarize_folds
from src.utils.config import dump, load_config, resolve_loss, run_name, training_signature
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", choices=["a", "b"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run_name")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--overwrite", action="store_true", help="delete previous results of this run")
    return ap.parse_args()


def check_run_dir(run_dir: Path, cfg, overwrite, log):
    cfg_file = run_dir / "config.yaml"
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
        ck = Path(cfg["paths"]["checkpoint_dir"]) / cfg["task"] / run_dir.name
        shutil.rmtree(ck, ignore_errors=True)
        log.info(f"removed previous results of {run_dir.name}")
    elif cfg_file.exists():
        old = yaml.safe_load(open(cfg_file, encoding="utf-8"))
        if training_signature(old) != training_signature(cfg):
            raise SystemExit(f"{run_dir} was trained with a different config. "
                             f"Use another --run_name or pass --overwrite.")
    dump(cfg, cfg_file)


def train_transformer(cfg, train, val, test, n_labels, run_dir, log):
    import torch
    from src.models.factory import build_model, build_tokenizer
    from src.training.losses import build_loss
    from src.training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_name = resolve_loss(cfg)
    counts = np.bincount(train.y, minlength=n_labels)
    log.info(f"device={device} model={cfg['model']['name']} loss={loss_name} "
             f"precision={cfg['training']['precision']}")
    tokenizer = build_tokenizer(cfg)
    ck_root = Path(cfg["paths"]["checkpoint_dir"]) / cfg["task"] / run_dir.name
    cache = run_dir / "folds"; cache.mkdir(parents=True, exist_ok=True)

    all_folds = sorted(int(f) for f in train.fold.unique())
    for k in (cfg["_folds"] if cfg["_folds"] is not None else all_folds):
        f = cache / f"fold{k}.npz"
        if f.exists():
            log.info(f"fold {k}: cached -> skip"); continue
        set_seed(cfg["seed"] + k)
        tr, va = train[train.fold != k], train[train.fold == k]
        model = build_model(cfg, n_labels)
        trainer = Trainer(cfg, model, tokenizer, build_loss(loss_name, cfg["training"], counts), device, log)
        best = trainer.fit(tr, va, tag=f"[fold {k}]")
        p_val = trainer.predict(val.text)
        p_test = trainer.predict(test.text) if test is not None else None
        if cfg["checkpoint"]["save"] == "best":
            model.save(ck_root / f"fold{k}", tokenizer, half=cfg["checkpoint"].get("half", True),
                       extra={"fold": k, "epoch": best["epoch"], "macro_f1": best["f1"],
                              "task": cfg["task"], "labels": label_names(cfg["task"]),
                              "max_len": cfg["data"]["max_len"]})
        np.savez(f, oof=best["oof"], val=p_val, test=p_test if p_test is not None else np.empty(0),
                 f1=best["f1"], epoch=best["epoch"])
        json.dump(best["history"], open(cache / f"fold{k}_history.json", "w"), indent=1)
        log.info(f"fold {k}: best macro-F1 {best['f1']:.4f} @ epoch {best['epoch']}")
        del trainer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    missing = [k for k in all_folds if not (cache / f"fold{k}.npz").exists()]
    if missing:
        log.info(f"folds {missing} not trained yet -> rerun to aggregate"); return None

    oof = np.zeros((len(train), n_labels)); pv = np.zeros((len(val), n_labels))
    pt = np.zeros((len(test), n_labels)) if test is not None else None
    fold_f1, epochs = [], []
    for k in all_folds:
        z = np.load(cache / f"fold{k}.npz")
        oof[train.index[train.fold == k]] = z["oof"]
        pv += z["val"] / len(all_folds)
        if pt is not None:
            if z["test"].size == 0:
                log.warning(f"fold {k} was trained before the test file existed -> no test preds. "
                            f"Use inference.py --checkpoints (if checkpoints were saved) or retrain.")
                pt = None
            else:
                pt += z["test"] / len(all_folds)
        fold_f1.append(float(z["f1"])); epochs.append(int(z["epoch"]))
    return {"oof": oof, "val": pv, "test": pt, "fold_f1": fold_f1,
            "extra": {"model": cfg["model"]["name"], "loss": loss_name, "best_epochs": epochs}}


def main():
    a = parse()
    cfg = load_config(a.config, a.set, task=a.task, seed=a.seed, run_name=a.run_name)
    name = run_name(cfg)
    res_dir = Path(cfg["paths"]["results_dir"])
    run_dir = res_dir / cfg["task"] / name
    log = get_logger("hastika", Path(cfg["paths"]["log_dir"]) / f"{cfg['task']}_{name}.log")
    log.info(f"===== task {cfg['task']} | run {name} | config {a.config} =====")
    check_run_dir(run_dir, cfg, a.overwrite, log)
    cfg["_folds"] = a.folds

    ensure_processed(cfg, log.info)
    train, val, test = load_split(cfg, "train"), load_split(cfg, "val"), load_split(cfg, "test")
    labels = label_names(cfg["task"])
    set_seed(cfg["seed"])

    if cfg["model"]["type"] == "tfidf":
        from src.models.tfidf import cross_validate
        best = cross_validate(cfg, train, val, test, len(labels), log.info)
        out = {"oof": best["oof"], "val": best["val"], "test": best["test"], "fold_f1": best["fold_f1"],
               "extra": {"model": f"tfidf-{cfg['model'].get('clf', 'lr')} C={best['C']}",
                         "loss": "balanced" if best["balanced"] else "-", "best_C": best["C"]}}
    else:
        out = train_transformer(cfg, train, val, test, len(labels), run_dir, log)
        if out is None:
            return

    np.save(run_dir / "oof.npy", out["oof"]); np.save(run_dir / "val.npy", out["val"])
    if out["test"] is not None:
        np.save(run_dir / "test.npy", out["test"])
    metrics = {**compute_metrics(train.y, out["oof"].argmax(1)), **summarize_folds(out["fold_f1"]),
               **out["extra"], "has_test": out["test"] is not None, "labels": labels}
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=1)
    rebuild_metrics_table(res_dir)
    log.info(f"==> {cfg['task']}/{name}: OOF macro-F1 {metrics['macro_f1']:.4f} | acc {metrics['accuracy']:.4f} "
             f"| folds {metrics['fold_f1_mean']:.4f} ± {metrics['fold_f1_std']:.4f}")


if __name__ == "__main__":
    main()
