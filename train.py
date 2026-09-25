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
                              split_rows, text_columns, use_valdataset)
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
    # action="extend": a second --set must add to the first, not replace it. With plain nargs="*"
    # `--set a=1 --set b=2` silently keeps only b=2, and the first override vanishes without a word.
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
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
    from src.models.factory import (build_featurizer, build_model, build_side_vocab, build_tokenizer,
                                    load_mix_tables)
    from src.training.losses import build_loss
    from src.training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_name = resolve_loss(cfg)
    cols, infer_cols = text_columns(cfg), infer_columns(cfg)
    fit, ev = split_rows(require_columns(train, set(cols) | set(infer_cols), "train"),
                         use_valdataset(cfg))
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
             f" | {len(fit)} fit / {len(ev)} eval"
             + ("  [use_valdataset=false: train tren 100% du lieu, khong cham diem duoc]"
                if len(ev) == 0 else ""))

    set_seed(cfg["seed"])
    t0 = time.time()
    featurizer = None
    if cfg["model"].get("hybrid"):
        # The TF-IDF branch is fitted on Latin text, so a second script would reach it as an
        # empty vector: tta would average in a prediction made without the branch, and
        # text_type=kn/both would fit it on a script it never sees at test time.
        if cfg["data"].get("tta") or cols != ["text"]:
            raise SystemExit("model.hybrid chi dung voi data.text_type=latin va khong tta: nhanh "
                             "TF-IDF hoc tren chu Latin, chu Kannada di vao se la vector rong.")
        # Fit slice only -- the eval rows must not shape the vocabulary.
        featurizer = build_featurizer(cfg, fit[train_col])
        log.info(f"hybrid tfidf: {featurizer.dim} dac trung (fit tren {len(fit)} dong) "
                 f"-> {cfg['model'].get('hybrid_dim', 256)} chieu, ghep vao vector pooled")
    log.info(f"tai tokenizer + trong so {cfg['model']['name']} ...")   # quiet: no HF progress bars
    tokenizer = build_tokenizer(cfg)
    side_vocab = None
    if cfg["model"].get("side_embedding"):
        # Same reasons as the hybrid: fitted on Latin text, on the fit slice only.
        if cfg["data"].get("tta") or cols != ["text"]:
            raise SystemExit("model.side_embedding chi dung voi data.text_type=latin va khong tta.")
        side_vocab = build_side_vocab(cfg, fit[train_col], tokenizer)
        log.info(f"side embedding {side_vocab.kind}: {side_vocab.n_chars} ky tu, {side_vocab.n_keys} "
                 f"khoa phien am (>= {side_vocab.min_count} lan trong lat fit), gate khoi tao = 0")
    model = build_model(cfg, n_labels, featurizer, side_vocab)
    if model.mix_cfg:
        model.set_mix_tables(load_mix_tables(cfg, tokenizer, log.info))
        log.info(f"embed_mix ({model.mix_cfg['mode']}): bang cua encoder + "
                 f"{len(model.mix_cfg['sources'])} bang dong bang, trong so tron khoi tao = 0")
    log.info(f"san sang sau {time.time() - t0:.0f}s")
    log.info(f"freeze: {model.freeze_summary}")
    trainer = Trainer(cfg, model, tokenizer, build_loss(loss_name, cfg["training"], counts), device, log)
    best = trainer.fit(fit.assign(text=fit[train_col]),
                       ev.assign(text=ev[train_col]) if len(ev) else None)

    def predict(df):
        """Average over the requested views. fit() leaves the model at its best epoch, so one
        view here reproduces best["pred"] exactly -- which is why it is reused when tta is off."""
        return sum(trainer.predict(df[c]) for c in infer_cols) / len(infer_cols)

    single = infer_cols == [train_col]
    if model.mix_cfg:
        # What the mix settled on, over the fit slice's real tokens. 0 = the source was ignored;
        # 1 = its table replaced the encoder's for those tokens.
        import torch
        enc = tokenizer(fit[train_col].tolist()[:2000], truncation=True,
                        max_length=cfg["data"]["max_len"], padding=True, return_tensors="pt")
        dev = next(model.parameters()).device
        with torch.no_grad():
            a = model.mix_alphas(enc["input_ids"].to(dev)).float().cpu()
        real = (enc["attention_mask"] == 1) & ~torch.isin(enc["input_ids"],
                                                          torch.tensor(tokenizer.all_special_ids))
        for k, src in enumerate(model.mix_cfg["sources"]):
            ak = a[..., k][real]
            log.info(f"embed_mix a[{src}]: trung binh {ak.mean():+.4f} | |a| {ak.abs().mean():.4f} "
                     f"| p5..p95 {ak.quantile(.05):+.3f}..{ak.quantile(.95):+.3f}")
    gate = model.side_gate_norm()    # None unless model.side_embedding is on
    if gate is not None:
        # 0 would mean the side stream was never used -- worth knowing before crediting it
        log.info(f"side_gate: trung binh |g| = {gate:.4f} (0 = nhanh phu khong duoc dung)")
    mix = model.layer_mix()          # None unless model.layers == "mix"
    if mix is not None:
        spread = max(mix) - min(mix)
        log.info("layer_mix: " + " ".join(f"{i}:{w:.3f}" for i, w in enumerate(mix))
                 + f"  (phan hoa {spread:.4f}, deu = {1 / len(mix):.4f})")
        if spread < 0.01:
            # AdamW steps a parameter by about its lr, so too small a rate leaves the 13 logits
            # where they started and "mix" quietly becomes a plain average over all layers. That
            # failure is invisible in the score, so say it out loud.
            log.info(f"!! trong so mix gan nhu KHONG doi -> mix dang chi la trung binh deu. "
                     f"Tang training.layer_mix_lr (dang {cfg['training'].get('layer_mix_lr')}) "
                     f"len 5e-3 hoac train nhieu step hon.")
    out = {"eval": None if len(ev) == 0 else (best["pred"] if single else predict(ev)),
           "val": predict(val), "test": None if test is None else predict(test),
           "extra": {"model": cfg["model"]["name"], "loss": loss_name, "best_epoch": best["epoch"],
                     **({"layer_mix": mix} if mix is not None else {})}}
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

    np.save(run_dir / "val.npy", out["val"])
    if out["test"] is not None:
        np.save(run_dir / "test.npy", out["test"])
    _, ev = split_rows(train, use_valdataset(cfg))
    # No eval.npy at all when there is no held-out slice, rather than an empty one: evaluate.py
    # discovers runs by that file, so this keeps an unscorable run out of every comparison and
    # every blend instead of letting it contribute predictions nobody checked.
    if out["eval"] is not None:
        np.save(run_dir / "eval.npy", out["eval"])
    metrics = {**(compute_metrics(ev.y, out["eval"].argmax(1)) if out["eval"] is not None else {}),
               **out["extra"], "n_eval": len(ev), "has_test": out["test"] is not None,
               "labels": labels, "scored": out["eval"] is not None}
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=1)
    rebuild_metrics_table(res_dir)

    # ready-to-submit file for every split we have predictions for
    subs = res_dir / "submissions"
    write_submission(val.id.values, out["val"], labels, subs / f"{cfg['task']}_val_{name}", log.info)
    if out["test"] is not None:
        write_submission(test.id.values, out["test"], labels, subs / f"{cfg['task']}_test_{name}", log.info)
    if out["eval"] is None:
        log.info(f"==> {cfg['task']}/{name}: train tren 100% du lieu ({len(train)} dong), "
                 f"KHONG co diem noi bo. File nop da sinh; run nay khong vao blend.")
    else:
        log.info(f"==> {cfg['task']}/{name} [{len(ev)} eval rows]: "
                 f"macro-F1 {metrics['macro_f1']:.4f} | acc {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
