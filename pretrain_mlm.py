#!/usr/bin/env python3
"""
Domain-adaptive pretraining: run masked-language modelling over the Kanglish corpus, then
fine-tune from the result instead of from the public checkpoint.

  python pretrain_mlm.py --model google/muril-base-cased --epochs 15
  python train.py --config configs/muril.yaml --task b --set model.name=checkpoints/mlm/muril-base-cased

Why this and not simply adding the external rows to training: MuRIL's vocabulary was built for
Indian languages in their own scripts, so romanised Kanglish is shredded into pieces -- measured
at 2.05 subwords per word against 1.04 for English, with only 36.7% of words surviving as a
single token. `Bajetigu` becomes Ba / ##jet / ##ig / ##u, and the embeddings of those pieces were
learned in contexts that have nothing to do with Kannada. MLM moves them; it needs no labels, so
it also sidesteps the reason concatenating the external offensive-language rows does nothing
(measured +0.0012 +/- 0.0087): "offensive" and "hate" are different labels, but the text is the
same language either way.

Scale, stated plainly: the corpus is ~0.31M tokens where MuRIL saw ~16B and published DAPT work
uses 1-100M. What makes it not hopeless is that only 8,405 vocabulary entries (4.3%) ever occur,
so the budget concentrates on exactly the embeddings that are wrong. Expect a point or two, not
a leap.

The held-out slice is excluded by default. Its text carries no label, so including it would not
leak labels -- but the model would have read those exact sentences, and every macro-F1 measured
on that slice afterwards would be quietly optimistic. It costs 639 rows of 14,502 to stay honest.
The organisers' unlabelled val/test inputs *are* included: that is ordinary transductive use, and
it is the setting the submission actually runs in. Say so in the paper.
"""
import argparse
import math
import time
from pathlib import Path

import pandas as pd
import src.utils.hf_quiet    # noqa: F401  -- import before transformers: silences its bars
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import clean_text, dedup_key, ensure_processed
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

# Every file that holds Kanglish, labelled or not. (path, column, needs_cleaning)
SOURCES = [
    ("data/raw/binary_train.csv", "Comment"),
    ("data/raw/multiclass_train.csv", "Comment"),
    ("data/raw/binary_validation_inputs.csv", "Comment"),
    ("data/raw/multiclass_validation_inputs.csv", "Comment"),
    ("data/raw/binary_test_inputs.csv", "Comment"),
    ("data/raw/multiclass_test_inputs.csv", "Comment"),
    ("data/external_offenseval_kn.csv", "text"),
]


def build_corpus(cfg, include_eval: bool, log=print):
    """-> a de-duplicated list of cleaned comments."""
    held_out = set()
    if not include_eval:
        for task in ("a", "b"):
            f = Path(cfg["paths"]["processed_dir"]) / f"{task}_train.csv"
            if f.exists():
                d = pd.read_csv(f, keep_default_na=False)
                held_out |= {dedup_key(t) for t in d[d.is_val == 1].text}
        log(f"bo {len(held_out)} dong cua lat held-out (de diem noi bo van trung thuc)")

    seen, corpus = set(held_out), []
    for path, col in SOURCES:
        p = Path(path)
        if not p.exists():
            continue
        raw = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False)
        if col not in raw.columns:
            log(f"  bo qua {path}: khong co cot {col!r}")
            continue
        new = 0
        for t in raw[col]:
            t = clean_text(t)
            k = dedup_key(t)
            if k and k not in seen:
                seen.add(k)
                corpus.append(t)
                new += 1
        log(f"  {path:46s} {len(raw):>6} dong -> them {new:>6}")
    return corpus


class Encoded(Dataset):
    def __init__(self, texts, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--out", help="default: checkpoints/mlm/<model basename>")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="small on purpose: the MLM logits are [batch, len, vocab] and MuRIL's "
                         "vocab is 197,285, so batch 32 at len 128 needs 3.2 GB for one such "
                         "tensor and OOMs a T4. Raise the effective batch with --grad_accum.")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--include_eval", action="store_true",
                    help="also pretrain on the held-out slice. Every macro-F1 measured on that "
                         "slice afterwards becomes optimistic; only do this for a final model.")
    return ap.parse_args()


def main():
    a = parse()
    from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, get_linear_schedule_with_warmup)
    from src.training.trainer import resolve_precision

    out = Path(a.out or f"checkpoints/mlm/{a.model.rstrip('/').split('/')[-1]}")
    log = get_logger("mlm", Path("logs") / f"mlm_{out.name}.log")
    log.info(f"===== MLM | model {a.model} -> {out} =====")

    cfg = load_config(a.config, task="a")
    ensure_processed(cfg, log.info)
    corpus = build_corpus(cfg, a.include_eval, log.info)
    if not corpus:
        raise SystemExit("kho van ban rong -- kiem tra data/raw")

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model).float().to(device)
    amp = resolve_precision("auto", device)

    ds = Encoded(corpus, tok, a.max_len)
    n_tok = sum(len(x) for x in ds.enc["input_ids"])
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2,
                    collate_fn=DataCollatorForLanguageModeling(tok, mlm_probability=a.mlm_probability))
    log.info(f"kho: {len(corpus):,} dong, {n_tok:,} token ({n_tok / len(corpus):.1f} token/dong)")
    log.info(f"device={device} precision={amp} | {a.epochs} epoch x {len(dl)} step "
             f"(batch {a.batch_size} x grad_accum {a.grad_accum} = {a.batch_size * a.grad_accum} "
             f"hieu dung), lr {a.lr:g}")

    # The MLM head emits [batch, len, vocab] and MuRIL's vocab is 197,285, so that one tensor --
    # kept twice, forwards and backwards -- dominates everything else and is what OOMs a T4.
    # Print the estimate before training rather than after the crash.
    n_par = sum(p.numel() for p in model.parameters())
    vocab = model.config.vocab_size
    gb_logits = 2 * a.batch_size * a.max_len * vocab * 4 / 1e9
    gb_model = n_par * 16 / 1e9          # weights + grads + AdamW's two states, fp32
    log.info(f"uoc luong VRAM: logits {gb_logits:.2f} GB (batch x len x vocab {vocab:,}) "
             f"+ model/AdamW {gb_model:.2f} GB + kich hoat ~1 GB = ~{gb_logits + gb_model + 1:.1f} GB")
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  GPU co {total:.1f} GB"
                 + ("  -- CHAT, giam --batch_size hoac --max_len neu OOM"
                    if gb_logits + gb_model + 1 > 0.75 * total else ""))

    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and any(k in n for k in ("bias", "LayerNorm.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps = math.ceil(len(dl) / a.grad_accum) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)

    model.train()
    for ep in range(1, a.epochs + 1):
        t0, total = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp or torch.float32,
                                enabled=amp is not None):
                loss = model(**batch).loss
            total += loss.item()
            scaler.scale(loss / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0 or i + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # A skipped step (the scaler saw inf/nan and lowered its scale) must not advance
                # the schedule; stepping anyway is what produces the "lr_scheduler.step() before
                # optimizer.step()" warning, and it silently shortens the LR schedule.
                prev = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= prev:
                    sch.step()
                opt.zero_grad(set_to_none=True)
        mean = total / len(dl)
        # Perplexity is the number to watch: it starts high because the vocabulary pieces are
        # wrong for this text, and falling is exactly the adaptation this script exists to do.
        log.info(f"ep {ep}/{a.epochs} loss {mean:.4f} ppl {torch.tensor(mean).exp():.2f} "
                 f"({time.time() - t0:.0f}s)")

    out.mkdir(parents=True, exist_ok=True)
    # save_pretrained, not torch.save: the point is for train.py to load this with
    # --set model.name=<out>, which goes through AutoModel.from_pretrained.
    #
    # Two warnings show up around this and both are harmless:
    #   "OrderedVocab ... contains holes"  -- MuRIL's vocabulary has unused indices. Verified that
    #       the embedding matrix round-trips bit-exact through save/load anyway.
    #   "pooler.dense.* MISSING" on the next load -- an MLM head has no pooler, so the checkpoint
    #       carries none and AutoModel makes a fresh one. It never trains (_apply_freezing freezes
    #       pooler.*) and never reaches the output (forward reads last_hidden_state).
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log.info(f"da luu -> {out}")
    log.info(f"dung: python train.py --config configs/muril.yaml --task b "
             f"--set model.name={out} --run_suffix _mlm")


if __name__ == "__main__":
    main()
