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

# Every file that holds Kanglish, labelled or not: (path or glob, text column). The test files
# are globs because their names are the organisers' (hastika_binary_test.csv), the same patterns
# preprocessing.py uses; the unlabelled test text is ordinary transductive use -- say so in the paper.
SOURCES = [
    ("data/raw/binary_train.csv", "Comment"),
    ("data/raw/multiclass_train.csv", "Comment"),
    ("data/raw/binary_validation_inputs.csv", "Comment"),
    ("data/raw/multiclass_validation_inputs.csv", "Comment"),
    ("data/raw/*binary*test*.csv", "Comment"),
    ("data/raw/*multiclass*test*.csv", "Comment"),
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
    files = [(p, col) for path, col in SOURCES
             for p in (sorted(Path().glob(path)) if any(c in path for c in "*?[") else [Path(path)])]
    for p, col in files:
        path = str(p)
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


class LossOnly(torch.nn.Module):
    """Return the loss and nothing else, for DataParallel.

    DataParallel gathers every tensor a forward returns onto the first device. An MLM forward
    returns the loss *and* the [batch, len, vocab] logits, and with MuRIL's 197,285-entry
    vocabulary those logits are the whole memory problem -- gathering them would pile every
    replica's copy onto GPU 0 and undo the point of splitting the batch. Returning the loss alone
    leaves the logits on the device that produced them.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, **batch):
        # 1-D, not scalar: DataParallel concatenates replica outputs along dim 0.
        return self.inner(**batch).loss.unsqueeze(0)


class Encoded(Dataset):
    def __init__(self, texts, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len)
        # word index of every token (None for [CLS]/[SEP]) -- what whole-word masking groups by.
        # Only a fast tokenizer can say; without one the collator falls back to token masking.
        self.words = [self.enc.word_ids(i) for i in range(len(texts))] if tok.is_fast else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "word_ids": None if self.words is None else self.words[i]}


class MaskCollator:
    """BERT's masking: pick ~`prob` of the tokens; of those 80% -> [MASK], 10% -> a random token,
    10% left as they are; the loss is taken on the picked positions only (labels -100 elsewhere).

    whole_word: pick WORDS, and mask every piece of a picked word together. Kanglish is cut into
    2-4 pieces per word (thu -> th ##u, maklu -> ma ##k ##lu); masking ##u while 'th' stays
    visible is a task the model solves from the neighbouring piece without reading the sentence.
    Grouping by the tokenizer's word_ids(), rather than by '##', works for any fast tokenizer.

    `seed` fixes the draw -- used for the evaluation split, so every epoch is scored on the same
    masks and the numbers are comparable. None draws from the global RNG (seeded by set_seed).
    """

    def __init__(self, tok, prob=0.15, whole_word=False, seed=None):
        self.tok, self.prob, self.whole_word = tok, prob, whole_word
        self.g = None if seed is None else torch.Generator().manual_seed(seed)
        self.special = set(tok.all_special_ids)

    def _pick(self, ids, words):
        n = len(ids)
        real = torch.tensor([t not in self.special for t in ids])
        if not self.whole_word or words is None:
            return real & (torch.rand(n, generator=self.g) < self.prob)
        sizes = {}
        for w in words:
            if w is not None:
                sizes[w] = sizes.get(w, 0) + 1
        if not sizes:
            return torch.zeros(n, dtype=torch.bool)
        budget = max(1, round(self.prob * int(real.sum())))
        order = [list(sizes)[j] for j in torch.randperm(len(sizes), generator=self.g).tolist()]
        chosen, covered = set(), 0
        for w in order:
            if covered >= budget:
                break
            # A long word may overshoot the budget a little rather than be skipped: skipping it
            # (what a strict budget does) would never mask exactly the long, fragmented Kanglish
            # words this is for.
            if covered and covered + sizes[w] > 1.5 * budget:
                continue
            chosen.add(w)
            covered += sizes[w]
        return torch.tensor([w in chosen for w in words]) & real

    def __call__(self, batch):
        B, T = len(batch), max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long)
        attention = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            x = torch.tensor(b["input_ids"], dtype=torch.long)
            n = len(x)
            input_ids[i, :n], attention[i, :n] = x, 1
            pick = self._pick(b["input_ids"], b["word_ids"])
            labels[i, :n][pick] = x[pick]
        picked = labels != -100
        r = torch.rand((B, T), generator=self.g)
        input_ids[picked & (r < 0.8)] = self.tok.mask_token_id
        rand = picked & (r >= 0.8) & (r < 0.9)
        input_ids[rand] = torch.randint(len(self.tok), (B, T), generator=self.g)[rand]
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def copy_mlm_head(model, donor_name, log):
    """Give a classification checkpoint (e.g. cnerg_muril, a BertForSequenceClassification) the
    MLM head it never had, taken from the checkpoint it was fine-tuned from. Without this the
    head's transform and bias start random, and the first epochs are spent learning them while
    their gradient runs back through -- and disturbs -- the encoder this is meant to adapt.

    The decoder weight is NOT copied: it is tied to this model's own word embeddings."""
    from transformers import AutoModelForMaskedLM
    if not hasattr(model, "cls"):
        raise SystemExit(f"--mlm_head_from chi ho tro kien truc BERT (co model.cls); "
                         f"model nay la {type(model).__name__}")
    donor = AutoModelForMaskedLM.from_pretrained(donor_name)
    if donor.config.vocab_size != model.config.vocab_size or donor.config.hidden_size != model.config.hidden_size:
        raise SystemExit(f"--mlm_head_from {donor_name}: vocab/hidden "
                         f"{donor.config.vocab_size}/{donor.config.hidden_size} khong khop "
                         f"{model.config.vocab_size}/{model.config.hidden_size}")
    sd = {k: v for k, v in donor.cls.predictions.state_dict().items() if k != "decoder.weight"}
    model.cls.predictions.load_state_dict(sd, strict=False)
    same = all(torch.equal(model.cls.predictions.state_dict()[k], v) for k, v in sd.items())
    log(f"dau MLM chep tu {donor_name}: {sorted(sd)} -> {'OK' if same else 'LOI: khong khop'}")
    if not same:
        raise SystemExit("chep dau MLM that bai")
    del donor


@torch.no_grad()
def eval_loss(model, batches, device, amp):
    """Mean masked-token loss over the fixed evaluation batches (token-weighted)."""
    model.eval()
    tot, n = 0.0, 0
    for b in batches:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=amp or torch.float32, enabled=amp is not None):
            loss = model(**b).loss
        k = int((b["labels"] != -100).sum())
        tot, n = tot + float(loss) * k, n + k
    model.train()
    return tot / max(n, 1)


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
    ap.add_argument("--single_gpu", action="store_true",
                    help="use one GPU even when several are visible. Otherwise every visible GPU "
                         "is used, and --batch_size is the TOTAL split across them.")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--max_rows", type=int, default=0,
                    help="keep only the first N corpus rows. For smoke-testing the whole script, "
                         "save included, without waiting for a real run.")
    ap.add_argument("--include_eval", action="store_true",
                    help="also pretrain on the held-out slice. Every macro-F1 measured on that "
                         "slice afterwards becomes optimistic; only do this for a final model.")
    ap.add_argument("--wwm", action="store_true",
                    help="whole-word masking: mask all pieces of a word together (see MaskCollator)")
    ap.add_argument("--eval_ratio", type=float, default=0.0,
                    help="hold this share of the corpus out of MLM training, score its masked-token "
                         "perplexity every epoch, and save only the best epoch. 0 = old behaviour: "
                         "no evaluation, the last epoch is saved.")
    ap.add_argument("--mlm_head_from",
                    help="checkpoint to copy the MLM head from when --model has none, e.g. "
                         "google/muril-base-cased for Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL")
    return ap.parse_args()


def main():
    a = parse()
    from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup
    from src.training.trainer import resolve_precision

    out_dir = Path(a.out or f"checkpoints/mlm/{a.model.rstrip('/').split('/')[-1]}")
    log = get_logger("mlm", Path("logs") / f"mlm_{out_dir.name}.log")
    log.info(f"===== MLM | model {a.model} -> {out_dir} =====")

    cfg = load_config(a.config, task="a")
    ensure_processed(cfg, log.info)
    corpus = build_corpus(cfg, a.include_eval, log.info)
    if not corpus:
        raise SystemExit("kho van ban rong -- kiem tra data/raw")
    if a.max_rows:
        corpus = corpus[:a.max_rows]
        log.info(f"--max_rows {a.max_rows}: cat kho con {len(corpus)} dong (chi de smoke-test)")

    set_seed(a.seed)
    # Evaluation split: a fixed, seeded share of the corpus the MLM never trains on. Perplexity on
    # the training text only ever falls; on this it turns back up once the model starts learning
    # the 0.3M-token corpus by heart, and that is the epoch to stop at.
    ev_texts = []
    if a.eval_ratio > 0:
        import random
        idx = list(range(len(corpus)))
        random.Random(a.seed).shuffle(idx)
        n_ev = max(1, int(round(a.eval_ratio * len(corpus))))
        ev_texts = [corpus[i] for i in idx[:n_ev]]
        corpus = [corpus[i] for i in idx[n_ev:]]
        log.info(f"tach {len(ev_texts):,} dong ({a.eval_ratio:.0%}) lam tap danh gia MLM, "
                 f"con {len(corpus):,} dong de train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    model, info = AutoModelForMaskedLM.from_pretrained(a.model, output_loading_info=True)
    missing_head = sorted(k for k in info["missing_keys"] if k.startswith("cls."))
    if missing_head:
        # A classification checkpoint (cnerg_muril) has no MLM head: loading it as a masked LM
        # leaves the head random. Refuse rather than silently spend the run relearning it.
        if not a.mlm_head_from:
            raise SystemExit(f"{a.model} khong co dau MLM (thieu {missing_head}). Them "
                             f"--mlm_head_from <checkpoint goc>, vd google/muril-base-cased.")
        copy_mlm_head(model, a.mlm_head_from, log.info)
    elif a.mlm_head_from:
        log.info(f"{a.model} da co dau MLM -> bo qua --mlm_head_from")
    model = model.float().to(device)
    amp = resolve_precision("auto", device)
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    use_dp = n_gpu > 1 and not a.single_gpu
    # `model` stays the real module throughout: the optimizer and save_pretrained both need it,
    # and DataParallel only wraps it for the forward pass.
    net = torch.nn.DataParallel(LossOnly(model)) if use_dp else model
    if use_dp:
        log.info(f"DataParallel tren {n_gpu} GPU -- batch {a.batch_size} chia thanh "
                 f"{a.batch_size // n_gpu}/GPU, nen VRAM moi GPU giam tuong ung")

    ds = Encoded(corpus, tok, a.max_len)
    n_tok = sum(len(x) for x in ds.enc["input_ids"])
    if a.wwm and ds.words is None:
        log.info("!! --wwm can tokenizer fast (word_ids) -> quay ve che tung token")
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2,
                    collate_fn=MaskCollator(tok, a.mlm_probability, whole_word=a.wwm))
    # the evaluation masks are drawn once, with a fixed seed: same positions every epoch
    ev_batches = []
    if ev_texts:
        ev_ds = Encoded(ev_texts, tok, a.max_len)
        ev_batches = list(DataLoader(ev_ds, batch_size=a.batch_size, shuffle=False,
                                     collate_fn=MaskCollator(tok, a.mlm_probability, whole_word=a.wwm,
                                                             seed=a.seed)))
    log.info(f"kho: {len(corpus):,} dong, {n_tok:,} token ({n_tok / len(corpus):.1f} token/dong) | "
             f"che {'CA TU (wwm)' if a.wwm and ds.words is not None else 'tung token'} "
             f"{a.mlm_probability:.0%}")
    log.info(f"device={device} precision={amp} | {a.epochs} epoch x {len(dl)} step "
             f"(batch {a.batch_size} x grad_accum {a.grad_accum} = {a.batch_size * a.grad_accum} "
             f"hieu dung), lr {a.lr:g}")

    # The MLM head emits [batch, len, vocab] and MuRIL's vocab is 197,285, so that one tensor --
    # kept twice, forwards and backwards -- dominates everything else and is what OOMs a T4.
    # Print the estimate before training rather than after the crash.
    n_par = sum(p.numel() for p in model.parameters())
    vocab = model.config.vocab_size
    per_gpu = a.batch_size / max(n_gpu, 1) if use_dp else a.batch_size
    gb_logits = 2 * per_gpu * a.max_len * vocab * 4 / 1e9
    gb_model = n_par * 16 / 1e9          # weights + grads + AdamW's two states, fp32
    # Under DataParallel only GPU 0 carries the master weights, the gradients and AdamW's two
    # states; the others hold a weight replica alone. So GPU 0 is the one that runs out first.
    gb_gpu0 = gb_logits + gb_model + 1
    log.info(f"uoc luong VRAM: logits {gb_logits:.2f} GB/GPU "
             f"({per_gpu:g} dong x {a.max_len} len x vocab {vocab:,})"
             + (f" | GPU0 + model/grad/AdamW {gb_model:.2f} GB -> ~{gb_gpu0:.1f} GB, "
                f"GPU khac ~{gb_logits + gb_model / 4 + 1:.1f} GB" if use_dp else
                f" + model/AdamW {gb_model:.2f} GB + kich hoat ~1 GB = ~{gb_gpu0:.1f} GB"))
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  moi GPU co {total:.1f} GB"
                 + ("  -- CHAT, giam --batch_size hoac --max_len neu OOM"
                    if gb_gpu0 > 0.75 * total else ""))

    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and any(k in n for k in ("bias", "LayerNorm.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps = math.ceil(len(dl) / a.grad_accum) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(a.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)

    def save(tag):
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        log.info(f"da luu -> {out_dir} ({tag})")

    best = (float("inf"), 0)
    if ev_batches:
        l0 = eval_loss(model, ev_batches, device, amp)
        log.info(f"truoc khi train: eval loss {l0:.4f} ppl {torch.tensor(l0).exp():.2f}")
    model.train()
    for ep in range(1, a.epochs + 1):
        t0, total = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp or torch.float32,
                                enabled=amp is not None):
                # `fwd`, not `out`: `out_dir` used to be called `out`, and this line shadowed it
                # so the run trained for its full schedule and then died on out.mkdir().
                fwd = net(**batch)
                loss = fwd.mean() if use_dp else fwd.loss   # DataParallel -> one loss per replica
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
        msg = f"ep {ep}/{a.epochs} loss {mean:.4f} ppl {torch.tensor(mean).exp():.2f}"
        if ev_batches:
            le = eval_loss(model, ev_batches, device, amp)
            better = le < best[0]
            msg += f" | eval loss {le:.4f} ppl {torch.tensor(le).exp():.2f}{' *' if better else ''}"
            if better:
                best = (le, ep)
        log.info(msg + f" ({time.time() - t0:.0f}s)")
        if ev_batches and best[1] == ep:
            # Saved as it improves rather than kept in memory: a Kaggle session that times out
            # still leaves the best epoch so far on disk.
            save(f"epoch {ep}, eval ppl {torch.tensor(best[0]).exp():.2f}")

    if ev_batches:
        log.info(f"epoch tot nhat: {best[1]} (eval ppl {torch.tensor(best[0]).exp():.2f}) -- da luu o tren")
        log.info(f"dung: python train.py --config configs/<config cua {a.model}>.yaml --task b "
                 f"--set model.name={out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    # save_pretrained, not torch.save: the point is for train.py to load this with
    # --set model.name=<out>, which goes through AutoModel.from_pretrained.
    #
    # Two warnings show up around this and both are harmless:
    #   "OrderedVocab ... contains holes"  -- MuRIL's vocabulary has unused indices. Verified that
    #       the embedding matrix round-trips bit-exact through save/load anyway.
    #   "pooler.dense.* MISSING" on the next load -- an MLM head has no pooler, so the checkpoint
    #       carries none and AutoModel makes a fresh one. It never trains (_apply_freezing freezes
    #       pooler.*) and never reaches the output (forward reads last_hidden_state).
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log.info(f"da luu -> {out_dir}")
    log.info(f"dung: python train.py --config configs/muril.yaml --task b "
             f"--set model.name={out_dir} --run_suffix _mlm")


if __name__ == "__main__":
    main()
