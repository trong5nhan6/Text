#!/usr/bin/env python3
"""
Task A with an instruction-tuned LLM and NO training: zero-shot or retrieval few-shot prompting.

  python llm_prompt.py --model Qwen/Qwen2.5-7B-Instruct --mode fewshot --k 8 --load_in_4bit
  python llm_prompt.py --model Qwen/Qwen2.5-0.5B-Instruct --mode zeroshot --limit 20   # smoke test

Writes results/a/<run>/ exactly like train.py -- eval.npy (held-out), val.npy, test.npy,
metrics.json, config.yaml -- plus the val/test submissions, so evaluate.py, inference.py and the
final notebook's two-tier cascade read it like any other run.

How a comment is scored
  zeroshot  a definition of the two labels, then the comment.
  fewshot   the same, plus the k most similar comments of the FIT slice with their labels
            (char n-gram TF-IDF cosine), most similar last -- so the model sees how THIS dataset
            draws the line, which differs from everyday intuition in places (group criticism
            labelled Non-Hate, abuse of a named person labelled Hate).
  The answer is never generated: P(Hate) = softmax over the logits of the first token of
  "Hate" and of "Non-Hate" right after the assistant turn opens.

Calibration: an untrained LLM leans hard to one label, so its raw P(Hate) is shifted by one
constant so that the share of Hate predictions on the held-out slice matches the Hate rate of
the fit slice (~49%). That uses the label PRIOR only, never a held-out label.

Band only (--band_file, the second tier of a cascade): an .npz with, per split (eval/val/test),
`<split>_mask` (rows to score) and `<split>_p1` (the first tier's probabilities, n x 2). Only the
masked rows are scored -- a third of the time at delta 0.2 -- and every other row keeps p1, so
the saved arrays still cover every row and "the LLM alone" equals the cascade with mix 0. The
calibration target is then the first tier's mean P(Hate) on the masked held-out rows (the band
is not a random sample, so the fit-slice prior does not apply to it).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import src.utils.hf_quiet  # noqa: F401
from src.data.dataset import label_names, load_split, split_rows
from src.data.preprocessing import add_val_leak, dedup_key, ensure_processed
from src.evaluation.metrics import compute_metrics, rebuild_metrics_table
from src.evaluation.submission import write_submission
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

SYSTEM = ("You are an expert annotator of hate speech in YouTube comments written in Kannada-English "
          "code-mixed text (Kannada written in Latin script, mixed with English).")
DEFINITION = ("Label the comment as one of:\n"
              "Hate - the comment attacks, abuses, insults, threatens or demeans a person or a group "
              "(e.g. because of gender, religion, politics, region), including slurs and abusive language.\n"
              "Non-Hate - anything else, including neutral comments, opinions and criticism without abuse.\n"
              "Answer with exactly one word: Hate or Non-Hate.")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", choices=["zeroshot", "fewshot"], default="fewshot")
    ap.add_argument("--k", type=int, default=8, help="fewshot: examples per comment")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024, help="tokens per prompt; longer is cut from the left")
    ap.add_argument("--val_leak", action="store_true",
                    help="retrieval pool also holds the leak-labelled validation inputs (data.val_leak_labels)")
    ap.add_argument("--band_file", help="score only the rows this .npz masks (see the docstring)")
    ap.add_argument("--band_delta", type=float, help="the delta that built --band_file; only names the run")
    ap.add_argument("--no_calibration", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_name", help="default: llm<mode>_<model>[_k<k>][_vl][_band<delta>]")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows per split, write nothing")
    ap.add_argument("--config", default="configs/base.yaml")
    return ap.parse_args()


def default_run_name(model, mode, k, val_leak, band_delta=None):
    import re
    short = re.sub(r"[^a-z0-9.-]", "", model.split("/")[-1].lower())[:20]
    return (f"llm{mode}_{short}" + (f"_k{k}" if mode == "fewshot" else "") + ("_vl" if val_leak else "")
            + (f"_band{band_delta:g}" if band_delta is not None else ""))


def build_messages(text, examples):
    """examples: [(text, label)], most similar LAST."""
    user = DEFINITION
    if examples:
        user += "\n\nLabelled examples from the same dataset:\n" + "\n".join(
            f'Comment: "{t}"\nLabel: {l}' for t, l in examples)
    user += f'\n\nNow label this comment.\nComment: "{text}"\nLabel:'
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def label_token_ids(tok):
    """first-token ids that start each label, with and without a leading space."""
    ids = {"Hate": set(), "Non-Hate": set()}
    for lab in ids:
        for v in (lab, " " + lab):
            t = tok.encode(v, add_special_tokens=False)
            if t:
                ids[lab].add(t[0])
    if ids["Hate"] & ids["Non-Hate"]:
        raise SystemExit(f"tokenizer cat 'Hate' va 'Non-Hate' thanh cung token dau: {ids}")
    return {k: sorted(v) for k, v in ids.items()}


@torch.no_grad()
def score(model, tok, prompts, lab_ids, bs, max_len, device, log):
    """-> logit(Hate) - logit(Non-Hate) per prompt (log-sum-exp over each label's first tokens)."""
    out, t0 = [], time.time()
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        logits = model(**enc).logits[:, -1, :].float()             # left padding: last = next token
        lh = torch.logsumexp(logits[:, lab_ids["Hate"]], -1)
        ln = torch.logsumexp(logits[:, lab_ids["Non-Hate"]], -1)
        out.append((lh - ln).cpu().numpy())
        if (i // bs) % 25 == 0:
            log(f"  {i + len(prompts[i:i + bs])}/{len(prompts)} ({time.time() - t0:.0f}s)")
    return np.concatenate(out) if out else np.zeros(0)


def main():
    a = parse()
    set_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = load_config(a.config, ["data.val_leak_labels=true"] if a.val_leak else [], task="a")
    name = a.run_name or default_run_name(a.model, a.mode, a.k, a.val_leak, a.band_delta)
    res = Path(cfg["paths"]["results_dir"])
    run_dir = res / "a" / name
    log = get_logger("llm_prompt", Path(cfg["paths"]["log_dir"]) / f"a_{name}.log").info
    log(f"===== {a.mode} | {a.model} -> {name} =====")

    ensure_processed(cfg, log)
    train, val, test = load_split(cfg, "train"), load_split(cfg, "val"), load_split(cfg, "test")
    if a.val_leak:
        train = add_val_leak(cfg, train, val, log)
    fit, ev = split_rows(train, True)
    labels = label_names("a")                                       # ["Non-Hate", "Hate"]
    splits = {"eval": ev.text.tolist(), "val": val.text.tolist(),
              "test": [] if test is None else test.text.tolist()}
    p1 = {k: None for k in splits}                                  # first tier (band mode only)
    rows = {k: np.arange(len(v)) for k, v in splits.items()}        # which rows of each split get scored
    if a.band_file:
        z = np.load(a.band_file)
        for k, v in splits.items():
            if not v:
                continue
            mask, p1[k] = z[f"{k}_mask"].astype(bool), z[f"{k}_p1"]
            if len(mask) != len(v) or len(p1[k]) != len(v):
                raise SystemExit(f"{a.band_file}: {k} co {len(mask)} dong, tap {k} co {len(v)} cau")
            rows[k] = np.flatnonzero(mask)
        log("chi cham vung khong chac: " + ", ".join(f"{k} {len(rows[k])}/{len(v)}" for k, v in splits.items() if v))
    if a.limit:
        splits = {k: v[:a.limit] for k, v in splits.items()}
        rows = {k: r[r < a.limit] for k, r in rows.items()}
        p1 = {k: None if v is None else v[:a.limit] for k, v in p1.items()}
        log(f"--limit {a.limit}: chi cham {a.limit} cau dau moi tap, KHONG ghi ket qua")
    todo = {k: [splits[k][i] for i in rows[k]] for k in splits}     # the texts actually sent to the LLM

    # ---- retrieval (fewshot): k most similar fit comments, never the comment itself
    shots = {k: [[] for _ in v] for k, v in todo.items()}
    if a.mode == "fewshot":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(fit.text)
        F = vec.transform(fit.text)
        fkeys = fit.text.map(dedup_key).to_numpy()
        for k, texts in todo.items():
            if not texts:
                continue
            S = (vec.transform(texts) @ F.T).toarray()
            qkeys = np.array([dedup_key(t) for t in texts])
            S[qkeys[:, None] == fkeys[None, :]] = -1                  # no copy of the comment itself
            top = np.argsort(-S, 1)[:, :a.k][:, ::-1]                # most similar last
            shots[k] = [[(fit.text.iat[j], fit.label.iat[j]) for j in row] for row in top]
        log(f"fewshot: {a.k} vi du / cau, lay tu {len(fit)} cau phan fit")

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"                                       # last position = next token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{a.model} khong co chat template -- dung ban *-Instruct / *-it")
    lab_ids = label_token_ids(tok)
    kw = {"dtype": torch.float16}
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        kw = {"dtype": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto" if device.type == "cuda" else None, **kw)
    model.eval()
    log(f"model nap xong ({device}); token nhan: {lab_ids}")

    d = {}
    for k, texts in todo.items():
        prompts = [tok.apply_chat_template(build_messages(t, ex), tokenize=False, add_generation_prompt=True)
                   for t, ex in zip(texts, shots[k])]
        if prompts:
            log(f"{k}: {len(prompts)} cau, prompt dai TB {np.mean([len(tok(p).input_ids) for p in prompts[:50]]):.0f} token")
        d[k] = score(model, tok, prompts, lab_ids, a.batch_size, a.max_len, device, log)

    if p1["eval"] is not None:                                      # band: the first tier's own estimate there
        prior = float(p1["eval"][rows["eval"], 1].mean()) if len(rows["eval"]) else 0.5
    else:
        prior = float((fit.label == "Hate").mean())
    bias = 0.0 if a.no_calibration or len(d["eval"]) == 0 else float(np.quantile(d["eval"], 1 - prior))

    def full(k, b):
        """scores of the scored rows -> n x 2 probabilities for every row of split k."""
        q = 1 / (1 + np.exp(-(d[k] - b)))
        out = np.zeros((len(splits[k]), 2)) if p1[k] is None else p1[k].astype(float).copy()
        out[rows[k]] = np.stack([1 - q, q], 1)
        return out

    y_ev = ev.y.to_numpy()[:len(splits["eval"])]
    raw = compute_metrics(y_ev, full("eval", 0.0).argmax(1))
    cal = compute_metrics(y_ev, full("eval", bias).argmax(1))
    r = rows["eval"]
    if len(r):
        log(f"tren {len(r)} cau held-out LLM cham: acc KHONG hieu chinh {((d['eval'] > 0) == y_ev[r]).mean():.3f} "
            f"(du doan Hate {(d['eval'] > 0).mean():.0%}) | hieu chinh (bias {bias:+.2f}, Hate ~ {prior:.0%}) "
            f"{((d['eval'] > bias) == y_ev[r]).mean():.3f}"
            + ("" if p1["eval"] is None else f" | tang 1 tren cung cac cau: {(p1['eval'][r].argmax(1) == y_ev[r]).mean():.3f}"))
    log(f"held-out ca {len(y_ev)} cau: macro-F1 KHONG hieu chinh {raw['macro_f1']:.4f} | hieu chinh {cal['macro_f1']:.4f}")
    if a.limit:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    for k in ("eval", "val", "test"):
        if len(splits[k]):
            np.save(run_dir / f"{k}.npy", full(k, bias))
    yaml.safe_dump({"task": "a", "run_name": name, "llm_prompt": vars(a), "calibration_bias": bias},
                   open(run_dir / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    json.dump({**cal, "model": a.model, "loss": a.mode, "n_eval": len(y_ev), "labels": labels,
               "has_test": test is not None, "scored": True, "macro_f1_uncalibrated": raw["macro_f1"],
               "calibration_bias": bias, "band_only": bool(a.band_file),
               "n_scored": {k: int(len(v)) for k, v in rows.items()}}, open(run_dir / "metrics.json", "w"), indent=1)
    write_submission(val.id.values, full("val", bias), labels, res / "submissions" / f"a_val_{name}", log)
    if test is not None:
        write_submission(test.id.values, full("test", bias), labels, res / "submissions" / f"a_test_{name}", log)
    rebuild_metrics_table(res)
    log(f"==> a/{name} [{len(y_ev)} eval rows]: macro-F1 {cal['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
