"""
Text cleaning + one fixed train/eval split.

  python -m src.data.preprocessing                # uses configs/base.yaml paths

Writes {processed_dir}/{task}_train.csv (id, text, text_kn, text_en, label, group, y, is_val),
{task}_val.csv and, when a test file exists in raw_dir, {task}_test.csv,
plus {task}_split.json recording how the split was made.

`text_kn` is the same comment in Kannada script (data/xlit_kn.json) and `text_en` its English
machine translation (data/mt_en.json); each column is simply absent when its cache is not there.

`is_val` is the whole scheme: 0 = the fit slice, 1 = the held-out slice
(data.val_ratio, stratified by label). Every model fits on is_val == 0 and is
scored on is_val == 1, so their predictions line up row for row and can be blended.
Re-run after the test inputs are released: the split stays identical (same split_seed).
"""
import argparse
import hashlib
import html
import inspect
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TASKS = {
    "a": {
        "labels": ["Non-Hate", "Hate"],
        "train": "binary_train.csv", "val": "binary_validation_inputs.csv",
        "test_glob": "binary*test*.csv", "label_col": "Label",
    },
    "b": {
        "labels": ["Gender", "Political", "Religion", "Geo-political", "Violence", "Others"],
        "train": "multiclass_train.csv", "val": "multiclass_validation_inputs.csv",
        "test_glob": "multiclass*test*.csv", "label_col": "Hate Category",
    },
}

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
URL = re.compile(r"https?://\S+|www\.\S+")


def clean_text(t) -> str:
    import ftfy
    t = ftfy.fix_text(str(t))                       # repair mojibake (~14% of rows)
    t = html.unescape(t)
    t = re.sub(r"<br\s*/?>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = URL.sub(" URL ", t)
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)          # "thuuuuu" -> "thuuu"
    t = unicodedata.normalize("NFC", t)
    return re.sub(r"\s+", " ", t).strip()


def dedup_key(t: str) -> str:
    t = EMOJI.sub("", t.lower())
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"[^\w\s]", "", t).strip()


def _find(raw_dir: Path, name: str) -> Path:
    for d in (raw_dir, raw_dir.parent):             # also accept the organiser layout data/*.csv
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {raw_dir}")


def read_inputs(path) -> pd.DataFrame:
    """Any CSV with an id column and a Comment/text column -> (id, text)."""
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    id_col = next(c for c in df.columns if c.strip().lower() in ("id", "index"))
    txt_col = next(c for c in df.columns if c.strip().lower() in ("comment", "text"))
    return pd.DataFrame({"id": df[id_col], "text": df[txt_col].map(clean_text)})


def add_transliteration(df, log=print):
    """Add a `text_kn` column: the same comment in Kannada script, looked up in
    data/xlit_kn.json (built once on Kaggle -- see notebooks/build_xlit_cache.ipynb).

    Only a column, never a replacement: `text` stays byte-identical, so every finished run
    remains comparable. Models ignore it until something is wired up to read it.
    Skipped silently when the cache is absent, so the pipeline runs without it.
    """
    from src.data.transliterate import load_cache, transliterate
    cache = load_cache()
    if not cache:
        log("  data/xlit_kn.json khong co -> bo qua cot text_kn")
        return df
    out, miss = transliterate(df.text.tolist(), cache=cache)
    df.insert(df.columns.get_loc("text") + 1, "text_kn", out)
    log(f"  text_kn: {len(df) - miss}/{len(df)} tra duoc trong cache"
        + (f", {miss} cau thieu -> giu nguyen ban goc" if miss else ""))
    return df


def add_translation(df, log=print):
    """Add a `text_en` column: the same comment machine-translated to English, from
    data/mt_en.json (built once on Kaggle -- see notebooks/build_mt_cache.ipynb).

    Same contract as add_transliteration: a column, never a replacement, and skipped silently
    when the cache is absent.

    Read the translations before trusting this view. Measured on a 12-comment sample: NLLB
    sanitises or literalises slurs (`Dagar` -> "dagger"), hallucinates (`Husulimaga`, a slur, ->
    "What is the meaning of life?"), and 2.4% of the full cache degenerates into repetition. MT
    destroys exactly the surface features this task is decided on, so treat `text_en` as an
    ablation arm, not as an improvement.
    """
    from src.data.translate import load_cache, translate
    cache = load_cache()
    if not cache:
        log("  data/mt_en.json khong co -> bo qua cot text_en")
        return df
    out, miss = translate(df.text.tolist(), cache=cache)
    after = "text_kn" if "text_kn" in df.columns else "text"
    df.insert(df.columns.get_loc(after) + 1, "text_en", out)
    log(f"  text_en: {len(df) - miss}/{len(df)} tra duoc trong cache"
        + (f", {miss} cau thieu -> giu nguyen ban goc" if miss else ""))
    return df


def code_fingerprint() -> str:
    """Hash of the functions that decide what lands in data/processed. Without it, editing
    clean_text leaves a stale data/processed in place and training silently continues on the
    old text -- val_ratio and split_seed alone cannot notice that."""
    src = "".join(inspect.getsource(f)
                  for f in (clean_text, dedup_key, read_inputs, add_transliteration,
                            add_translation, prepare_task))
    # join the lines back with nothing: the hash must not depend on CRLF vs LF, so a
    # checkout on Windows and one on Kaggle agree
    return hashlib.sha256("".join(src.splitlines()).encode()).hexdigest()[:12]


def split_spec(cfg) -> dict:
    d = cfg["data"]
    return {"val_ratio": d["val_ratio"], "split_seed": d["split_seed"], "code": code_fingerprint()}


def prepare_task(task: str, raw_dir, processed_dir, val_ratio=0.1, split_seed=42, log=print):
    spec = TASKS[task]
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(_find(raw_dir, spec["train"]), encoding="utf-8-sig", keep_default_na=False)
    df = pd.DataFrame({"id": raw["id"], "text": raw["Comment"].map(clean_text),
                       "label": raw[spec["label_col"]].str.strip()})
    unknown = set(df.label) - set(spec["labels"])
    assert not unknown, f"unknown labels {unknown}"
    df["group"] = df.text.map(dedup_key)

    # duplicates: drop groups with conflicting labels, keep one row otherwise
    n0 = len(df)
    n_lab = df.groupby("group").label.transform("nunique")
    n_conflict = int((n_lab > 1).sum())
    df = df[n_lab == 1].drop_duplicates("group").reset_index(drop=True)
    df["y"] = df.label.map({l: i for i, l in enumerate(spec["labels"])})

    # one stratified slice; groups are unique after de-duplication, so no duplicate can straddle it
    _, va = train_test_split(np.arange(len(df)), test_size=val_ratio,
                             stratify=df.y, random_state=split_seed)
    df["is_val"] = 0
    df.loc[va, "is_val"] = 1

    df = add_translation(add_transliteration(df, log), log)
    df.to_csv(processed_dir / f"{task}_train.csv", index=False)
    json.dump({"val_ratio": val_ratio, "split_seed": split_seed, "code": code_fingerprint()},
              open(processed_dir / f"{task}_split.json", "w"), indent=1)

    val = add_translation(add_transliteration(read_inputs(_find(raw_dir, spec["val"])), log), log)
    val.to_csv(processed_dir / f"{task}_val.csv", index=False)

    msg = ""
    tests = sorted(raw_dir.glob(spec["test_glob"])) or sorted(raw_dir.parent.glob(spec["test_glob"]))
    if tests:
        test = add_translation(add_transliteration(read_inputs(tests[0]), log), log)
        test.to_csv(processed_dir / f"{task}_test.csv", index=False)
        msg = f", test={len(test)} ({tests[0].name})"
    n_eval = int(df.is_val.sum())
    log(f"[task {task}] train {n0} -> {len(df)} (dropped {n_conflict} conflicting + "
        f"{n0 - n_conflict - len(df)} duplicate rows), val={len(val)}{msg}")
    log(f"[task {task}] split: {len(df) - n_eval} fit / {n_eval} eval ({val_ratio:.0%})")
    return df


def ensure_processed(cfg, log=print):
    """Build data/processed on demand, and rebuild it when the split settings changed."""
    p = Path(cfg["paths"]["processed_dir"])
    task, want = cfg["task"], split_spec(cfg)
    have_file = p / f"{task}_split.json"
    have = json.load(open(have_file)) if have_file.exists() else None
    changed = {} if have is None else {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
    stale = bool(changed)
    if stale:
        what = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in changed.items())
        log(f"rebuilding data/processed ({what})")
    if not (p / f"{task}_train.csv").exists() or have is None or stale:
        prepare_task(task, cfg["paths"]["raw_dir"], p, want["val_ratio"], want["split_seed"], log)


def main():
    from src.utils.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tasks", nargs="+", default=["a", "b"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    spec = split_spec(cfg)
    for t in a.tasks:
        df = prepare_task(t, cfg["paths"]["raw_dir"], cfg["paths"]["processed_dir"],
                          spec["val_ratio"], spec["split_seed"])
        print(pd.crosstab(df.is_val, df.label).to_string(), "\n")


if __name__ == "__main__":
    main()
