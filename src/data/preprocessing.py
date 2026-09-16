"""
Text cleaning + one fixed train/eval split.

  python -m src.data.preprocessing                # uses configs/base.yaml paths

Writes {processed_dir}/{task}_train.csv (id, text, label, y, group, is_val),
{task}_val.csv and, when a test file exists in raw_dir, {task}_test.csv,
plus {task}_split.json recording how the split was made.

`is_val` is the whole scheme: 0 = the fit slice, 1 = the held-out slice
(data.val_ratio, stratified by label). Every model fits on is_val == 0 and is
scored on is_val == 1, so their predictions line up row for row and can be blended.
Re-run after the test inputs are released: the split stays identical (same split_seed).
"""
import argparse
import html
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


def split_spec(cfg) -> dict:
    d = cfg["data"]
    return {"val_ratio": d["val_ratio"], "split_seed": d["split_seed"]}


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

    df.to_csv(processed_dir / f"{task}_train.csv", index=False)
    json.dump({"val_ratio": val_ratio, "split_seed": split_seed},
              open(processed_dir / f"{task}_split.json", "w"), indent=1)

    val = read_inputs(_find(raw_dir, spec["val"]))
    val.to_csv(processed_dir / f"{task}_val.csv", index=False)

    msg = ""
    tests = sorted(raw_dir.glob(spec["test_glob"])) or sorted(raw_dir.parent.glob(spec["test_glob"]))
    if tests:
        test = read_inputs(tests[0])
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
    stale = have is not None and {k: have.get(k) for k in want} != want
    if stale:
        log(f"split settings changed {have} -> {want}: rebuilding data/processed")
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
