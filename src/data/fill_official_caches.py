"""
Fill the transliteration and translation caches for the official split comments they miss.

  python -m src.data.fill_official_caches --step todo   # write the missing comments to --todo
  python -m src.data.fill_official_caches --step xlit   # IndicXlit, needs fairseq
  python -m src.data.fill_official_caches --step mt     # NLLB, plain transformers
  python -m src.data.fill_official_caches --step zip    # both caches -> /kaggle/working

data/xlit_kn.json and data/mt_en.json were built from data/raw before the official splits came
out, so they cover only 368 of the 806 official Task A test comments. The comments to add are
read from data/HASTIKA_task splits/*.csv (the official CSVs stay out of git, so on Kaggle the
notebook downloads them first). Same settings as the first build: IndicXlit beam 4, then
NLLB-200 600M from the Kannada script, 4 beams.

The steps are separate because fairseq has no build for the Python 3.13 of Kaggle: the xlit step
runs in a Python 3.9 venv that has a prebuilt fairseq wheel, the others in the main kernel. The
xlit step imports only src.data.transliterate, so the venv needs no pandas or ftfy.
"""
import argparse
import json
import zipfile
from pathlib import Path

from src.data import translate, transliterate

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "data" / "HASTIKA_task splits"
TODO = Path("/kaggle/working/todo_official.json")


def official_texts() -> list:
    """Every cleaned comment of the official splits, deduplicated."""
    import pandas as pd
    from src.data.preprocessing import clean_text
    seen = {}
    for p in sorted(OFFICIAL.glob("*.csv")):
        df = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False)
        col = next(c for c in df.columns if c.strip().lower() in ("comment", "text"))
        for t in df[col].map(clean_text):
            seen[t] = None
    return [t for t in seen if t.strip()]


def todo_texts() -> list:
    """The comments of the official splits missing from either cache."""
    x, m = transliterate.load_cache(), translate.load_cache()
    return [t for t in official_texts() if t not in x or t not in m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["todo", "xlit", "mt", "zip"], required=True)
    ap.add_argument("--todo", default=str(TODO))
    ap.add_argument("--out", default="/kaggle/working/fill_official_caches.zip")
    a = ap.parse_args()
    todo = Path(a.todo)

    if a.step == "todo":
        texts = todo_texts()
        todo.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        print(f"{len(texts)} cau can bo sung -> {todo}")
        return

    texts = json.loads(todo.read_text(encoding="utf-8"))
    if a.step == "xlit":
        transliterate.build_cache(texts, lang="kn", beam=4)
    elif a.step == "mt":
        translate.build_cache(texts, source="kn", model="nllb", batch_size=24, num_beams=4)
    else:
        x, m = transliterate.load_cache(), translate.load_cache()
        miss = sum(t not in x for t in texts), sum(t not in m for t in texts)
        print(f"thieu sau khi chay: xlit {miss[0]}, mt {miss[1]}")
        if any(miss):
            raise SystemExit("cache chua du, khong dong goi")
        out = Path(a.out)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as f:
            for p in (transliterate.CACHE, translate.CACHE):
                f.write(p, f"data/{p.name}")
        print(f"tai ve: {out} ({out.stat().st_size/1e6:.1f} MB), giai nen o thu muc goc cua repo")


if __name__ == "__main__":
    main()
