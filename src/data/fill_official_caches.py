"""
Fill the transliteration and translation caches for the official split comments they miss.

  python -m src.data.fill_official_caches

data/xlit_kn.json and data/mt_en.json were built from data/raw before the official splits came
out, so they cover only 368 of the 806 official Task A test comments. The comments to add are
read from data/HASTIKA_task splits/*.csv when that folder is present (the official CSVs stay out
of git, so on Kaggle the notebook downloads it first), else from data/cache_todo_official.json.
Same settings as the first build: IndicXlit beam 4, then NLLB-200 600M from the Kannada script,
4 beams. Needs fairseq, so run it on Kaggle or Linux.
"""
import json
import shutil
import zipfile
from pathlib import Path

from src.data import translate, transliterate

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "data" / "HASTIKA_task splits"
TODO = ROOT / "data" / "cache_todo_official.json"


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
    """The comments missing from either cache."""
    if OFFICIAL.is_dir() and any(OFFICIAL.glob("*.csv")):
        texts = official_texts()
    else:
        texts = json.loads(TODO.read_text(encoding="utf-8"))
    x, m = transliterate.load_cache(), translate.load_cache()
    return [t for t in texts if t not in x or t not in m]


def main():
    texts = todo_texts()
    print(f"{len(texts)} cau can bo sung")
    transliterate.build_cache(texts, lang="kn", beam=4)
    translate.build_cache(texts, source="kn", model="nllb", batch_size=24, num_beams=4)
    x, m = transliterate.load_cache(), translate.load_cache()
    print(f"thieu sau khi chay: xlit {sum(t not in x for t in texts)}, mt {sum(t not in m for t in texts)}")
    out = Path("/kaggle/working")
    if out.exists():
        z = out / "fill_official_caches.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
            for p in (transliterate.CACHE, translate.CACHE):
                f.write(p, f"data/{p.name}")
        print(f"tai ve: {z} ({z.stat().st_size/1e6:.1f} MB), giai nen o thu muc goc cua repo")


if __name__ == "__main__":
    main()
