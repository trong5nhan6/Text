"""
Machine translation to English, cached on disk. The TRAA arm of the
"transliterate or translate?" comparison (Puranik et al., FIRE 2021).

  python -m src.data.translate --source kn       # from the Kannada transliteration (default)
  python -m src.data.translate --source roman    # straight from the romanised text

Translation models expect native script, not romanised text, so the default source is
data/xlit_kn.json -- build that first with src.data.transliterate. `--source roman` feeds the
original Kanglish in directly; it is the weaker setting but worth measuring once.

Like the transliteration cache this runs once on Kaggle and the JSON is committed, so no other
machine needs the model. Keys are the *cleaned* original text in both cases, so a lookup from
preprocessing.py is the same call whichever source was used.
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "mt_en.json"

# NLLB is the easy option: plain transformers, no fairseq, no extra toolkit, and Kannada is one
# of its 200 languages. IndicTrans2 is stronger for Indic->English but needs IndicTransToolkit.
MODELS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "nllb-1.3b": "facebook/nllb-200-distilled-1.3B",
}


def load_cache(path=CACHE) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def translate(texts, cache=None, path=CACHE):
    """-> (list of English texts, number of cache misses). A miss returns the original."""
    cache = load_cache(path) if cache is None else cache
    return [cache.get(t, t) for t in texts], sum(1 for t in texts if t not in cache)


def build_cache(texts, source="kn", model="nllb", batch_size=24, max_len=128,
                num_beams=4, path=CACHE, log=print):
    """texts: the cleaned ORIGINAL comments -- they stay the cache keys whatever is fed to the
    model. Saves every batch, so an interrupted Kaggle session resumes instead of restarting."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    path = Path(path)
    cache = load_cache(path)
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    log(f"{len(texts)} cau, da co {len(texts) - len(todo)}, can dich {len(todo)}")
    if not todo:
        return cache

    if source == "kn":                       # translate the Kannada transliteration
        from src.data.transliterate import load_cache as load_xlit
        xlit = load_xlit()
        if not xlit:
            raise SystemExit("data/xlit_kn.json chua co -- chay src.data.transliterate truoc, "
                             "hoac dung --source roman")
        missing = [t for t in todo if t not in xlit]
        if missing:
            log(f"canh bao: {len(missing)} cau thieu trong cache chuyen tu -> dung ban goc")
        src_texts = [xlit.get(t, t) for t in todo]
        src_lang = "kan_Knda"
    else:                                    # feed the romanised text straight in
        src_texts, src_lang = list(todo), "kan_Knda"

    name = MODELS.get(model, model)
    log(f"nap {name} (src_lang={src_lang}, source={source})")
    tok = AutoTokenizer.from_pretrained(name, src_lang=src_lang)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device).eval()
    eng_id = tok.convert_tokens_to_ids("eng_Latn")

    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        keys, batch = todo[i:i + batch_size], src_texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(device)
        with torch.no_grad():
            gen = mdl.generate(**enc, forced_bos_token_id=eng_id,
                               max_new_tokens=max_len, num_beams=num_beams)
        for k, out in zip(keys, tok.batch_decode(gen, skip_special_tokens=True)):
            cache[k] = out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        done = min(i + batch_size, len(todo))
        rate = done / (time.time() - t0)
        log(f"  {done}/{len(todo)}  ({rate:.1f} cau/s, con ~{(len(todo)-done)/max(rate,1e-9)/60:.0f} phut)")
    log(f"xong: {len(cache)} cau trong {path} ({path.stat().st_size/1e6:.1f} MB)")
    return cache


def main():
    from src.data.transliterate import raw_texts
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["kn", "roman"], default="kn",
                    help="kn = dich tu ban chuyen tu (giong bai bao) | roman = dich thang tu chu Latin")
    ap.add_argument("--model", default="nllb", help=f"{sorted(MODELS)} hoac ten model HF bat ky")
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--out", default=str(CACHE))
    a = ap.parse_args()

    texts = raw_texts()
    print(f"data/raw: {len(texts)} cau duy nhat")
    cache = build_cache(texts, source=a.source, model=a.model, batch_size=a.batch_size,
                        num_beams=a.num_beams, path=Path(a.out))
    print("\nVi du:")
    for t in [t for t in texts if 4 <= len(t.split()) <= 9][:6]:
        print(f"  {t[:50]:52s} -> {cache.get(t, '(thieu)')[:60]}")


if __name__ == "__main__":
    main()
