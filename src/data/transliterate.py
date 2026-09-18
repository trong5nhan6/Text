"""
Roman -> native-script transliteration (Kanglish -> Kannada), cached on disk.

  python -m src.data.transliterate            # build data/xlit_kn.json from data/raw

Why a cache rather than calling the model inline: IndicXlit needs fairseq, which does not
build on Windows (torch's headers want /std:c++17 and fairseq's setup does not pass it) and
is slow anyway. So the cache is built once on Kaggle, committed to the repo, and every other
machine only reads the JSON -- no fairseq, no GPU, no network.

The keys are the *cleaned* text, the same string preprocessing.py puts in the `text` column,
so a lookup there is exact.
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "xlit_kn.json"


def _stub_urduhack():
    """Everything that has to happen before `from ai4bharat.transliteration import XlitEngine`.

    (1) ai4bharat.transliteration imports urduhack at module level to normalise Shahmukhi, and
        urduhack drags in TensorFlow. We only ever ask for Kannada, so register a no-op module
        under that name first.
    (2) Also applies the torch.load patch below. That is not what the name says, but notebook
        cells do not update on `git pull` -- only src/ does -- so a cell written before the two
        were split still calls just this one. Keeping it a superset means such a cell works
        after a pull instead of failing on a fix that is already in the repo.

    Must run in every process that imports XlitEngine, which is why it lives here and not in
    the notebook.
    """
    import sys
    import types
    if "urduhack" not in sys.modules:
        try:
            import urduhack                       # noqa: F401  -- the real one is fine if present
        except ImportError:
            stub = types.ModuleType("urduhack")
            stub.normalize = lambda s: s
            sys.modules["urduhack"] = stub
    _allow_fairseq_checkpoint()


def _allow_fairseq_checkpoint():
    """PyTorch 2.6 flipped torch.load's `weights_only` default from False to True. fairseq
    (2022) calls torch.load without the flag in load_checkpoint_to_cpu, and the IndicXlit
    checkpoint holds an argparse.Namespace, which the safe unpickler refuses:

        UnpicklingError: Unsupported global: GLOBAL argparse.Namespace

    Allowlisting the classes it needs is the targeted fix; falling back to weights_only=False
    covers whatever else the checkpoint carries. Unpickling arbitrary objects is only safe
    because this file is downloaded by AI4Bharat's own library from their release over https.
    """
    import argparse
    import torch
    try:
        allow = [argparse.Namespace]
        try:
            import omegaconf
            from omegaconf.base import ContainerMetadata, Metadata
            allow += [omegaconf.DictConfig, omegaconf.ListConfig, ContainerMetadata, Metadata]
        except Exception:
            pass
        torch.serialization.add_safe_globals(allow)
    except AttributeError:
        return                                   # torch < 2.6: nothing to do

    if not getattr(torch.load, "_hastika_patched", False):
        _orig = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)

        _load._hastika_patched = True
        torch.load = _load


def load_cache(path=CACHE) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def transliterate(texts, cache=None, path=CACHE):
    """-> (list of transliterated texts, number of cache misses).

    A miss falls back to the original string rather than failing: the pipeline stays runnable,
    and the count tells you the cache needs rebuilding (it will, once the test file arrives).
    """
    cache = load_cache(path) if cache is None else cache
    out = [cache.get(t, t) for t in texts]
    return out, sum(1 for t in texts if t not in cache)


def raw_texts(raw_dir=None) -> list:
    """Every cleaned comment in data/raw, deduplicated -- the full set worth transliterating."""
    from src.data.preprocessing import TASKS, _find, clean_text
    import pandas as pd
    raw_dir = Path(raw_dir or ROOT / "data" / "raw")
    seen = {}
    for spec in TASKS.values():
        names = [spec["train"], spec["val"]]
        paths = [_find(raw_dir, n) for n in names]
        paths += sorted(raw_dir.glob(spec["test_glob"]))          # present from 20 Sep
        for p in paths:
            df = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False)
            col = next(c for c in df.columns if c.strip().lower() in ("comment", "text"))
            for t in df[col].map(clean_text):
                seen[t] = None
    return [t for t in seen if t.strip()]


def build_cache(texts, lang="kn", beam=4, save_every=500, path=CACHE, log=print):
    """Fill the cache for whatever is missing. Saves as it goes, so a killed Kaggle session
    loses at most `save_every` sentences and the next run picks up where it stopped."""
    _stub_urduhack()
    _allow_fairseq_checkpoint()
    from ai4bharat.transliteration import XlitEngine
    path = Path(path)
    cache = load_cache(path)
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    log(f"{len(texts)} cau, da co {len(texts) - len(todo)}, can lam {len(todo)}")
    if not todo:
        return cache

    engine = XlitEngine(lang, beam_width=beam, src_script_type="roman")
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        try:
            cache[t] = engine.translit_sentence(t)[lang]
        except Exception as e:                       # one bad sentence must not kill the run
            log(f"  bo qua ({type(e).__name__}): {t[:60]}")
            cache[t] = t
        if i % save_every == 0 or i == len(todo):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            rate = i / (time.time() - t0)
            log(f"  {i}/{len(todo)}  ({rate:.1f} cau/s, con ~{(len(todo)-i)/max(rate,1e-9)/60:.0f} phut)")
    log(f"xong: {len(cache)} cau trong {path} ({path.stat().st_size/1e6:.1f} MB)")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="kn", help="ISO code of the target script, kn = Kannada")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--out", default=str(CACHE))
    a = ap.parse_args()
    texts = raw_texts()
    print(f"data/raw: {len(texts)} cau duy nhat")
    cache = build_cache(texts, lang=a.lang, beam=a.beam, path=Path(a.out))
    sample = [t for t in texts if 3 <= len(t.split()) <= 6][:5]
    print("\nVi du:")
    for t in sample:
        print(f"  {t[:55]:57s} -> {cache.get(t, '(thieu)')[:55]}")


if __name__ == "__main__":
    main()
