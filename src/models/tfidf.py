"""B0 — TF-IDF (char + word n-grams) + linear classifier on the fixed train/eval split."""
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from src.data.dataset import split_rows
from src.evaluation.metrics import compute_metrics


def build_tfidf(mcfg: dict, C: float, balanced: bool):
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(mcfg.get("char_ngram", [2, 5])),
                                 min_df=2, sublinear_tf=True, max_features=mcfg.get("max_features"))),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=tuple(mcfg.get("word_ngram", [1, 2])),
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[\U0001F000-\U0001FAFF]")),
    ])
    cw = "balanced" if balanced else None
    clf = (LogisticRegression(C=C, max_iter=5000, class_weight=cw) if mcfg.get("clf", "lr") == "lr"
           else LinearSVC(C=C, class_weight=cw))
    return make_pipeline(feats, clf)


def _proba(pipe, texts):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(texts)
    d = pipe.decision_function(texts)
    if d.ndim == 1:
        d = np.stack([-d, d], 1)
    return softmax(d * 2.0, axis=1)      # pseudo-probabilities, only used for ensembling


def fit_and_score(cfg, train, val, test, n_labels, log=print):
    """Fit one model per C on the fit slice, keep the C with the best held-out macro-F1."""
    mcfg = cfg["model"]
    cw = mcfg.get("class_weight", "auto")
    balanced = (cfg["task"] == "b") if cw == "auto" else cw == "balanced"
    fit, ev = split_rows(train)
    best = None
    for C in mcfg.get("C_grid", [mcfg.get("C", 1.0)]):
        pipe = build_tfidf(mcfg, C, balanced).fit(fit.text, fit.y)
        p_eval = _proba(pipe, ev.text)
        m = compute_metrics(ev.y, p_eval.argmax(1))
        log(f"  C={C}: macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m["macro_f1"], "C": C, "balanced": balanced, "eval": p_eval,
                    "val": _proba(pipe, val.text),
                    "test": None if test is None else _proba(pipe, test.text)}
    return best
