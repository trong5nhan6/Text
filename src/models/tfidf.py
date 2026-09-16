"""B0 — TF-IDF (char + word n-grams) + linear classifier, cross-validated on the fixed folds."""
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

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


def cross_validate(cfg, train, val, test, n_labels, log=print):
    mcfg = cfg["model"]
    cw = mcfg.get("class_weight", "auto")
    balanced = (cfg["task"] == "b") if cw == "auto" else cw == "balanced"
    folds = sorted(int(f) for f in train.fold.unique())
    best = None
    for C in mcfg.get("C_grid", [mcfg.get("C", 1.0)]):
        oof = np.zeros((len(train), n_labels)); pv = np.zeros((len(val), n_labels))
        pt = None if test is None else np.zeros((len(test), n_labels))
        fold_f1 = []
        for k in folds:
            tr, va = train[train.fold != k], train[train.fold == k]
            pipe = build_tfidf(mcfg, C, balanced).fit(tr.text, tr.y)
            oof[va.index] = _proba(pipe, va.text)
            fold_f1.append(compute_metrics(va.y, oof[va.index].argmax(1))["macro_f1"])
            pv += _proba(pipe, val.text) / len(folds)
            if test is not None:
                pt += _proba(pipe, test.text) / len(folds)
        m = compute_metrics(train.y, oof.argmax(1))
        log(f"  C={C}: OOF macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} "
            f"(fold {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f})")
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m["macro_f1"], "C": C, "oof": oof, "val": pv, "test": pt,
                    "fold_f1": fold_f1, "balanced": balanced}
    return best
