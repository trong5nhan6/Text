"""B0 — TF-IDF (char + word n-grams) + a linear/NB classifier on the fixed train/eval split."""
import numpy as np
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data.dataset import require_columns, split_rows, text_columns
from src.evaluation.metrics import compute_metrics

# Regularisation grid per classifier. `lr`/`svm` take C (inverse strength, bigger = weaker),
# the rest take alpha (direct strength, bigger = stronger), so the useful ranges differ.
DEFAULT_GRIDS = {
    "lr":    [0.5, 1, 2, 4, 8, 16],
    "svm":   [0.05, 0.1, 0.25, 0.5, 1],
    "ridge": [0.1, 0.5, 1, 3, 10],
    "cnb":   [0.01, 0.05, 0.1, 0.3, 1],
    "sgd":   [1e-6, 1e-5, 1e-4],
}
# classifiers with no predict_proba: without calibration their scores go through a softmax
# of the decision function, which is monotone (argmax unchanged) but on an arbitrary scale --
# fine alone, misleading when blended against a model that reports real probabilities.
NEEDS_CALIBRATION = {"svm", "ridge"}


WORD_PATTERN = r"(?u)\b\w+\b|[\U0001F000-\U0001FAFF]"     # keep 1-char words and emoji


def build_features(mcfg: dict, columns=("text",)) -> ColumnTransformer:
    """char + word n-grams for each text column. With two columns the blocks are concatenated
    on the same row, so the model sees both scripts at once. Extra rows would not work here:
    Kannada n-grams share no characters with Latin ones, so features learned from one view
    never fire on the other."""
    blocks = []
    for col in columns:
        blocks += [
            (f"char_{col}", TfidfVectorizer(analyzer="char_wb",
                                            ngram_range=tuple(mcfg.get("char_ngram", [2, 5])),
                                            min_df=2, sublinear_tf=True,
                                            max_features=mcfg.get("max_features")), col),
            (f"word_{col}", TfidfVectorizer(analyzer="word",
                                            ngram_range=tuple(mcfg.get("word_ngram", [1, 2])),
                                            sublinear_tf=True,
                                            token_pattern=WORD_PATTERN), col),
        ]
    return ColumnTransformer(blocks)


def build_estimator(kind: str, param: float, balanced: bool, mcfg: dict, seed: int = 42):
    cw = "balanced" if balanced else None
    if kind == "lr":
        est = LogisticRegression(C=param, max_iter=5000, class_weight=cw)
    elif kind == "svm":
        est = LinearSVC(C=param, class_weight=cw)
    elif kind == "ridge":
        est = RidgeClassifier(alpha=param, class_weight=cw)
    elif kind == "cnb":
        est = ComplementNB(alpha=param)          # has its own imbalance handling; ignores class_weight
    elif kind == "sgd":
        est = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=param,
                            l1_ratio=mcfg.get("l1_ratio", 0.15), max_iter=3000,
                            class_weight=cw, random_state=seed)
    else:
        raise ValueError(f"unknown model.clf {kind!r}; expected one of {sorted(DEFAULT_GRIDS)}")

    calibrate = mcfg.get("calibrate")            # null = off | true | false | "auto"
    if calibrate == "auto":
        calibrate = kind in NEEDS_CALIBRATION
    if calibrate:
        est = CalibratedClassifierCV(est, method="sigmoid", cv=5)
    return est


def build_tfidf(mcfg: dict, param: float, balanced: bool, seed: int = 42, columns=("text",)):
    return make_pipeline(build_features(mcfg, columns),
                         build_estimator(mcfg.get("clf", "lr"), param, balanced, mcfg, seed))


def _proba(pipe, texts):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(texts)
    d = pipe.decision_function(texts)
    if d.ndim == 1:
        d = np.stack([-d, d], 1)
    return softmax(d * 2.0, axis=1)      # pseudo-probabilities, only used for ensembling


def fit_and_score(cfg, train, val, test, n_labels, log=print):
    """Fit one model per grid value on the fit slice, keep the best held-out macro-F1."""
    mcfg = cfg["model"]
    kind = mcfg.get("clf", "lr")
    cw = mcfg.get("class_weight", "auto")
    balanced = (cfg["task"] == "b") if cw == "auto" else cw == "balanced"
    if kind == "cnb" and balanced:
        log("  note: ComplementNB has no class_weight; its own weighting handles the imbalance")
    grid = mcfg.get("param_grid") or mcfg.get("C_grid") or DEFAULT_GRIDS[kind]
    unit = "C" if kind in ("lr", "svm") else "alpha"
    cols = text_columns(cfg)
    if cols != ["text"]:
        log(f"  text_type={cfg['data']['text_type']} -> dac trung tu cot {cols}")
    fit, ev = split_rows(require_columns(train, cols, "train"))
    require_columns(val, cols, "val")
    if test is not None:
        require_columns(test, cols, "test")
    best = None
    for param in grid:
        pipe = build_tfidf(mcfg, param, balanced, cfg.get("seed", 42), cols).fit(fit[cols], fit.y)
        p_eval = _proba(pipe, ev[cols])
        m = compute_metrics(ev.y, p_eval.argmax(1))
        log(f"  {unit}={param}: macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m["macro_f1"], "C": param, "unit": unit,
                    "balanced": balanced, "eval": p_eval,
                    "val": _proba(pipe, val[cols]),
                    "test": None if test is None else _proba(pipe, test[cols])}
    return best
