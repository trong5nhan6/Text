"""B0 — TF-IDF (char + word n-grams) + a linear/NB classifier on the fixed train/eval split."""
import numpy as np
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.data.dataset import require_columns, split_rows, text_columns, use_valdataset
from src.evaluation.metrics import compute_metrics

# Regularisation grid per classifier. `lr`/`svm` take C (inverse strength, bigger = weaker),
# the rest take alpha (direct strength, bigger = stronger), so the useful ranges differ.
DEFAULT_GRIDS = {
    "lr":    [0.5, 1, 2, 4, 8, 16],
    "svm":   [0.05, 0.1, 0.25, 0.5, 1],
    "ridge": [0.1, 0.5, 1, 3, 10],
    "cnb":   [0.01, 0.05, 0.1, 0.3, 1],
    "sgd":   [1e-6, 1e-5, 1e-4],
    "mlp":   [1e-4, 1e-3, 1e-2],          # L2 penalty (alpha) of the MLP
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
    elif kind == "mlp":
        # A hidden layer over the n-grams: the linear models above can only add feature weights
        # up, this can learn that two n-grams mean something together. Early stopping on 10% of
        # the fit slice picks the epoch count; the held-out slice is never seen. No class_weight
        # in sklearn's MLP -- `balanced` goes in as sample weights instead (see fit_and_score).
        est = MLPClassifier(hidden_layer_sizes=tuple(mcfg.get("mlp_hidden") or [256]),
                            alpha=param, batch_size=mcfg.get("mlp_batch_size", 64),
                            learning_rate_init=mcfg.get("mlp_lr", 1e-3),
                            max_iter=mcfg.get("mlp_max_iter", 100), early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=mcfg.get("mlp_patience", 5),
                            random_state=seed)
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


def _mlp_sample_weight(y):
    """sklearn's MLP has no class_weight; `balanced` becomes n / (k * count[class]) per row."""
    counts = np.bincount(y)
    return (len(y) / (len(counts) * counts))[y]


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
    if cfg.get("data", {}).get("tta"):
        # Measured: feeding Kannada to a Latin-fitted model leaves 25k of 75M feature slots
        # non-zero, so it predicts one class for 634 of 639 rows -- at 0.93 confidence, because
        # with no features the intercept decides. Averaging that in only drags the real
        # prediction down (0.8215 -> 0.5988). The two scripts share no characters, so there is
        # no second view for a bag-of-n-grams to average over; use text_type=both instead,
        # which puts both scripts on the same row. TTA is for the transformers, whose encoder
        # is shared across scripts.
        raise SystemExit("data.tta khong dung duoc voi model.type=tfidf: dac trung char n-gram "
                         "cua hai he chu roi rac nhau. Dung data.text_type=both thay the.")
    if cols != ["text"]:
        log(f"  text_type={cfg['data']['text_type']} -> dac trung tu cot {cols}")
    fit, ev = split_rows(require_columns(train, cols, "train"), use_valdataset(cfg))
    if len(ev) == 0 and len(grid) > 1:
        # Every value in the grid is fitted and the best held-out score wins. With no held-out
        # slice there is nothing to compare, and silently taking the first value would hide the
        # fact that the regularisation strength was never chosen. Pin it explicitly, to whatever
        # the ordinary run already found and printed in results/metrics.csv.
        raise SystemExit(
            f"data.use_valdataset=false can mot gia tri {unit} duy nhat, nhung luoi dang co "
            f"{len(grid)}: {grid}. Xem {unit} tot nhat o results/metrics.csv roi ghim lai, vi du "
            f'--set "model.param_grid=[{grid[len(grid) // 2]}]"')
    require_columns(val, cols, "val")
    if test is not None:
        require_columns(test, cols, "test")

    best = None
    for param in grid:
        pipe = build_tfidf(mcfg, param, balanced, cfg.get("seed", 42), cols)
        fit_kw = {}
        if kind == "mlp" and balanced:
            # sample_weight reached MLPClassifier.fit only in recent sklearn; an older one would
            # raise TypeError here. Fall back to unweighted and say so.
            import inspect
            if "sample_weight" in inspect.signature(MLPClassifier.fit).parameters:
                fit_kw["mlpclassifier__sample_weight"] = _mlp_sample_weight(fit.y.to_numpy())
            elif param == grid[0]:
                log("  note: sklearn nay khong co sample_weight cho MLP -> train khong can bang lop")
        pipe.fit(fit[cols], fit.y, **fit_kw)
        p_eval = None if len(ev) == 0 else _proba(pipe, ev[cols])
        m = {"macro_f1": float("nan"), "accuracy": float("nan")} if p_eval is None else             compute_metrics(ev.y, p_eval.argmax(1))
        if p_eval is None:
            log(f"  {unit}={param}: fit tren 100% du lieu ({len(fit)} dong), khong cham diem")
        else:
            log(f"  {unit}={param}: macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")
        if best is None or (p_eval is not None and m["macro_f1"] > best["macro_f1"]):
            best = {"macro_f1": m["macro_f1"], "C": param, "unit": unit,
                    "balanced": balanced, "eval": p_eval,
                    "val": _proba(pipe, val[cols]),
                    "test": None if test is None else _proba(pipe, test[cols])}
    return best
