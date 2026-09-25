"""TF-IDF side branch for the hybrid transformer (model.hybrid: tfidf).

The encoder and the n-gram model fail on different rows -- on the task-a held-out slice the
TF-IDF trio and the transformer trio are each wrong on ~100 rows but on only 68 of the same ones.
Averaging their probabilities (evaluate.py) uses that only at the output. Here the sparse vector
goes into the classifier itself, next to the pooled encoder state, so the head can learn *when*
to trust the spelling-level evidence (slurs, their variants) and when the context.

The featurizer is fitted on the fit slice of the run and nothing else: the eval rows' n-grams
must not shape the vocabulary any more than their labels may shape the weights.
"""
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

WORD_PATTERN = r"(?u)\b\w+\b|[\U0001F000-\U0001FAFF]"     # same tokens as src/models/tfidf.py

_PHON_RULES = [("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("ou", "u"),
               ("w", "v"), ("z", "j"), ("ph", "f"), ("q", "k"), ("ck", "k"), ("x", "ks"),
               ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"),
               ("jh", "j")]


def phonetic_key(t: str) -> str:
    """Collapse the spelling variants of romanised Kannada onto one key: `soole`/`sule`,
    `maadi`/`madi`, `thumba`/`tumba`. 70% of the training vocabulary occurs once, much of it
    being the same word spelt another way. As an extra TF-IDF view this measured +0.002 macro-F1
    on task a and +0.013 on task b (5-fold CV, LR on char+word n-grams)."""
    t = re.sub(r"[^\w\s\U0001F000-\U0001FAFF]", " ", t.lower())
    for a, b in _PHON_RULES:
        t = t.replace(a, b)
    t = re.sub(r"(\w)\1+", r"\1", t)              # doubled letters: 'sulle' -> 'sule'
    return re.sub(r"(\w\w\w)[aeiou]\b", r"\1", t)  # final vowel: 'maadi'/'madi' -> 'mad'


class TfidfFeaturizer:
    """char_wb + word n-grams of the text, and optionally of its phonetic key. Callable on a list
    of strings -> CSR matrix, which is how the Collator uses it per batch."""

    def __init__(self, max_features=20000, phonetic=True):
        self.max_features, self.phonetic = max_features, phonetic
        self.vecs = []

    def _views(self, texts):
        texts = list(texts)
        return [texts, [phonetic_key(t) for t in texts]] if self.phonetic else [texts]

    def fit(self, texts):
        self.vecs = []
        for view in self._views(texts):
            for v in (TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                      sublinear_tf=True, max_features=self.max_features),
                      TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                      sublinear_tf=True, token_pattern=WORD_PATTERN,
                                      max_features=self.max_features)):
                self.vecs.append(v.fit(view))
        return self

    @property
    def dim(self) -> int:
        return sum(len(v.vocabulary_) for v in self.vecs)

    def __call__(self, texts):
        views = self._views(texts)
        per_view = len(self.vecs) // len(views)
        return hstack([v.transform(views[i // per_view]) for i, v in enumerate(self.vecs)]).tocsr()

    def to_dense(self, texts) -> np.ndarray:
        return self(texts).toarray().astype(np.float32)

    def save(self, path):
        with open(Path(path), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(Path(path), "rb") as f:
            return pickle.load(f)
