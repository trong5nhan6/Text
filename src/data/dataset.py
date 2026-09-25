from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import TASKS


def load_split(cfg, split: str):
    """split: train | val | test. Returns None for a missing test file."""
    p = Path(cfg["paths"]["processed_dir"]) / f"{cfg['task']}_{split}.csv"
    if not p.exists():
        if split == "test":
            return None
        raise FileNotFoundError(p)
    return pd.read_csv(p, keep_default_na=False)


def label_names(task: str):
    return TASKS[task]["labels"]


def use_valdataset(cfg) -> bool:
    """data.use_valdataset: false trains on every labelled row instead of holding 10% back.

    null and true both mean "hold the slice back", which is the default and what every finished
    run did. Only an explicit false changes anything -- and it changes a lot, so read split_rows.
    """
    return cfg.get("data", {}).get("use_valdataset") is not False


def split_rows(train, use_val: bool = True):
    """-> (fit_df, eval_df). Every run uses the same split, so eval_df is identical
    across runs and their saved predictions line up row for row.

    use_val=False returns every row as the fit set and an empty eval set. That is the
    train-on-everything mode: it buys ~10% more training data and gives up, in exchange, the
    only labelled scoring set there is. Without it a run cannot pick its best epoch, cannot stop
    early, cannot be scored, and cannot join a blend -- so it is for the final submission model
    only, after the epoch count has been settled by an ordinary run.
    """
    if not use_val:
        return train, train.iloc[:0]
    return train[train.is_val == 0], train[train.is_val == 1]


# data.text_type -> which column(s) the model reads. "en" is short for English, matching "kn"
# for Kannada; "both" stays Latin+Kannada, the pair that shares a language.
TEXT_COLUMNS = {"latin": ["text"], "kn": ["text_kn"], "en": ["text_en"],
                "both": ["text", "text_kn"]}


def text_columns(cfg):
    """-> the column names data.text_type asks for. null means latin."""
    tt = cfg.get("data", {}).get("text_type") or "latin"
    if tt not in TEXT_COLUMNS:
        raise ValueError(f"data.text_type={tt!r}; expected one of {sorted(TEXT_COLUMNS)}")
    return TEXT_COLUMNS[tt]


def infer_columns(cfg):
    """Which views to average over at prediction time. Without data.tta, several training columns
    collapse to the Latin original and a single one is used as-is. With tta, the Latin original
    and the selected view are both predicted and averaged -- so text_type=latin makes tta a
    no-op, there being no second view; substituting text_kn there would be a different
    experiment, quietly run under the name of this one."""
    cols = text_columns(cfg)
    if not cfg.get("data", {}).get("tta"):
        return ["text"] if len(cols) > 1 else cols
    return list(dict.fromkeys(["text", *cols]))


def swap_views(df, cols):
    """The same rows seen through the other script, with the column names the fitted model
    expects: a model fitted on `text` is handed text_kn under the name `text`."""
    other = {"text": "text_kn", "text_kn": "text"}
    return df[[other[c] for c in cols]].rename(columns={other[c]: c for c in cols})


def require_columns(df, cols, what="data"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{what} thieu cot {missing}. Chay build_xlit_cache.ipynb de sinh "
                         f"data/xlit_kn.json roi `python -m src.data.preprocessing`.")
    return df


def eval_y(train):
    """Gold labels of the held-out slice, aligned with each run's eval.npy."""
    return train.y.to_numpy()[train.is_val.to_numpy() == 1]


class TextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], (None if self.labels is None else self.labels[i])


class Collator:
    def __init__(self, tokenizer, max_len: int, featurizer=None, side=None):
        self.tok, self.max_len, self.featurizer, self.side = tokenizer, max_len, featurizer, side

    def __call__(self, batch):
        texts = [b[0] for b in batch]
        enc = self.tok(texts, truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        if self.featurizer is not None:
            # model.hybrid: the TF-IDF vector of the same texts, built per batch so nothing
            # dataset-sized is ever densified. Rides along as the `tfidf` kwarg of forward().
            enc["tfidf"] = torch.from_numpy(self.featurizer.to_dense(texts))
        side = self.side.encode(enc, texts) if self.side is not None else {}
        if batch[0][1] is not None:
            enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        # after encode(): it reads word_ids() off `enc`, which only the tokenizer's own output has
        for k, v in side.items():
            enc[k] = v
        return enc


def make_loader(texts, labels, tokenizer, cfg, train: bool, featurizer=None, side=None):
    t = cfg["training"]
    return DataLoader(
        TextDataset(texts, labels),
        batch_size=t["batch_size"] if train else t["eval_batch_size"],
        shuffle=train,
        collate_fn=Collator(tokenizer, cfg["data"]["max_len"], featurizer, side),
        num_workers=t.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
    )
