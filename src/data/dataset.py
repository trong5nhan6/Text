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


def split_rows(train):
    """-> (fit_df, eval_df). Every run uses the same split, so eval_df is identical
    across runs and their saved predictions line up row for row."""
    return train[train.is_val == 0], train[train.is_val == 1]


TEXT_COLUMNS = {"latin": ["text"], "kn": ["text_kn"], "both": ["text", "text_kn"]}


def text_columns(cfg):
    """-> the column names data.text_type asks for. null means latin."""
    tt = cfg.get("data", {}).get("text_type") or "latin"
    if tt not in TEXT_COLUMNS:
        raise ValueError(f"data.text_type={tt!r}; expected one of {sorted(TEXT_COLUMNS)}")
    return TEXT_COLUMNS[tt]


def infer_columns(cfg):
    """Which views to average over at prediction time. data.tta uses both scripts regardless of
    what the model trained on; without it, prediction uses the training view."""
    cols = text_columns(cfg)
    if not cfg.get("data", {}).get("tta"):
        return ["text"] if cols == ["text", "text_kn"] else cols
    return ["text", "text_kn"]


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
    def __init__(self, tokenizer, max_len: int):
        self.tok, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        enc = self.tok([b[0] for b in batch], truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        if batch[0][1] is not None:
            enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return enc


def make_loader(texts, labels, tokenizer, cfg, train: bool):
    t = cfg["training"]
    return DataLoader(
        TextDataset(texts, labels),
        batch_size=t["batch_size"] if train else t["eval_batch_size"],
        shuffle=train,
        collate_fn=Collator(tokenizer, cfg["data"]["max_len"]),
        num_workers=t.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
    )
