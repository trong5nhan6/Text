import os
from pathlib import Path

# Hugging Face draws tqdm bars while downloading and while materialising weights. Outside a TTY
# -- a notebook cell, a redirected log -- tqdm cannot rewrite its line, so one bar becomes
# hundreds of "Loading weights: 3%|..." lines that bury our own output. Must run before
# transformers (and through it huggingface_hub) reads these at import time.
# Set HASTIKA_HF_VERBOSE=1 to get the bars back.
_QUIET = not os.environ.get("HASTIKA_HF_VERBOSE")
if _QUIET:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows-only noise

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

if _QUIET:
    hf_logging.disable_progress_bar()       # the "Loading weights" bar; verbosity is left alone

from src.models.classifier import TransformerClassifier
from src.models.heads import head_config


def build_tokenizer(cfg_or_name):
    name = cfg_or_name if isinstance(cfg_or_name, str) else cfg_or_name["model"]["name"]
    return AutoTokenizer.from_pretrained(name)


def build_model(cfg, num_labels: int):
    m = cfg["model"]
    if m["type"] != "transformer":
        raise ValueError(f"build_model only handles transformers, got {m['type']} (tfidf -> src.models.tfidf)")
    return TransformerClassifier(m["name"], num_labels, m.get("pooling", "cls"), m.get("dropout", 0.1),
                                 unfreeze_last_n_blocks=m.get("unfreeze_last_n_blocks"),
                                 freeze_embeddings=m.get("freeze_embeddings"),
                                 head_cfg=head_config(m))


def load_from_checkpoint(ckpt_dir):
    """-> (model, tokenizer, meta) from a folder written by TransformerClassifier.save()."""
    ckpt_dir = Path(ckpt_dir)
    model, meta = TransformerClassifier.load(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")
    return model, tok, meta
