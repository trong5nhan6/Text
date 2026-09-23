from pathlib import Path

import src.utils.hf_quiet    # noqa: F401  -- import first: silences HF's bars at import time
from transformers import AutoTokenizer

from src.models.classifier import TransformerClassifier
from src.models.heads import head_config


def build_tokenizer(cfg_or_name):
    name = cfg_or_name if isinstance(cfg_or_name, str) else cfg_or_name["model"]["name"]
    tok = AutoTokenizer.from_pretrained(name)
    # Decoder LLMs usually ship without a pad token, and the Collator pads every batch. Reusing
    # eos is the standard choice: attention_mask already tells the model to ignore those
    # positions, so the id itself never matters.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding, so "the last real token" is attention_mask.sum()-1 -- which is what
    # pooling="last" indexes. A left-padded batch would put padding there instead.
    tok.padding_side = "right"
    return tok


def build_model(cfg, num_labels: int):
    m = cfg["model"]
    if m["type"] != "transformer":
        raise ValueError(f"build_model only handles transformers, got {m['type']} (tfidf -> src.models.tfidf)")
    return TransformerClassifier(m["name"], num_labels, m.get("pooling", "cls"), m.get("dropout", 0.1),
                                 unfreeze_last_n_blocks=m.get("unfreeze_last_n_blocks"),
                                 freeze_embeddings=m.get("freeze_embeddings"),
                                 head_cfg=head_config(m), layers=m.get("layers"),
                                 load_in_4bit=m.get("load_in_4bit"), lora=m.get("lora"))


def load_from_checkpoint(ckpt_dir):
    """-> (model, tokenizer, meta) from a folder written by TransformerClassifier.save()."""
    ckpt_dir = Path(ckpt_dir)
    model, meta = TransformerClassifier.load(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")
    return model, tok, meta
