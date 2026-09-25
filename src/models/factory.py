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


def build_featurizer(cfg, texts):
    """model.hybrid=tfidf -> a TfidfFeaturizer fitted on `texts` (the fit slice); else None."""
    m = cfg["model"]
    kind = m.get("hybrid")
    if not kind:
        return None
    if kind != "tfidf":
        raise ValueError(f"model.hybrid={kind!r}; expected null or tfidf")
    from src.models.hybrid import TfidfFeaturizer
    return TfidfFeaturizer(m.get("hybrid_max_features", 20000),
                           m.get("hybrid_phonetic", True)).fit(texts)


def build_side_vocab(cfg, texts, tokenizer):
    """model.side_embedding -> a SideVocab fitted on `texts` (the fit slice); else None."""
    m = cfg["model"]
    kind = m.get("side_embedding")
    if not kind:
        return None
    if not getattr(tokenizer, "is_fast", False):
        # word_ids() / word_to_chars() are what line the side vectors up with the subwords, and
        # only a fast tokenizer has them. CANINE's is not one (and has no subwords to align).
        raise SystemExit(f"model.side_embedding can tokenizer 'fast' (co word_ids); "
                         f"{m['name']} khong co.")
    from src.models.side import SideVocab
    return SideVocab(kind, m.get("side_min_count", 2), m.get("side_max_word_len", 20)).fit(
        texts, tokenizer, cfg["data"]["max_len"])


def build_model(cfg, num_labels: int, featurizer=None, side_vocab=None):
    m = cfg["model"]
    if m["type"] != "transformer":
        raise ValueError(f"build_model only handles transformers, got {m['type']} (tfidf -> src.models.tfidf)")
    hybrid = None if featurizer is None else {
        "in_dim": featurizer.dim, "dim": m.get("hybrid_dim", 256),
        "dropout": m.get("hybrid_dropout", 0.3), "lr": m.get("hybrid_lr", 1e-3)}
    side = None if side_vocab is None else {
        "kind": side_vocab.kind, "n_chars": side_vocab.n_chars, "n_keys": side_vocab.n_keys,
        "char_dim": m.get("side_char_dim", 32), "char_filters": m.get("side_char_filters", 64),
        "key_dim": m.get("side_key_dim", 128), "dropout": m.get("side_dropout", 0.1),
        "lr": m.get("side_lr", 1e-3)}
    model = TransformerClassifier(m["name"], num_labels, m.get("pooling", "cls"), m.get("dropout", 0.1),
                                  unfreeze_last_n_blocks=m.get("unfreeze_last_n_blocks"),
                                  freeze_embeddings=m.get("freeze_embeddings"),
                                  head_cfg=head_config(m), layers=m.get("layers"),
                                  load_in_4bit=m.get("load_in_4bit"), lora=m.get("lora"),
                                  dtype=m.get("dtype"),
                                  multisample_dropout=m.get("multisample_dropout"), hybrid=hybrid,
                                  side=side)
    model.featurizer = featurizer
    model.side_vocab = side_vocab
    return model


def load_from_checkpoint(ckpt_dir):
    """-> (model, tokenizer, meta) from a folder written by TransformerClassifier.save()."""
    ckpt_dir = Path(ckpt_dir)
    model, meta = TransformerClassifier.load(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")
    return model, tok, meta
