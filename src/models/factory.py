from pathlib import Path

from transformers import AutoTokenizer

from src.models.classifier import TransformerClassifier


def build_tokenizer(cfg_or_name):
    name = cfg_or_name if isinstance(cfg_or_name, str) else cfg_or_name["model"]["name"]
    return AutoTokenizer.from_pretrained(name)


def build_model(cfg, num_labels: int):
    m = cfg["model"]
    if m["type"] != "transformer":
        raise ValueError(f"build_model only handles transformers, got {m['type']} (tfidf -> src.models.tfidf)")
    return TransformerClassifier(m["name"], num_labels, m.get("pooling", "cls"), m.get("dropout", 0.1))


def load_from_checkpoint(ckpt_dir):
    """-> (model, tokenizer, meta) from a folder written by TransformerClassifier.save()."""
    ckpt_dir = Path(ckpt_dir)
    model, meta = TransformerClassifier.load(ckpt_dir)
    tok = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")
    return model, tok, meta
