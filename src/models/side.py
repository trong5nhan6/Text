"""Side embeddings mixed into the encoder's input embeddings (model.side_embedding).

One encoder, two views of every word at its input. The encoder's own subword embedding is what
MuRIL already knows; next to it goes a word-level vector built from things the subword table
cannot see:

  char      a CharCNN over the word's characters. Emoji are characters, so 😡 -- which MuRIL and
            mBERT turn into [UNK] -- arrives intact; so do the spelling variants of Kanglish,
            which share most of their characters.
  phonetic  an embedding of the word's phonetic key (src.models.hybrid.phonetic_key), so
            thu/thuu/thoo land on one row. Keys seen fewer than side_min_count times in the fit
            slice share the UNK row -- 70% of the vocabulary occurs once, and a row trained on
            one example would only memorise it.

The word vector is projected to the hidden size and copied onto every subword of that word, via
the tokenizer's own word_ids(), so the two streams line up token for token whatever the
tokenizer is. They are mixed as  e = e_subword + gate * e_side  with the gate at zero: training
starts from the unmodified pretrained model, and the side stream only gets weight if it helps.
(CharBERT, Ma et al. 2020, and CharacterBERT, El Boukkouri et al. 2020, add character
information at the same place.)
"""
import pickle
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from src.models.hybrid import phonetic_key

KINDS = {"char": ("char",), "phonetic": ("phonetic",), "char+phonetic": ("char", "phonetic")}
PAD, UNK = 0, 1


def _words(enc, i: int, text: str):
    """The tokenizer's own words for example i, as strings, in word_ids() order."""
    ids = [w for w in enc.word_ids(i) if w is not None]
    n = max(ids) + 1 if ids else 0
    out = []
    for w in range(n):
        span = enc.word_to_chars(i, w)
        out.append("" if span is None else text[span.start:span.end])
    return out


class SideVocab:
    """Character and phonetic-key vocabularies, fitted on the fit slice. Also turns a tokenised
    batch into the tensors SideEmbedding reads, which is why the Collator holds it."""

    def __init__(self, kind: str, min_count: int = 2, max_word_len: int = 20):
        if kind not in KINDS:
            raise ValueError(f"model.side_embedding={kind!r}; expected null or one of {sorted(KINDS)}")
        self.kind, self.parts = kind, KINDS[kind]
        self.min_count, self.max_word_len = min_count, max_word_len
        self.chars, self.keys = {}, {}

    def fit(self, texts, tokenizer, max_len: int):
        cc, kc = Counter(), Counter()
        texts = list(texts)
        for s in range(0, len(texts), 256):
            chunk = texts[s:s + 256]
            enc = tokenizer(chunk, truncation=True, max_length=max_len)
            for i, t in enumerate(chunk):
                for w in _words(enc, i, t):
                    cc.update(w[:self.max_word_len])
                    kc[phonetic_key(w)] += 1
        # characters: every one seen (there are few, and an unseen emoji should still get a row
        # of its own next time only if it was in the fit slice); keys: min_count, see docstring
        self.chars = {c: i + 2 for i, c in enumerate(sorted(cc))}
        self.keys = {k: i + 2 for i, k in enumerate(sorted(k for k, n in kc.items() if n >= self.min_count))}
        return self

    @property
    def n_chars(self):
        return len(self.chars) + 2

    @property
    def n_keys(self):
        return len(self.keys) + 2

    def encode(self, enc, texts):
        """-> dict of tensors: side_tok2word [B,T] (-1 = no word), side_chars [B,W,L], side_keys [B,W]."""
        B, T = enc["input_ids"].shape
        words = [_words(enc, i, t) for i, t in enumerate(texts)]
        W = max(1, max(len(w) for w in words))
        L = max(1, min(self.max_word_len, max((len(x) for ws in words for x in ws), default=1)))
        tok2word = torch.full((B, T), -1, dtype=torch.long)
        chars = torch.zeros((B, W, L), dtype=torch.long)
        keys = torch.zeros((B, W), dtype=torch.long)
        for i, ws in enumerate(words):
            for t, w in enumerate(enc.word_ids(i)):
                if w is not None:
                    tok2word[i, t] = w
            for j, w in enumerate(ws):
                ids = [self.chars.get(c, UNK) for c in w[:L]]
                if ids:
                    chars[i, j, :len(ids)] = torch.tensor(ids)
                keys[i, j] = self.keys.get(phonetic_key(w), UNK)
        return {"side_tok2word": tok2word, "side_chars": chars, "side_keys": keys}

    def save(self, path):
        with open(Path(path), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(Path(path), "rb") as f:
            return pickle.load(f)


class SideEmbedding(nn.Module):
    """Word-level side vector, spread onto the subwords and returned as [B, T, hidden]."""

    def __init__(self, cfg: dict, hidden: int):
        super().__init__()
        parts = KINDS[cfg["kind"]]
        width = 0
        if "char" in parts:
            d, f = cfg.get("char_dim", 32), cfg.get("char_filters", 64)
            self.char_table = nn.Embedding(cfg["n_chars"], d, padding_idx=PAD)
            # widths 2-5: character n-grams, the same range the TF-IDF branch found useful
            self.convs = nn.ModuleList(nn.Conv1d(d, f, k, padding=k // 2) for k in (2, 3, 4, 5))
            width += 4 * f
        if "phonetic" in parts:
            self.key_table = nn.Embedding(cfg["n_keys"], cfg.get("key_dim", 128), padding_idx=PAD)
            width += cfg.get("key_dim", 128)
        self.proj = nn.Sequential(nn.Dropout(cfg.get("dropout", 0.1)), nn.Linear(width, hidden))

    def forward(self, side_tok2word, side_chars=None, side_keys=None):
        feats = []
        if hasattr(self, "char_table"):
            B, W, L = side_chars.shape
            x = self.char_table(side_chars.view(B * W, L)).transpose(1, 2)        # [BW, d, L]
            pooled = [torch.relu(c(x)).max(-1).values for c in self.convs]        # [BW, f] x4
            feats.append(torch.cat(pooled, -1).view(B, W, -1))
        if hasattr(self, "key_table"):
            feats.append(self.key_table(side_keys))
        word = self.proj(torch.cat(feats, -1))                                     # [B, W, hidden]
        idx = side_tok2word.clamp(min=0)
        tok = torch.gather(word, 1, idx.unsqueeze(-1).expand(-1, -1, word.size(-1)))
        return tok * (side_tok2word >= 0).unsqueeze(-1).to(tok.dtype)             # special/pad -> 0
