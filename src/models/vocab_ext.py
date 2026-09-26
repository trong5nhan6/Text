"""Vocabulary extension for a WordPiece tokenizer (pretrain_mlm.py --extend_vocab).

Kanglish is cut into many pieces by MuRIL's vocabulary (maklu -> ma ##k ##lu). This adds the
frequent, badly fragmented words of the corpus to the tokenizer as whole tokens, grows the
embedding table to match, and starts every new row at the mean of the pieces it replaces
(Fast Vocabulary Transfer, Gee et al. 2022) -- so a new token begins where the model's reading
of its pieces already was, not at random. MLM then trains the new rows.

The words go into the WordPiece vocabulary itself, not through tokenizer.add_tokens(): an added
token is matched as a raw string anywhere in the text, inside other words too, while a vocabulary
entry takes part in WordPiece's longest-match -- so 'makluge' becomes 'maklu ##ge' as well.
"""
import re
from collections import Counter


def _words(tok, text):
    """The tokenizer's own words: its normaliser (lowercasing, for cnerg_muril) then its
    pre-tokeniser (whitespace and punctuation splits)."""
    bt = tok.backend_tokenizer
    norm = bt.normalizer.normalize_str(text) if bt.normalizer is not None else text
    return [w for w, _ in bt.pre_tokenizer.pre_tokenize_str(norm)]


def find_new_words(texts, tok, min_count=10, min_pieces=3, max_words=0):
    """-> (list of (word, count, pieces)), stats dict. Alphabetic words only: punctuation,
    numbers and emoji stay as they are."""
    cnt = Counter(w for t in texts for w in _words(tok, t) if re.fullmatch(r"[^\W\d_]{2,}", w))
    vocab = tok.get_vocab()
    pieces = {w: tok.tokenize(w) for w in cnt}
    cand = [(w, n, pieces[w]) for w, n in cnt.items()
            if n >= min_count and len(pieces[w]) >= min_pieces and w not in vocab]
    cand.sort(key=lambda x: (-x[1], x[0]))
    if max_words:
        cand = cand[:max_words]
    total_words = sum(cnt.values())
    total_pieces = sum(n * len(pieces[w]) for w, n in cnt.items())
    saved = sum(n * (len(p) - 1) for _, n, p in cand)
    stats = {"distinct_words": len(cnt), "word_occurrences": total_words,
             "pieces_per_word_before": total_pieces / max(total_words, 1),
             "pieces_per_word_after": (total_pieces - saved) / max(total_words, 1),
             "new_words": len(cand),
             "mean_count_of_new": sum(n for _, n, _ in cand) / max(len(cand), 1)}
    return cand, stats


def extend(tok, model, words, log=print):
    """Add `words` (strings) to tok's WordPiece vocabulary and grow `model` (a *ForMaskedLM) to
    match, each new embedding row = mean of the rows of the pieces it replaces; the MLM output
    bias the same way. -> (first new id, number added). Mutates both in place."""
    import torch
    from tokenizers.models import WordPiece

    wp = tok.backend_tokenizer.model
    if not isinstance(wp, WordPiece):
        raise SystemExit(f"--extend_vocab chi ho tro tokenizer WordPiece (MuRIL, mBERT); "
                         f"tokenizer nay la {type(wp).__name__}")
    old_pieces = {w: tok(w, add_special_tokens=False)["input_ids"] for w in words}
    vocab = dict(tok.backend_tokenizer.get_vocab(with_added_tokens=False))
    # MuRIL's ids have holes (197,258 entries up to id 197,284), so new ids start after the
    # embedding table, not after len(vocab).
    first = max(model.get_input_embeddings().weight.shape[0], max(vocab.values()) + 1)
    new = [w for w in words if w not in vocab]
    for i, w in enumerate(new):
        vocab[w] = first + i
    tok.backend_tokenizer.model = WordPiece(vocab=vocab, unk_token=wp.unk_token,
                                            continuing_subword_prefix=wp.continuing_subword_prefix,
                                            max_input_chars_per_word=wp.max_input_chars_per_word)
    model.resize_token_embeddings(first + len(new))
    emb = model.get_input_embeddings().weight
    bias = model.cls.predictions.bias if hasattr(model, "cls") else None
    with torch.no_grad():
        for i, w in enumerate(new):
            ids = torch.tensor(old_pieces[w])
            emb[first + i] = emb[ids].mean(0)
            if bias is not None:
                bias[first + i] = bias[ids].mean()
    # the tokenizer must now produce the new ids -- check on the words themselves
    bad = [w for w in new[:50] if tok(w, add_special_tokens=False)["input_ids"] != [vocab[w]]]
    if bad:
        raise SystemExit(f"mo rong vocab loi: {bad[:5]} khong thanh 1 token")
    log(f"mo rong vocab: +{len(new)} tu (id {first}..{first + len(new) - 1}), "
        f"embedding {tuple(emb.shape)}, khoi tao = trung binh cac manh cu")
    return first, len(new)
