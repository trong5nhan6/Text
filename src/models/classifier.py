import json
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.models.heads import build_head

_BLOCK_RE = re.compile(r"^(?P<prefix>.+?)\.(?P<idx>\d+)\.")


def block_layout(backbone):
    """-> (container prefix, number of blocks) for the repeated transformer block.

    Found from the parameter names rather than hard-coded, because the families differ:
    BERT / XLM-R / IndicBERT / mDeBERTa name it `encoder.layer.{i}` with 12 blocks, while
    ModernBERT names it `layers.{i}` and has 22.
    """
    names = [n for n, _ in backbone.named_parameters()]
    prefixes = Counter(m.group("prefix") for n in names if (m := _BLOCK_RE.match(n)))
    if not prefixes:
        return None, 0
    prefix = prefixes.most_common(1)[0][0]
    idx = {int(m.group("idx")) for n in names
           if (m := _BLOCK_RE.match(n)) and m.group("prefix") == prefix}
    return prefix, max(idx) + 1


class TransformerClassifier(nn.Module):
    """Pretrained encoder + pooling + dropout + linear head."""

    def __init__(self, backbone_name: str, num_labels: int, pooling: str = "cls",
                 dropout: float = 0.1, backbone_config=None,
                 unfreeze_last_n_blocks=None, freeze_embeddings=None, head_cfg=None):
        super().__init__()
        if backbone_config is None:                      # training: load pretrained weights
            # .float() is not redundant: some published checkpoints store fp16 weights
            # (mDeBERTa-v3 does) and transformers keeps the checkpoint's dtype, which then
            # meets the fp32 head as "mat1 and mat2 must have the same dtype". AMP wants fp32
            # master weights anyway -- mixed precision is applied by autocast, not by the weights.
            self.backbone = AutoModel.from_pretrained(backbone_name).float()
        else:                                            # inference: architecture only, weights come from ckpt
            # .float() for the same reason, from the other direction: a config saved off an fp16
            # backbone builds an fp16 one, and load_state_dict copies in place, so fp16 would stick.
            self.backbone = AutoModel.from_config(backbone_config).float()
        head_cfg = dict(head_cfg or {"head": "linear"})
        self.meta = {"backbone_name": backbone_name, "num_labels": num_labels,
                     "pooling": pooling, "dropout": dropout, "head_cfg": head_cfg}
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        # Kept under the name `head` on purpose: param_groups() routes head.* to head_lr and
        # excludes it from the LLRD ladder by that prefix, and both must keep holding.
        self.head = build_head(head_cfg.get("head"), self.backbone.config.hidden_size,
                               num_labels, head_cfg)
        self.freeze_summary = self._apply_freezing(unfreeze_last_n_blocks, freeze_embeddings)

    def _apply_freezing(self, unfreeze_last_n, freeze_emb) -> str:
        """Freeze everything below the last `unfreeze_last_n` blocks. The head and the modules
        that sit *above* the last block (mDeBERTa's encoder.LayerNorm, ModernBERT's final_norm)
        always stay trainable -- freezing those would cut the path the gradient needs.
        `freeze_emb=None` means: follow the block setting."""
        if freeze_emb is None:
            freeze_emb = unfreeze_last_n is not None
        prefix, n_blocks = block_layout(self.backbone)
        total = sum(p.numel() for p in self.parameters())

        if unfreeze_last_n is not None:
            if prefix is None:
                raise ValueError(f"no transformer blocks found in {self.meta['backbone_name']}; "
                                 f"unfreeze_last_n_blocks cannot be applied")
            if not 0 <= unfreeze_last_n <= n_blocks:
                raise ValueError(f"unfreeze_last_n_blocks={unfreeze_last_n} out of range for "
                                 f"{self.meta['backbone_name']} ({n_blocks} blocks)")
        first_trainable = None if unfreeze_last_n is None else n_blocks - unfreeze_last_n

        n_pooler = 0
        for name, p in self.backbone.named_parameters():
            if name.startswith("pooler."):
                # BERT-family only, and dead weight here: forward() reads last_hidden_state, never
                # pooler_output, so these never receive a gradient. Freeze them so they stay out of
                # the optimizer and out of the trainable count.
                p.requires_grad_(False)
                n_pooler += p.numel()
                continue
            if "embed" in name.lower():
                p.requires_grad_(not freeze_emb)
                continue
            if first_trainable is None:
                continue                                  # blocks untouched: train everything
            m = _BLOCK_RE.match(name)
            if m and m.group("prefix") == prefix:
                p.requires_grad_(int(m.group("idx")) >= first_trainable)
            # anything else is the tail above the blocks -> left trainable

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parts = []
        if first_trainable is None:
            parts.append(f"all {n_blocks} blocks trainable" if n_blocks
                         else "block structure not detected; nothing frozen")
        elif unfreeze_last_n == 0:
            parts.append(f"all {n_blocks} blocks frozen (head only)")
        else:
            parts.append(f"blocks 0-{first_trainable - 1} frozen, "
                         f"{unfreeze_last_n}/{n_blocks} trainable")
        parts.append("embeddings " + ("frozen" if freeze_emb else "trainable"))
        parts.append(f"{trainable / 1e6:.1f}M / {total / 1e6:.1f}M params "
                     f"({100 * trainable / total:.1f}%)")
        return " | ".join(parts)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **_):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        h = self.backbone(**kw).last_hidden_state
        # soft_moe aggregates the token axis itself, so it takes the sequence and `pooling` is
        # unused. Every other head takes the pooled vector, with dropout applied here so that
        # `linear` stays exactly what it was.
        if getattr(self.head, "needs_tokens", False):
            return self.head(h, attention_mask)
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            pooled = h[:, 0]
        return self.head(self.dropout(pooled))

    @property
    def aux_loss(self):
        """Router load-balancing term of a sparse MoE head; 0.0 for every other head, which the
        trainer can add unconditionally."""
        return getattr(self.head, "aux_loss", 0.0)

    def param_groups(self, lr, head_lr, weight_decay, llrd=None):
        """Optimizer groups. Two schemes, chosen by `training.llrd`:

        llrd=None -- one flat `lr` for the whole backbone (plus `head_lr` for the head).
        llrd=0.9  -- layer-wise LR decay: every block gets 0.9x the LR of the block above it,
            so the last block trains at `lr`, block 0 at lr*0.9**n_blocks, and the embeddings
            one notch below that. The lower blocks are what makes a multilingual checkpoint
            worth starting from -- on 3k-6k rows a flat 2e-5 rewrites them faster than the task
            signal can justify, and the encoder ends up no better than char n-grams.

        The head is excluded from the decay in both cases: it is randomly initialised, so there
        is nothing in it to preserve, and it keeps `head_lr`.

        Frozen parameters are skipped, so this composes with unfreeze_last_n_blocks: the decay
        is still indexed by absolute block number, and whatever is frozen simply never appears.
        """
        no_decay = ("bias", "LayerNorm.weight", "layer_norm", "layernorm", "norm.weight")
        prefix, n_blocks = block_layout(self.backbone)
        prefix = None if prefix is None else f"backbone.{prefix}"

        def depth(name):
            """0 = embeddings, 1..n_blocks = blocks bottom-up, n_blocks+1 = the tail above them."""
            if "embed" in name.lower():
                return 0
            m = _BLOCK_RE.match(name) if prefix else None
            if m and m.group("prefix") == prefix:
                return int(m.group("idx")) + 1
            return n_blocks + 1

        trainable = [(n, p) for n, p in self.named_parameters()
                     if p.requires_grad and not n.startswith("head.")]
        # The exponent is measured from the topmost depth that actually has parameters, not from
        # n_blocks+1: BertModel has nothing above its last block (its pooler is frozen here), so a
        # fixed ceiling would leave that rung empty and shift the whole ladder down one notch --
        # the last block would train at 0.9*lr instead of lr. mDeBERTa, which does keep an
        # encoder.LayerNorm up there, gives that module lr and the last block 0.9*lr.
        top = max((depth(n) for n, _ in trainable), default=0)

        groups = {}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
            if n.startswith("head."):
                key, p_lr = ("head", wd), head_lr or lr
            elif llrd:
                d = depth(n)
                key, p_lr = (d, wd), lr * llrd ** (top - d)
            else:
                key, p_lr = ("backbone", wd), lr
            groups.setdefault(key, {"params": [], "weight_decay": wd, "lr": p_lr})
            groups[key]["params"].append(p)
        return [groups[k] for k in sorted(groups, key=lambda k: (str(k[0]), k[1]))]

    # ---------- checkpoint I/O ----------
    def save(self, out_dir, tokenizer=None, half=True, extra=None):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: (v.half() if half and v.is_floating_point() else v).cpu() for k, v in self.state_dict().items()}
        torch.save(sd, out_dir / "model.pt")
        self.backbone.config.save_pretrained(out_dir / "backbone")
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir / "tokenizer")
        json.dump({**self.meta, **(extra or {})}, open(out_dir / "meta.json", "w"), indent=1)

    @classmethod
    def load(cls, ckpt_dir, map_location="cpu"):
        ckpt_dir = Path(ckpt_dir)
        meta = json.load(open(ckpt_dir / "meta.json"))
        bcfg = AutoConfig.from_pretrained(ckpt_dir / "backbone")
        # .get: checkpoints written before model.head existed carry no head_cfg and are linear.
        model = cls(meta["backbone_name"], meta["num_labels"], meta["pooling"], meta["dropout"],
                    backbone_config=bcfg, head_cfg=meta.get("head_cfg"))
        sd = torch.load(ckpt_dir / "model.pt", map_location=map_location)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        return model, meta
