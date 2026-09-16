import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class TransformerClassifier(nn.Module):
    """Pretrained encoder + pooling + dropout + linear head."""

    def __init__(self, backbone_name: str, num_labels: int, pooling: str = "cls",
                 dropout: float = 0.1, backbone_config=None):
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
        self.meta = {"backbone_name": backbone_name, "num_labels": num_labels,
                     "pooling": pooling, "dropout": dropout}
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.config.hidden_size, num_labels)
        nn.init.normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **_):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        h = self.backbone(**kw).last_hidden_state
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            pooled = h[:, 0]
        return self.head(self.dropout(pooled))

    def param_groups(self, lr, head_lr, weight_decay):
        no_decay = ("bias", "LayerNorm.weight", "layer_norm", "layernorm", "norm.weight")
        groups = {}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head = n.startswith("head.")
            wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
            key = (is_head, wd)
            groups.setdefault(key, {"params": [], "weight_decay": wd, "lr": (head_lr or lr) if is_head else lr})
            groups[key]["params"].append(p)
        return list(groups.values())

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
        model = cls(meta["backbone_name"], meta["num_labels"], meta["pooling"], meta["dropout"],
                    backbone_config=bcfg)
        sd = torch.load(ckpt_dir / "model.pt", map_location=map_location)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        return model, meta
