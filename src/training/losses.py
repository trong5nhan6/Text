import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(counts, mode: str):
    counts = np.asarray(counts, dtype=float)
    if mode in (None, "none"):
        return None
    w = counts.sum() / (len(counts) * counts)
    if mode == "sqrt_inv":
        w = np.sqrt(w)
    elif mode != "inverse":
        raise ValueError(f"unknown class_weight mode {mode}")
    return torch.tensor(w, dtype=torch.float)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma, self.ls = gamma, label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none",
                             label_smoothing=self.ls)
        # pt is the true-class probability, so it is read off the unsmoothed, unweighted CE
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(name: str, tcfg: dict, class_counts) -> nn.Module:
    """name: ce | wce | focal (already resolved from 'auto')."""
    ls = tcfg.get("label_smoothing", 0.0)
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    w = class_weights(class_counts, tcfg.get("class_weight", "sqrt_inv"))
    if name == "wce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    if name == "focal":
        return FocalLoss(tcfg.get("focal_gamma", 2.0), w, ls)
    raise ValueError(f"unknown loss {name}")
