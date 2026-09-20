"""Classification heads: plain linear, sparse MoE (top-k routed), soft MoE (token-level).

Chosen by `model.head`. The default `linear` is what every finished run used, and it stays
byte-for-byte what it was -- the other two are additions, not replacements.

Sizes, for the record. The linear head on MuRIL is 768*6+6 = 4,614 parameters; a 4-expert MoE
with expert_dim 64 is ~201k, 44x larger, fitted on 2,827 rows (task b) or 5,747 (task a). That
ratio is the whole risk: a head is not where a fine-tuned encoder's capacity lives, and this one
already reaches train-F1 0.99. Treat these as an ablation, not as a score lever.
"""
import torch
import torch.nn as nn


def _expert(hidden: int, num_labels: int, dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(hidden, dim), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(dim, num_labels))


class LinearHead(nn.Module):
    """The original head. Dropout is applied by the caller, so this is exactly nn.Linear."""

    needs_tokens = False

    def __init__(self, hidden: int, num_labels: int, **_):
        super().__init__()
        self.fc = nn.Linear(hidden, num_labels)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, attention_mask=None):
        return self.fc(x)


class SparseMoEHead(nn.Module):
    """Top-k routed mixture of experts over the pooled vector (Shazeer et al., 2017).

    Every expert is evaluated and the gate zeroes the ones that were not selected, rather than
    gathering rows per expert. With a handful of tiny experts that is the faster path and the
    arithmetic is identical; the sparsity here buys a routed *function class*, not fewer FLOPs.

    `aux_loss` is the Switch Transformer load-balancing term, E * sum_i f_i * P_i, where f is the
    fraction of examples routed to expert i and P the mean router probability. Without it the
    router collapses onto one expert within a few hundred steps and the layer silently becomes a
    single MLP -- the final score looks merely mediocre, never broken. It is 1.0 when the load is
    perfectly balanced and E when everything lands on one expert; the trainer logs it.
    """

    needs_tokens = False

    def __init__(self, hidden: int, num_labels: int, experts: int = 4, expert_dim: int = 64,
                 top_k: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        if experts < 2:
            raise ValueError(f"model.moe_experts must be >= 2, got {experts}")
        if not 1 <= top_k <= experts:
            raise ValueError(f"model.moe_top_k must be in 1..{experts}, got {top_k}")
        self.n_experts, self.top_k = experts, top_k
        self.router = nn.Linear(hidden, experts)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.router.bias)
        self.experts = nn.ModuleList(_expert(hidden, num_labels, expert_dim, dropout)
                                     for _ in range(experts))
        self.aux_loss = 0.0          # plain attribute, not a buffer: it must stay out of state_dict

    def forward(self, x, attention_mask=None):
        # Autocast has to be switched off, not worked around with .float(): it rewrites the ops,
        # so nn.Linear returns fp16 whatever its input dtype, while softmax is on the fp32 list
        # and returns fp32. Mixing the two is both a silent precision bug (a softmax over a few
        # close logits in fp16 quantises enough to change which expert wins) and a hard error --
        # scatter_ refuses an fp16 destination with an fp32 source.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.router(x.float())
            top, idx = logits.topk(self.top_k, dim=-1)
            gate = torch.zeros_like(logits).scatter_(1, idx, top.softmax(-1))

            P = logits.softmax(-1).mean(0)                   # mean router probability per expert
            f = torch.zeros_like(P).scatter_add_(
                0, idx.reshape(-1), torch.ones(idx.numel(), device=P.device, dtype=P.dtype))
            self.aux_loss = self.n_experts * (f / idx.numel() * P).sum()

        y = torch.stack([e(x) for e in self.experts], dim=1)  # [B, E, C]
        return (gate.unsqueeze(-1).to(y.dtype) * y).sum(1)


class SoftMoEHead(nn.Module):
    """Soft MoE (Puigcerver et al., 2023) over the token sequence, followed by masked mean pooling.

    This one cannot run on a pooled vector: each slot is a softmax-weighted average *over the
    token axis*, so with a sequence of length 1 the dispatch weights are identically 1 and the
    layer degenerates into a fixed weighted sum of experts applied to the same input -- a dense
    ensemble with none of Soft MoE's behaviour. It therefore consumes last_hidden_state and
    replaces pooling entirely; `model.pooling` has no effect when this head is selected.

    Padding is masked out of the dispatch softmax and out of the final mean, so a batch's score
    does not depend on how much padding its neighbours brought.

    No auxiliary loss: Soft MoE is fully dense, so there is no load to balance.
    """

    needs_tokens = True

    def __init__(self, hidden: int, num_labels: int, experts: int = 4, expert_dim: int = 64,
                 slots: int = 1, dropout: float = 0.1, **_):
        super().__init__()
        if experts < 2:
            raise ValueError(f"model.moe_experts must be >= 2, got {experts}")
        if slots < 1:
            raise ValueError(f"model.moe_slots must be >= 1, got {slots}")
        self.n_experts, self.slots = experts, slots
        self.phi = nn.Parameter(torch.empty(hidden, experts * slots))
        nn.init.normal_(self.phi, std=hidden ** -0.5)
        self.drop = nn.Dropout(dropout)
        self.experts = nn.ModuleList(_expert(hidden, num_labels, expert_dim, dropout)
                                     for _ in range(experts))

    def forward(self, h, attention_mask=None):
        # Autocast off for the same reason as SparseMoEHead: it would make the matmul fp16 and
        # the softmax fp32 regardless of the inputs' dtype. Here the dispatch softmax runs over
        # the token axis, where fp16 on a padded row of -65504 is especially lossy.
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = h.float() @ self.phi.float()                   # [B, T, E*s]
            if attention_mask is not None:
                pad = (attention_mask == 0).unsqueeze(-1)
                logits = logits.masked_fill(pad, torch.finfo(logits.dtype).min)

            dispatch = logits.softmax(dim=1)                        # over tokens -> slot contents
            slots = torch.einsum("btj,bth->bjh", dispatch, h.float())   # [B, E*s, H]
        slots = self.drop(slots).to(h.dtype)

        per_slot = slots.view(h.size(0), self.n_experts, self.slots, -1)
        y = torch.stack([e(per_slot[:, i]) for i, e in enumerate(self.experts)], dim=1)
        y = y.reshape(h.size(0), self.n_experts * self.slots, -1)   # [B, E*s, C]

        with torch.autocast(device_type=h.device.type, enabled=False):
            combine = logits.softmax(dim=2)                         # over slots -> back to tokens
        out = torch.einsum("btj,bjc->btc", combine.to(y.dtype), y)  # [B, T, C]
        if attention_mask is None:
            return out.mean(1)
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)


HEADS = {"linear": LinearHead, "sparse_moe": SparseMoEHead, "soft_moe": SoftMoEHead}


def build_head(kind, hidden: int, num_labels: int, cfg=None) -> nn.Module:
    """`cfg` is the model config block; only the moe_* keys are read."""
    kind = kind or "linear"
    if kind not in HEADS:
        raise ValueError(f"unknown model.head {kind!r}; expected one of {sorted(HEADS)}")
    c = cfg or {}
    return HEADS[kind](hidden, num_labels,
                       experts=c.get("moe_experts", 4), expert_dim=c.get("moe_expert_dim", 64),
                       top_k=c.get("moe_top_k", 2), slots=c.get("moe_slots", 1),
                       dropout=c.get("moe_dropout", 0.1))


def head_config(cfg) -> dict:
    """The subset of the model config a head depends on -- stored in meta.json so a checkpoint
    rebuilds the same head. Only the keys the selected head actually reads, so switching from
    sparse to soft does not leave a stale top_k in the record."""
    kind = cfg.get("head") or "linear"
    keys = {"linear": (), "sparse_moe": ("moe_experts", "moe_expert_dim", "moe_top_k", "moe_dropout"),
            "soft_moe": ("moe_experts", "moe_expert_dim", "moe_slots", "moe_dropout")}[kind]
    return {"head": kind, **{k: cfg[k] for k in keys if k in cfg}}
