"""Adversarial training and weight averaging -- the two regularisers this task keeps asking for.

Every model family tried here overfits the same way and at the same speed. From a real LLM run:

    ep 2  train 0.7723  eval 0.7981
    ep 3  train 0.8980  eval 0.7838
    ep 4  train 0.9802  eval 0.7636

Nine architectures, three tokenisation granularities and a 10x range of model size all landed in
0.78-0.83, so the ceiling is in the data. What is left is not a better encoder but a flatter
minimum, and these two are the cheapest ways to get one on a few thousand rows.
"""
import torch


class FGM:
    """Fast Gradient Method (Miyato et al., 2017): one adversarial step on the embeddings.

    After the ordinary backward pass the embedding gradient points at the direction that would
    most increase the loss. Nudging the embeddings that way, running the batch again and adding
    the second gradient trains the model to be right in a neighbourhood rather than at a point.
    Half the cost of a second model, and it grows more useful as data shrinks.

    The perturbation is normalised per tensor, so `eps` is a step size in embedding units and
    does not need retuning per model.

    It needs the embeddings to carry a gradient. freeze_embeddings and LoRA both leave them
    frozen, and then there is nothing to perturb -- `armed` says so rather than letting the run
    look regularised when it is not.
    """

    def __init__(self, model, eps: float = 1.0, match: str = "embed"):
        self.eps, self.match = eps, match.lower()
        self.params = [(n, p) for n, p in model.named_parameters()
                       if p.requires_grad and self.match in n.lower()]
        self.backup = {}

    @property
    def armed(self) -> bool:
        return bool(self.params)

    def attack(self):
        for n, p in self.params:
            if p.grad is None:
                continue
            norm = torch.norm(p.grad)
            if norm != 0 and torch.isfinite(norm):
                self.backup[n] = p.data.clone()
                p.data.add_(self.eps * p.grad / norm)

    def restore(self):
        for n, p in self.params:
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup.clear()


class EMA:
    """Exponential moving average of the trainable weights.

    SGD on a small set spends its last epochs bouncing around a minimum rather than descending
    into one; the average of those positions generalises better than any of them. Costs one extra
    copy of the trainable parameters and a multiply-add per step -- which under LoRA is a few
    million numbers, not the whole model.

    `swap_in` puts the averaged weights in place for evaluation and `swap_out` restores the live
    ones, so training continues from the real trajectory. The trainer keeps whichever weights it
    evaluated, so a run with EMA on saves the averaged model.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.live = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is not None:
                s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model):
        for n, p in model.named_parameters():
            s = self.shadow.get(n)
            if s is not None:
                self.live[n] = p.detach().clone()
                p.data.copy_(s)

    @torch.no_grad()
    def swap_out(self, model):
        for n, p in model.named_parameters():
            if n in self.live:
                p.data.copy_(self.live[n])
        self.live.clear()
