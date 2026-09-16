"""Trainer: AdamW + linear warmup, AMP, grad accumulation, early stopping on macro-F1."""
import copy
import math
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from src.data.dataset import make_loader
from src.evaluation.metrics import compute_metrics


def resolve_precision(name: str, device: torch.device):
    if device.type != "cuda" or name == "fp32":
        return None
    if name == "auto":
        native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8     # Ampere+ (A100, L4, 30xx/40xx)
        return torch.bfloat16 if native_bf16 else torch.float16           # T4 / P100 -> fp16
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


@torch.no_grad()
def predict_proba(model, loader, device, amp_dtype=None) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            out.append(model(**batch).float().softmax(-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


class Trainer:
    def __init__(self, cfg, model, tokenizer, loss_fn, device, logger):
        self.cfg, self.t = cfg, cfg["training"]
        self.model = model.to(device)
        self.tok = tokenizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.log = logger
        self.amp_dtype = resolve_precision(self.t.get("precision", "auto"), device)

    def fit(self, train_df, valid_df):
        t = self.t
        dl_tr = make_loader(train_df.text.tolist(), train_df.y.tolist(), self.tok, self.cfg, train=True)
        dl_va = make_loader(valid_df.text.tolist(), None, self.tok, self.cfg, train=False)

        opt = torch.optim.AdamW(self.model.param_groups(t["lr"], t.get("head_lr"), t["weight_decay"]))
        steps = math.ceil(len(dl_tr) / t["grad_accum"]) * t["epochs"]
        sch = get_linear_schedule_with_warmup(opt, int(t["warmup_ratio"] * steps), steps)
        scaler = torch.amp.GradScaler(enabled=self.amp_dtype == torch.float16)
        patience = t.get("early_stopping_patience") or t["epochs"]

        best = {"f1": -1.0, "epoch": 0, "state": None, "pred": None}
        history, bad = [], 0
        for ep in range(1, t["epochs"] + 1):
            self.model.train(); t0 = time.time(); total = 0.0
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(dl_tr):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                y = batch.pop("labels")
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float32,
                                    enabled=self.amp_dtype is not None):
                    logits = self.model(**batch)
                loss = self.loss_fn(logits.float(), y) / t["grad_accum"]
                scaler.scale(loss).backward()
                total += loss.item() * t["grad_accum"]
                if (i + 1) % t["grad_accum"] == 0 or i + 1 == len(dl_tr):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["max_grad_norm"])
                    scaler.step(opt); scaler.update(); sch.step()
                    opt.zero_grad(set_to_none=True)

            p_va = predict_proba(self.model, dl_va, self.device, self.amp_dtype)
            m = compute_metrics(valid_df.y, p_va.argmax(1))
            history.append({"epoch": ep, "loss": round(total / len(dl_tr), 4), **m})
            improved = m["macro_f1"] > best["f1"]
            self.log.info(f"ep {ep}/{t['epochs']} loss {total / len(dl_tr):.4f} "
                          f"macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} "
                          f"({time.time() - t0:.0f}s){' *' if improved else ''}")
            if improved:
                bad = 0
                best = {"f1": m["macro_f1"], "epoch": ep, "pred": p_va,
                        "state": {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}}
            else:
                bad += 1
                if bad >= patience:
                    self.log.info(f"early stop (no improvement for {patience} epochs)")
                    break

        self.model.load_state_dict(best["state"])      # restore best epoch
        best["history"] = history
        del best["state"], opt
        return best

    def predict(self, texts):
        if texts is None or len(texts) == 0:
            return None
        dl = make_loader(list(texts), None, self.tok, self.cfg, train=False)
        return predict_proba(self.model, dl, self.device, self.amp_dtype)
