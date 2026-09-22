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
        # self.model always stays the real module -- param_groups, save, freeze_summary and
        # layer_mix all read it. self.net is what forward goes through, and is the same object
        # unless there are several GPUs to split the batch across.
        self.net = self._maybe_parallel()

    def _maybe_parallel(self):
        n_gpu = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if n_gpu < 2 or self.t.get("single_gpu"):
            return self.model
        # DataParallel runs forward on replicas, so anything a head records on itself as a side
        # effect is written to a replica and thrown away -- self.model.aux_loss would stay at its
        # initial 0.0 forever. For a sparse MoE head that silently removes the load-balancing
        # term, the router collapses onto one expert, and the only symptom is a mediocre score.
        # Refuse rather than hide it.
        if getattr(self.model.head, "aux_loss", None) is not None:
            self.log.info(f"co {n_gpu} GPU nhung head la sparse_moe -> chay 1 GPU. DataParallel "
                          f"se lam mat aux_loss (router sup ve 1 expert ma khong bao loi).")
            return self.model
        gathered = self.t["batch_size"] // n_gpu
        self.log.info(f"DataParallel tren {n_gpu} GPU -- batch {self.t['batch_size']} chia thanh "
                      f"{gathered}/GPU")
        return torch.nn.DataParallel(self.model)

    def fit(self, train_df, valid_df):
        t = self.t
        dl_tr = make_loader(train_df.text.tolist(), train_df.y.tolist(), self.tok, self.cfg, train=True)
        # No held-out slice (data.use_valdataset: false): nothing to score against, so there is
        # no best epoch to keep and no early stopping. The run trains the full schedule and
        # returns its LAST epoch, which is why `epochs` has to be set deliberately in this mode.
        has_val = valid_df is not None and len(valid_df) > 0
        dl_va = (make_loader(valid_df.text.tolist(), None, self.tok, self.cfg, train=False)
                 if has_val else None)

        pg = self.model.param_groups(t["lr"], t.get("head_lr"), t["weight_decay"], t.get("llrd"),
                                     t.get("layer_mix_lr"))
        opt = torch.optim.AdamW(pg)
        if t.get("llrd"):
            back = [g["lr"] for g in pg if g["lr"] != (t.get("head_lr") or t["lr"])] or [t["lr"]]
            self.log.info(f"llrd={t['llrd']}: {len(pg)} nhom, lr backbone "
                          f"{min(back):.2e} (duoi) -> {max(back):.2e} (tren)")
        steps = math.ceil(len(dl_tr) / t["grad_accum"]) * t["epochs"]
        sch = get_linear_schedule_with_warmup(opt, int(t["warmup_ratio"] * steps), steps)
        scaler = torch.amp.GradScaler(enabled=self.amp_dtype == torch.float16)
        patience = t.get("early_stopping_patience") or t["epochs"]

        # Balanced routing gives aux == 1.0, total collapse onto one expert gives aux == n_experts.
        # Logged every epoch because a collapsed router is invisible in the macro-F1 alone.
        aux_w = t.get("moe_aux_weight", 0.0) or 0.0
        best = {"f1": -1.0, "epoch": 0, "state": None, "pred": None}
        history, bad = [], 0
        for ep in range(1, t["epochs"] + 1):
            self.model.train(); t0 = time.time(); total = 0.0; aux_sum = 0.0
            tr_pred, tr_true = [], []
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(dl_tr):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                y = batch.pop("labels")
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float32,
                                    enabled=self.amp_dtype is not None):
                    logits = self.net(**batch)
                # Running train scores, gathered from the forward passes the step already did, so
                # they cost nothing. They are NOT a clean evaluation: dropout is on and the
                # weights move between batches, so early-epoch batches are scored by a worse
                # model than late-epoch ones. Read them for the train/held-out gap, not as an
                # exact number.
                tr_pred.append(logits.detach().argmax(-1).cpu()); tr_true.append(y.detach().cpu())
                loss = self.loss_fn(logits.float(), y)
                aux = self.model.aux_loss          # 0.0 unless the head is a sparse MoE
                if aux_w:
                    loss = loss + aux_w * aux
                    aux_sum += float(aux)
                loss = loss / t["grad_accum"]
                scaler.scale(loss).backward()
                total += loss.item() * t["grad_accum"]
                if (i + 1) % t["grad_accum"] == 0 or i + 1 == len(dl_tr):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["max_grad_norm"])
                    scaler.step(opt); scaler.update(); sch.step()
                    opt.zero_grad(set_to_none=True)

            p_va = predict_proba(self.net, dl_va, self.device, self.amp_dtype) if has_val else None
            m = compute_metrics(valid_df.y, p_va.argmax(1)) if has_val else {}
            aux_avg = aux_sum / len(dl_tr) if aux_w else None
            tm = compute_metrics(torch.cat(tr_true).numpy(), torch.cat(tr_pred).numpy())
            history.append({"epoch": ep, "loss": round(total / len(dl_tr), 4),
                            "train_macro_f1": round(tm["macro_f1"], 4),
                            "train_accuracy": round(tm["accuracy"], 4),
                            **({"moe_aux": round(aux_avg, 4)} if aux_w else {}), **m})
            improved = has_val and m["macro_f1"] > best["f1"]
            self.log.info(f"ep {ep}/{t['epochs']} loss {total / len(dl_tr):.4f} "
                          f"| train F1 {tm['macro_f1']:.4f} acc {tm['accuracy']:.4f} "
                          + (f"| eval F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} " if has_val
                             else "| khong co lat eval (use_valdataset=false) ")
                          + (f"moe_aux {aux_avg:.3f} " if aux_w else "")
                          + f"({time.time() - t0:.0f}s){' *' if improved else ''}")
            if not has_val:
                best = {"f1": float("nan"), "epoch": ep, "pred": None, "state": None}
                continue
            if improved:
                bad = 0
                best = {"f1": m["macro_f1"], "epoch": ep, "pred": p_va,
                        "state": {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}}
            else:
                bad += 1
                if bad >= patience:
                    self.log.info(f"early stop (no improvement for {patience} epochs)")
                    break

        if best["state"] is not None:
            # self.model, not self.net: a DataParallel state_dict is prefixed with "module."
            self.model.load_state_dict(best["state"])  # restore best epoch
        else:
            self.log.info(f"khong co lat eval -> giu epoch CUOI ({best['epoch']}), "
                          f"khong chon duoc epoch tot nhat")
        best["history"] = history
        del best["state"], opt
        return best

    def predict(self, texts):
        if texts is None or len(texts) == 0:
            return None
        dl = make_loader(list(texts), None, self.tok, self.cfg, train=False)
        return predict_proba(self.net, dl, self.device, self.amp_dtype)
