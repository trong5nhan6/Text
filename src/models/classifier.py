import json
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.models.heads import build_head

_BLOCK_RE = re.compile(r"^(?P<prefix>.+?)\.(?P<idx>\d+)\.")

# model.dtype. None (the default) keeps the historical behaviour: load, then .float().
_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


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


def _neutralise_torchao_check():
    """Stop peft's torchao probe from raising, so LoRA does not depend on the host's torchao.

    While injecting adapters peft asks each optional backend whether it is available, and its
    torchao check raises ImportError when the installed torchao is older than it wants rather
    than simply answering no. Kaggle ships 0.10.0 against peft's >=0.16, so `get_peft_model`
    died before touching a single layer -- on an integration nothing in this pipeline uses.

    Answering "no" is exactly right here, and doing it in code means the run does not hinge on
    someone remembering to uninstall a package first. Only the probe is replaced, and only when
    it actually raises; a healthy torchao is left alone.
    """
    try:
        from peft import import_utils
    except ImportError:
        return
    try:
        import_utils.is_torchao_available()
        return                                  # healthy, or absent -- either way, leave it
    except ImportError:
        pass
    false = (lambda: False)
    import_utils.is_torchao_available = false
    # The dispatcher imported the function by name, so its own module-level reference is the
    # one that actually gets called; patching import_utils alone would change nothing.
    try:
        from peft.tuners.lora import torchao as lora_torchao
        lora_torchao.is_torchao_available = false
    except ImportError:
        pass


def _check_layers(layers, n_hs: int, backbone_name: str):
    """null | "mix" | a list of hidden-state indices. Validated here so a typo fails at build
    time with the valid range, instead of as an IndexError mid-epoch."""
    if layers is None or layers == "mix":
        return layers
    if isinstance(layers, int):
        layers = [layers]
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError(f"model.layers must be null, \"mix\", or a non-empty list of indices; "
                         f"got {layers!r}")
    layers = [int(i) + n_hs if int(i) < 0 else int(i) for i in layers]
    bad = [i for i in layers if not 0 <= i < n_hs]
    if bad:
        raise ValueError(f"model.layers {bad} out of range for {backbone_name}: it has {n_hs} "
                         f"hidden states (0 = embeddings .. {n_hs - 1} = last_hidden_state)")
    return list(layers)


class TransformerClassifier(nn.Module):
    """Pretrained encoder + pooling + dropout + linear head."""

    def __init__(self, backbone_name: str, num_labels: int, pooling: str = "cls",
                 dropout: float = 0.1, backbone_config=None,
                 unfreeze_last_n_blocks=None, freeze_embeddings=None, head_cfg=None,
                 layers=None, load_in_4bit=False, lora=None, dtype=None,
                 multisample_dropout=None, hybrid=None):
        super().__init__()
        self.name = backbone_name        # needed before self.meta exists, e.g. by _apply_lora
        self.quantized = bool(load_in_4bit)
        if load_in_4bit and backbone_config is None:
            # A 7B decoder needs ~14 GB in fp16 and ~4 GB in nf4, which is the difference between
            # fitting on a T4 and not. Never call .float() on it afterwards: that would
            # de-quantise straight back to the memory this exists to avoid.
            import torch as _t
            from transformers import BitsAndBytesConfig
            self.backbone = AutoModel.from_pretrained(
                backbone_name, quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=_t.float16))
        elif backbone_config is None:                    # training: load pretrained weights
            # .float() is not redundant: some published checkpoints store fp16 weights
            # (mDeBERTa-v3 does) and transformers keeps the checkpoint's dtype, which then
            # meets the fp32 head as "mat1 and mat2 must have the same dtype". AMP wants fp32
            # master weights anyway -- mixed precision is applied by autocast, not by the weights.
            #
            # That reasoning holds for an encoder being fully fine-tuned, and breaks for a
            # frozen LoRA base: those weights never take a gradient, so there is nothing for an
            # fp32 master copy to accumulate into, and it doubles the footprint for nothing.
            # sarvam-1 is 9.64 GB in fp32 against 4.82 in fp16, which on a 14.56 GB T4 is the
            # difference between running and OOM. model.dtype: fp16 says so explicitly.
            want = _DTYPES[dtype or "fp32"]
            self.backbone = AutoModel.from_pretrained(backbone_name, dtype=want)
            if want is None:
                self.backbone = self.backbone.float()
        else:                                            # inference: architecture only, weights come from ckpt
            # .float() for the same reason, from the other direction: a config saved off an fp16
            # backbone builds an fp16 one, and load_state_dict copies in place, so fp16 would stick.
            self.backbone = AutoModel.from_config(backbone_config).float()
        if lora:
            self._apply_lora(lora)
        self.n_dropout = max(1, int(multisample_dropout or 1))   # read by meta just below
        head_cfg = dict(head_cfg or {"head": "linear"})
        hidden = self.backbone.config.hidden_size
        n_hs = self._n_hidden_states()
        self.layers = _check_layers(layers, n_hs, backbone_name)
        # A list of layers widens the head; "mix" sums them, so the width is unchanged.
        width = hidden * (len(self.layers) if isinstance(self.layers, list) else 1)
        if self.layers == "mix":
            # One logit per hidden state, softmaxed at use. 13 parameters on a base model.
            # Zeros -> a uniform mix at initialisation, so training starts from the average.
            self.layer_weights = nn.Parameter(torch.zeros(n_hs))
        # model.hybrid: a TF-IDF branch projected to hybrid["dim"] and concatenated to the pooled
        # state. {"in_dim", "dim", "dropout"}; in_dim is the fitted featurizer's width, so it is
        # only known once the fit slice has been seen -- train.py fills it in before building.
        self.hybrid = dict(hybrid) if hybrid else None
        # Set by train.py / load(): the fitted TfidfFeaturizer. A plain attribute, not a module,
        # and read by the Trainer to build the loaders.
        self.featurizer = None
        if self.hybrid:
            if head_cfg.get("head") == "soft_moe":
                raise SystemExit("model.hybrid khong di chung voi head soft_moe: soft_moe doc chuoi "
                                 "token, khong co vector pooled nao de ghep nhanh TF-IDF vao.")
            # Heavy input dropout: the projection has in_dim x dim weights (~80k x 256 = 20M) fed
            # by 2.8k-5.7k rows, and a sparse input it can memorise. Dropout on the input is what
            # an n-gram model's L2 penalty does in tfidf.py.
            self.tfidf_proj = nn.Sequential(nn.Dropout(self.hybrid.get("dropout", 0.3)),
                                            nn.Linear(self.hybrid["in_dim"], self.hybrid["dim"]),
                                            nn.GELU())
            width += self.hybrid["dim"]
        self.meta = {"backbone_name": backbone_name, "num_labels": num_labels,
                     "pooling": pooling, "dropout": dropout, "head_cfg": head_cfg,
                     "layers": self.layers, "multisample_dropout": self.n_dropout,
                     "hybrid": self.hybrid}
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        # Kept under the name `head` on purpose: param_groups() routes head.* to head_lr and
        # excludes it from the LLRD ladder by that prefix, and both must keep holding.
        self.head = build_head(head_cfg.get("head"), width, num_labels, head_cfg)
        # Recorded now, as a plain attribute, because forward needs it and forward also runs
        # inside DataParallel replicas -- where parameters() yields nothing. replicate() sets
        # each replica._parameters[key] to None and re-attaches the broadcast tensor with
        # setattr, so next(self.head.parameters()) raises StopIteration there. A plain attribute
        # survives, since replicate copies __dict__.
        self.head_dtype = next(self.head.parameters()).dtype
        self.freeze_summary = self._apply_freezing(unfreeze_last_n_blocks, freeze_embeddings)

    # Attention and MLP projections, by the names the Llama/Gemma/Qwen families use. Anything a
    # given model does not have is dropped before peft sees it, so one list covers all of them.
    LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def _apply_lora(self, lora):
        """Low-rank adapters on the backbone. Full fine-tuning of a 2B decoder on 2,827 rows is
        roughly a million parameters per example; LoRA keeps the trainable count in the millions
        and leaves the pretrained weights alone, which is the whole reason a big model is worth
        starting from here."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            raise SystemExit("model.lora can peft: pip install peft bitsandbytes accelerate")
        cfg = lora if isinstance(lora, dict) else {}
        present = {n.split(".")[-1] for n, _ in self.backbone.named_modules()}
        targets = [t for t in cfg.get("target_modules", self.LORA_TARGETS) if t in present]
        if not targets:
            raise SystemExit(f"khong tim thay lop nao de gan LoRA trong {self.name}; "
                             f"dat model.lora.target_modules bang tay")
        if self.quantized:
            self.backbone = prepare_model_for_kbit_training(self.backbone)
        _neutralise_torchao_check()
        self.backbone = get_peft_model(self.backbone, LoraConfig(
            r=cfg.get("r", 16), lora_alpha=cfg.get("alpha", 32),
            lora_dropout=cfg.get("dropout", 0.05), bias="none", target_modules=targets))
        self.lora_targets = targets

    def _n_hidden_states(self) -> int:
        """hidden_states has one entry per block plus one for the embedding output, so a 12-block
        model yields 13 and index 12 is exactly last_hidden_state."""
        return getattr(self.backbone.config, "num_hidden_layers", 0) + 1

    def _combine(self, hidden_states):
        """Fold the requested hidden states into one [B, T, width] tensor.

        Done at token level rather than after pooling, which is the same thing -- concatenation
        and both pooling modes are linear in the token axis, so they commute -- but it keeps one
        code path for the pooled heads and for soft_moe, which needs the sequence.
        """
        # CANINE breaks both assumptions this makes: it returns 17 states, not
        # num_hidden_layers + 1, and they are at two different resolutions -- some at character
        # length, some downsampled 4x. Summing or concatenating across those shapes is either a
        # crash or, worse, a broadcast that silently computes nonsense. Check once, here.
        n = len(hidden_states)
        bad = [i for i in self.layers if i >= n] if isinstance(self.layers, list) else []
        if bad:
            raise SystemExit(
                f"model.layers {bad} vuot qua {n} hidden_states ma {self.meta['backbone_name']} "
                f"tra ve. Kiem tra so tang that su cua model nay.")
        # Checked over ALL the states, not just the selected ones: on CANINE a pair like [2, 12]
        # can happen to share a shape while both sit at the downsampled resolution, so the concat
        # succeeds and then silently misaligns with attention_mask. Layers are only in the same
        # space if every state is.
        shapes = {tuple(h.shape) for h in hidden_states}
        if len(shapes) > 1:
            raise SystemExit(
                f"model.layers khong dung duoc voi {self.meta['backbone_name']}: cac hidden_states "
                f"co shape khac nhau {sorted(shapes)}. Kien truc nay ha mau giua chung (CANINE), "
                f"nen cac tang khong nam cung mot khong gian. Dat model.layers: null.")

        if self.layers == "mix":
            w = self.layer_weights.float().softmax(0).to(hidden_states[0].dtype)
            return sum(w[i] * h for i, h in enumerate(hidden_states))
        return torch.cat([hidden_states[i] for i in self.layers], dim=-1)

    def layer_mix(self):
        """-> the learned per-layer weights, for reporting. None unless layers == "mix"."""
        if self.layers != "mix":
            return None
        with torch.no_grad():
            return [round(float(x), 4) for x in self.layer_weights.float().softmax(0)]

    def _apply_freezing(self, unfreeze_last_n, freeze_emb) -> str:
        """Freeze everything below the last `unfreeze_last_n` blocks. The head and the modules
        that sit *above* the last block (mDeBERTa's encoder.LayerNorm, ModernBERT's final_norm)
        always stay trainable -- freezing those would cut the path the gradient needs.
        `freeze_emb=None` means: follow the block setting.

        With LoRA this steps aside entirely. peft has already frozen every base weight and left
        only the adapters trainable, which is the whole point; the embedding branch below calls
        requires_grad_(True) on anything matching "embed" and would hand back the entire
        embedding matrix -- measured at 145M of 503M trainable on a 0.5B model where LoRA r=16
        should account for about 8M."""
        if getattr(self, "lora_targets", None):
            if unfreeze_last_n is not None or freeze_emb is not None:
                raise SystemExit("model.lora khong di chung voi unfreeze_last_n_blocks / "
                                 "freeze_embeddings: LoRA da quyet dinh cai gi duoc train.")
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            return (f"LoRA tren {len(self.lora_targets)} loai lop | "
                    f"{trainable / 1e6:.1f}M / {total / 1e6:.1f}M params "
                    f"({100 * trainable / total:.2f}%)")
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

    def forward(self, input_ids, attention_mask, token_type_ids=None, tfidf=None, **_):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        if self.layers is None:
            h = self.backbone(**kw).last_hidden_state
        else:
            # The intermediate states are computed on the way through regardless; without this
            # flag transformers simply drops them.
            h = self._combine(self.backbone(**kw, output_hidden_states=True).hidden_states)
        # soft_moe aggregates the token axis itself, so it takes the sequence and `pooling` is
        # unused. Every other head takes the pooled vector, with dropout applied here so that
        # `linear` stays exactly what it was.
        # The head stays fp32 for the classifier's own numerics, so a backbone loaded in fp16
        # (model.dtype) has to be cast up on the way in -- otherwise "mat1 and mat2 must have
        # the same dtype, but got Half and Float". Under autocast this is a no-op.
        h = h.to(self.head_dtype)
        if getattr(self.head, "needs_tokens", False):
            return self.head(h, attention_mask)
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        elif self.pooling == "last":
            # Decoder LLMs attend causally, so position 0 sees only itself and carries nothing
            # about the sentence -- `cls` pooling on one of them reads the BOS token and quietly
            # throws the comment away. The last real token is the only position that has seen
            # all of them. Collator pads on the right, so that index is attention_mask.sum-1.
            idx = attention_mask.sum(1).long() - 1
            pooled = h[torch.arange(h.size(0), device=h.device), idx]
        else:
            pooled = h[:, 0]
        if self.hybrid:
            if tfidf is None:
                raise RuntimeError("model.hybrid=tfidf nhung batch khong co `tfidf`: loader phai "
                                   "duoc tao voi featurizer (model.featurizer).")
            pooled = torch.cat([pooled, self.tfidf_proj(tfidf.to(self.head_dtype))], -1)
        if self.n_dropout > 1 and self.training:
            # Multi-sample dropout (Inoue, 2019): average the head over several dropout masks of
            # the same pooled vector. It lowers the variance of each update at roughly no cost --
            # only the head runs again, not the encoder -- and needs no extra data. Training only:
            # at eval dropout is off, so the masks would be identical and the average a no-op.
            return sum(self.head(self.dropout(pooled)) for _ in range(self.n_dropout)) / self.n_dropout
        return self.head(self.dropout(pooled))

    @property
    def aux_loss(self):
        """Router load-balancing term of a sparse MoE head; 0.0 for every other head, which the
        trainer can add unconditionally."""
        return getattr(self.head, "aux_loss", 0.0)

    def param_groups(self, lr, head_lr, weight_decay, llrd=None, mix_lr=None):
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
                     if p.requires_grad and not n.startswith(("head.", "tfidf_proj."))]
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
            if n == "layer_weights":
                # These 13 softmax logits need a much larger step than anything else. AdamW moves
                # a parameter by roughly its lr per step whatever the gradient size, so at
                # head_lr 1e-4 the logits drift ~1e-4/step and the mix is still uniform to four
                # decimals after an epoch -- "mix" would silently degrade into a plain average of
                # all layers. mix_lr is a separate, much larger rate. No weight decay: decaying a
                # softmax logit only pulls the mix back toward uniform, which is not a prior
                # worth imposing.
                key, p_lr, wd = ("mix", 0.0), mix_lr or head_lr or lr, 0.0
            elif n.startswith("head."):
                key, p_lr = ("head", wd), head_lr or lr
            elif n.startswith("tfidf_proj."):
                # Randomly initialised like the head, and fed sparse inputs of ~0.1 whose weights
                # each see a gradient only when their n-gram occurs: 1e-4 barely moves them in
                # six epochs, so the branch gets its own, larger rate (model.hybrid.lr).
                key, p_lr = ("hybrid", wd), (self.hybrid or {}).get("lr") or head_lr or lr
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
        # A quantised or LoRA-wrapped state_dict does not come back through load(): the 4-bit
        # tensors are not plain weights, and peft renames every module it wraps. Writing one
        # would produce a checkpoint that silently fails to restore later, so refuse now and say
        # what to do instead.
        if self.quantized or getattr(self, "lora_targets", None):
            raise SystemExit("khong luu duoc checkpoint khi dung model.load_in_4bit / model.lora. "
                             "Dat checkpoint.save=none; eval.npy va file nop van duoc sinh.")
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: (v.half() if half and v.is_floating_point() else v).cpu() for k, v in self.state_dict().items()}
        torch.save(sd, out_dir / "model.pt")
        self.backbone.config.save_pretrained(out_dir / "backbone")
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir / "tokenizer")
        if self.featurizer is not None:
            self.featurizer.save(out_dir / "tfidf.pkl")
        json.dump({**self.meta, **(extra or {})}, open(out_dir / "meta.json", "w"), indent=1)

    @classmethod
    def load(cls, ckpt_dir, map_location="cpu"):
        ckpt_dir = Path(ckpt_dir)
        meta = json.load(open(ckpt_dir / "meta.json"))
        bcfg = AutoConfig.from_pretrained(ckpt_dir / "backbone")
        # .get: checkpoints written before model.head existed carry no head_cfg and are linear.
        model = cls(meta["backbone_name"], meta["num_labels"], meta["pooling"], meta["dropout"],
                    backbone_config=bcfg, head_cfg=meta.get("head_cfg"),
                    layers=meta.get("layers"),
                    multisample_dropout=meta.get("multisample_dropout"),
                    hybrid=meta.get("hybrid"))
        sd = torch.load(ckpt_dir / "model.pt", map_location=map_location)
        model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        if model.hybrid:
            from src.models.hybrid import TfidfFeaturizer
            model.featurizer = TfidfFeaturizer.load(ckpt_dir / "tfidf.pkl")
        return model, meta
