"""YAML config loading with `_base_` inheritance and `key.sub=value` CLI overrides."""
import copy
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    return _deep_update(_load(path.parent / base), cfg) if base else cfg


def _parse_value(v: str):
    val = yaml.safe_load(v)
    if isinstance(val, str):
        if val.strip() in ("None", "none", "NONE"):   # YAML only knows null/~; Python prints None
            return None
        try:
            val = float(val)                 # YAML 1.1 reads "1e-3" as a string
        except ValueError:
            pass
    return val


def set_by_dotted(cfg: dict, key: str, value):
    node = cfg
    *parents, last = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def load_config(path, overrides=(), **top_level) -> dict:
    """overrides: ["training.lr=1e-5", ...]; top_level: task="b", seed=1 (None values ignored)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = _load(path)
    cfg["config_name"] = path.stem
    # Remembered before the overrides run, so run_name() can tell an overridden checkpoint from
    # the one the config file names and tag the run directory accordingly.
    cfg["_config_model_name"] = cfg.get("model", {}).get("name")
    for o in overrides or ():
        k, v = o.split("=", 1)
        set_by_dotted(cfg, k.strip(), _parse_value(v))
    for k, v in top_level.items():
        if v is not None:
            cfg[k] = v
    # resolve paths relative to the project root
    cfg["paths"] = {k: str((ROOT / v) if not Path(v).is_absolute() else Path(v)) for k, v in cfg["paths"].items()}
    return cfg


def resolve_loss(cfg: dict) -> str:
    """`auto` picks per task: plain CE for the balanced binary task, focal for the 6-class one
    (7.3x imbalance, macro-F1 scores every class the same). The resolved name goes into the run
    name, so switching the default lands in a new folder instead of overwriting the old runs."""
    loss = cfg["training"].get("loss", "auto")
    return ("ce" if cfg["task"] == "a" else "focal") if loss == "auto" else loss


def text_suffix(cfg: dict) -> str:
    """Runs on a different view of the text must not share a results/ folder."""
    d = cfg.get("data", {})
    tt = d.get("text_type")
    return ("" if tt in (None, "latin") else f"_{tt}") + ("_tta" if d.get("tta") else "")


_TAG_SKIP = {"checkpoints", "results", "models", ".", ".."}


def model_tag(cfg: dict) -> str:
    """A short tag for model.name when --set overrode what the config file asked for.

    Without it, `--config muril.yaml --set model.name=checkpoints/mlm/muril-base-cased` lands in
    results/{task}/muril_ce_s42 -- the same directory as the plain MuRIL run. train.py would
    refuse it (the signatures differ), so nothing is overwritten, but a loop over several configs
    would stop on the error instead of producing both runs. The tag makes the two names differ by
    themselves: muril_ce_s42 and muril_ce_s42_mlm.

    Prefers the parent directory when it carries the meaning (checkpoints/**mlm**/x -> "mlm") and
    falls back to the basename otherwise.
    """
    name, was = cfg.get("model", {}).get("name"), cfg.get("_config_model_name")
    if not name or not was or name == was:
        return ""
    parts = [p for p in str(name).replace("\\", "/").rstrip("/").split("/") if p]
    tag = parts[-2] if len(parts) >= 2 and parts[-2] not in _TAG_SKIP else parts[-1]
    return "_" + re.sub(r"[^A-Za-z0-9]+", "-", tag).strip("-").lower()[:20]


def run_name(cfg: dict) -> str:
    """<config>_<loss>_s<seed> (tfidf: <config>_<clf>), plus an optional --run_suffix.
    The suffix exists so a batch of runs with changed hyper-parameters can keep the
    informative default name instead of being renamed one by one."""
    if cfg.get("run_name"):
        return cfg["run_name"]
    if cfg["model"]["type"] == "tfidf":
        clf = cfg["model"].get("clf", "lr")
        # tfidf_mlp.yaml with clf mlp -> tfidf_mlp, not tfidf_mlp_mlp
        base = cfg["config_name"] if cfg["config_name"].endswith(f"_{clf}") else f"{cfg['config_name']}_{clf}"
    else:
        base = f"{cfg['config_name']}_{resolve_loss(cfg)}_s{cfg['seed']}"
    # _full is automatic: a model fitted on 100% of the rows must never land in the same
    # directory as one fitted on 90%, because only the latter has an eval.npy to compare.
    full = "" if cfg.get("data", {}).get("use_valdataset") is not False else "_full"
    # _hyb: the hybrid is a different model, and must not share a folder with its plain encoder
    hyb = "_hyb" if cfg["model"].get("hybrid") and cfg["model"]["type"] != "tfidf" else ""
    return base + text_suffix(cfg) + model_tag(cfg) + hyb + full + (cfg.get("run_suffix") or "")


def dump(cfg: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(path, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)


# keys that do not change the model's results -> ignored when checking a resumed run
# _config_model_name only feeds run_name; it is bookkeeping about where model.name came from,
# not something the trained weights depend on. Leaving it in the signature would make every run
# finished before it existed look like a config change and be refused.
_VOLATILE = {"paths", "run_name", "run_suffix", "config_name", "_config_model_name"}


def _drop_none(node):
    """A null option means "not set", which is the same as the key not being there. Dropping
    these keeps runs finished before an option existed from looking like a config change."""
    if isinstance(node, dict):
        return {k: _drop_none(v) for k, v in node.items() if v is not None}
    return node


def training_signature(cfg: dict) -> dict:
    """What a run's results actually depend on. Compared against the stored config.yaml to
    refuse resuming a run whose hyper-parameters changed -- so it must ignore every key the
    model does not read, or unrelated edits to base.yaml would invalidate finished runs."""
    sig = _drop_none(copy.deepcopy({k: v for k, v in cfg.items() if k not in _VOLATILE}))
    sig.pop("checkpoint", None)                          # saving weights cannot change them
    m, t = sig.get("model", {}), sig.get("training", {})

    # Options an inactive feature leaves unread. Without this, adding model.head defaulting to
    # "linear" and its five moe_* companions changed the signature of every run finished before
    # the MoE heads existed -- including the TF-IDF ones, which never look at a head at all --
    # and train.py refused to re-run any of them.
    if m.get("head", "linear") == "linear":
        m.pop("head", None)
    if m.get("head") != "sparse_moe":
        m.pop("moe_top_k", None)
        t.pop("moe_aux_weight", None)
    if m.get("head") != "soft_moe":
        m.pop("moe_slots", None)
    if "head" not in m:                                  # linear reads none of them
        for k in ("moe_experts", "moe_expert_dim", "moe_dropout"):
            m.pop(k, None)
    if m.get("head") != "mlp":
        m.pop("mlp_dims", None)
        m.pop("mlp_dropout", None)
    if not m.get("hybrid"):                              # same for the hybrid branch's knobs
        for k in ("hybrid_dim", "hybrid_dropout", "hybrid_lr", "hybrid_max_features",
                  "hybrid_phonetic"):
            m.pop(k, None)
    if m.get("layers") != "mix":
        t.pop("layer_mix_lr", None)

    if m.get("type") == "tfidf":
        sig.pop("training", None)                        # the transformer block is unused here
        sig.get("data", {}).pop("max_len", None)         # tokenizer-only setting
        return sig
    t.pop("num_workers", None)
    t.pop("eval_batch_size", None)
    return sig
