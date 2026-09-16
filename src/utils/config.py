"""YAML config loading with `_base_` inheritance and `key.sub=value` CLI overrides."""
import copy
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
    if isinstance(val, str):                 # YAML 1.1 reads "1e-3" as a string
        try:
            val = float(val)
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
    loss = cfg["training"].get("loss", "auto")
    return ("ce" if cfg["task"] == "a" else "wce") if loss == "auto" else loss


def run_name(cfg: dict) -> str:
    if cfg.get("run_name"):
        return cfg["run_name"]
    if cfg["model"]["type"] == "tfidf":
        return f"{cfg['config_name']}_{cfg['model'].get('clf', 'lr')}"
    return f"{cfg['config_name']}_{resolve_loss(cfg)}_s{cfg['seed']}"


def dump(cfg: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(path, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)


# keys that do not change the model's results -> ignored when checking a resumed run
_VOLATILE = {"paths", "run_name", "config_name"}


def training_signature(cfg: dict) -> dict:
    sig = {k: v for k, v in cfg.items() if k not in _VOLATILE}
    sig = copy.deepcopy(sig)
    sig.get("training", {}).pop("num_workers", None)
    sig.get("training", {}).pop("eval_batch_size", None)
    sig.pop("checkpoint", None)
    return sig
