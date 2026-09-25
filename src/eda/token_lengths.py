"""Token-length distribution per transformer tokenizer -> docs/eda/figures/06-08 + stats/08.

  python -m src.eda.token_lengths

Measured on the cleaned text the models actually read (data/processed), every file of both
tasks, duplicates dropped. Counts include the special tokens ([CLS]/[SEP], <s></s>), because
that is what data.max_len caps.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

import src.utils.hf_quiet  # noqa: F401
from transformers import AutoTokenizer

# Style copied from src/eda/eda.py rather than imported: that module runs the whole EDA at import.
ROOT = Path(__file__).resolve().parents[2]
PROC, FIG, ST = ROOT / "data" / "processed", ROOT / "docs" / "eda" / "figures", ROOT / "docs" / "eda" / "stats"
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)
# (label, checkpoint, max_len used by its config). cnerg_xlmr shares XLM-R's tokenizer exactly.
MODELS = [("muril", "google/muril-base-cased", 96),
          ("cnerg_muril", "Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL", 96),
          ("mBERT", "bert-base-multilingual-cased", 96),
          ("XLM-R / cnerg_xlmr", "xlm-roberta-base", 96),
          ("mDeBERTa-v3", "microsoft/mdeberta-v3-base", 96),
          ("ModernBERT", "answerdotai/ModernBERT-base", 96),
          ("CANINE (ky tu)", "google/canine-s", 256)]
# reference categorical order, slots 1-6 (validated: CVD + normal-vision pass on light)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def lengths():
    frames = [pd.read_csv(PROC / f, keep_default_na=False)
              for f in ("a_train.csv", "a_val.csv", "b_train.csv", "b_val.csv")]
    texts = pd.concat([f.text for f in frames]).drop_duplicates().tolist()
    out = {}
    for tag, name, _ in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        out[tag] = np.array([len(x) for x in tok(texts)["input_ids"]])
    return texts, out


def fig_histograms(L):
    """Small multiples: one panel per tokenizer, same x range for the subword ones."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 11))
    axes = axes.ravel()
    for ax, (tag, _, ml) in zip(axes, MODELS):
        x = L[tag]
        cap = 400 if ml > 96 else 160               # CANINE counts characters
        bins = np.arange(0, cap + 1, 4 if ml <= 96 else 10)
        ax.hist(np.clip(x, 0, cap), bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
        ax.axvline(ml, color=ORANGE, linewidth=2)
        cut = (x > ml).mean()
        ymax = ax.get_ylim()[1]
        ax.text(ml + cap * 0.02, ymax * 0.92, f"max_len {ml}\n{cut:.1%} cau bi cat",
                color=INK, fontsize=9, va="top")
        ax.set_title(f"{tag}   median {np.median(x):.0f} | p95 {np.percentile(x, 95):.0f} | max {x.max()}",
                     fontsize=10)
        ax.set_xlim(0, cap)
        ax.set_xlabel("so ky tu / cau" if ml > 96 else "so token / cau (gom token dac biet)")
        ax.set_ylabel("so cau")
    axes[-1].axis("off")
    axes[-1].text(0, 0.6, f"{len(next(iter(L.values()))):,} cau khac nhau\n"
                          "(train + dev cua task A va B, sau clean_text, bo trung)\n\n"
                          "Cot cuoi moi panel gom moi cau dai hon truc x.\n"
                          "Duong cam = max_len cua config; phan ben phai bi cat.",
                  color=INK2, fontsize=10, va="top")
    fig.suptitle("Phan bo so token moi cau theo tokenizer", x=0.01, ha="left",
                 fontweight="bold", color=INK, fontsize=13)
    fig.tight_layout()
    save(fig, "06_token_length_by_model.png")


def fig_truncation(L):
    """% of comments cut, as a function of max_len -- the curve to pick max_len from."""
    fig, ax = plt.subplots(figsize=(9, 5))
    grid = np.arange(32, 257, 2)
    subword = [m for m in MODELS if m[2] <= 96]
    pct = lambda tag, g: 100 * (L[tag] > g).mean()
    # The curves nearly coincide, so identity comes from a legend that also carries the two
    # numbers worth reading off, in the order the curves sit (worst first).
    order = sorted(zip(subword, SERIES), key=lambda mc: -pct(mc[0][0], 96))
    for (tag, _, _), col in order:
        ax.plot(grid, [pct(tag, g) for g in grid], color=col, linewidth=2,
                label=f"{tag:20s} {pct(tag, 96):.1f}% @96   {pct(tag, 128):.1f}% @128")
    for g, name in ((96, "96 = hien tai"), (128, "128")):
        ax.axvline(g, color=INK2, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
        ax.text(g + 2, 7.6, name, color=INK2, fontsize=9)
    ax.set_ylim(0, 8)
    ax.set_xlim(32, 256)
    ax.set_xlabel("max_len (token)")
    ax.set_ylabel("% cau bi cat")
    ax.set_title("Ty le cau bi cat theo max_len (tokenizer subword)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, title="% cau bi cat",
                    title_fontsize=9, prop={"family": "monospace", "size": 9})
    leg.get_title().set_color(INK2)
    save(fig, "07_truncation_vs_max_len.png")


def fig_by_label(texts, L):
    """Where the truncation lands: share of comments over 96 MuRIL tokens, per label."""
    rows = []
    for task, f in (("A", "a_train.csv"), ("B", "b_train.csv")):
        d = pd.read_csv(PROC / f, keep_default_na=False)
        idx = {t: i for i, t in enumerate(texts)}
        n = L["muril"][[idx[t] for t in d.text]]
        for lab, g in pd.Series(n).groupby(d.label.values):
            rows.append((f"{task}: {lab}", 100 * (g > 96).mean(), len(g)))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b = ax.barh([r[0] for r in rows], [r[1] for r in rows], color=BLUE, height=0.6,
                edgecolor="white", linewidth=2)
    ax.bar_label(b, labels=[f" {r[1]:.1f}%  (n={r[2]:,})" for r in rows], color=INK, fontsize=9)
    ax.set_xlabel("% cau > 96 token (MuRIL) -> bi cat")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.35)
    ax.set_title("Cau bi cat roi vao lop nao (tap train)")
    save(fig, "08_truncation_by_label.png")


def stats(L):
    rows = []
    for tag, _, ml in MODELS:
        x = L[tag]
        rows.append({"model": tag, "max_len": ml, "mean": x.mean(), "median": np.median(x),
                     "p90": np.percentile(x, 90), "p95": np.percentile(x, 95),
                     "p99": np.percentile(x, 99), "max": x.max(),
                     "pct_truncated": 100 * (x > ml).mean(), "n_truncated": int((x > ml).sum()),
                     "pct_tokens_lost": 100 * np.clip(x - ml, 0, None).sum() / x.sum()})
    pd.DataFrame(rows).round(2).to_csv(ST / "08_token_lengths.csv", index=False)


def main():
    texts, L = lengths()
    fig_histograms(L)
    fig_truncation(L)
    fig_by_label(texts, L)
    stats(L)
    print(f"-> {FIG}/06-08_*.png, {ST}/08_token_lengths.csv")


if __name__ == "__main__":
    main()
