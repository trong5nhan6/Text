#!/usr/bin/env python3
"""
HASTIKA @ ICON-2026 — Exploratory Data Analysis
Usage (from repo root):  python -m src.eda.eda
Outputs:  docs/eda/figures/*.png, docs/eda/stats/*.csv|json  (the report is docs/eda/README.md)
Requires: pandas, numpy, matplotlib, scikit-learn, ftfy
"""
import html, json, re, unicodedata
from collections import Counter
from pathlib import Path

import ftfy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data" / "raw", ROOT / "docs" / "eda"
FIG, ST = OUT / "figures", OUT / "stats"
for d in (FIG, ST):
    d.mkdir(parents=True, exist_ok=True)

# ---------- style (reference palette, light) ----------
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
LAB_A = {"Hate": ORANGE, "Non-Hate": BLUE}
CATS = ["Gender", "Political", "Others", "Religion", "Violence", "Geo-political"]

def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig)

# ---------- load ----------
bt = pd.read_csv(DATA / "binary_train.csv")
bv = pd.read_csv(DATA / "binary_validation_inputs.csv")
mt = pd.read_csv(DATA / "multiclass_train.csv").rename(columns={"Hate Category": "Label"})
mv = pd.read_csv(DATA / "multiclass_validation_inputs.csv")
files = {"binary_train": bt, "binary_val": bv, "multiclass_train": mt, "multiclass_val": mv}
report = {}

# ---------- cleaning ----------
MOJI = re.compile(r"à²|à³|ð[\x80-\xBF]|â€|Ã")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
KAN = re.compile(r"[ಀ-೿]")
URL = re.compile(r"https?://\S+|www\.\S+")

def clean(t: str) -> str:
    t = ftfy.fix_text(str(t))
    t = html.unescape(t)
    t = re.sub(r"<br\s*/?>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = URL.sub(" URL ", t)
    t = unicodedata.normalize("NFC", t)
    return re.sub(r"\s+", " ", t).strip()

def norm(t: str) -> str:
    """aggressive key for duplicate detection"""
    t = EMOJI.sub("", t.lower())
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)          # loooool -> lool
    return re.sub(r"[^\w\s]", "", t).strip()

for name, df in files.items():
    df["raw"] = df["Comment"].astype(str)
    df["text"] = df["raw"].map(clean)
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()
    df["mojibake"] = df["raw"].str.contains(MOJI)
    df["has_html"] = df["raw"].str.contains(r"<br|&[a-z]+;|&#\d+;", regex=True)
    df["has_url"] = df["raw"].str.contains(URL)
    df["has_emoji"] = df["text"].str.contains(EMOJI)
    df["has_kannada"] = df["text"].str.contains(KAN)
    df["upper_ratio"] = df["text"].map(lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s)))
    df["key"] = df["text"].map(norm)
    # cleaned CSVs are not written here: src/data/preprocessing.py owns data/processed/

# ---------- 1. overview / quality ----------
rows = []
for name, df in files.items():
    rows.append({
        "file": name, "rows": len(df), "null_text": int(df["Comment"].isna().sum()),
        "dup_id": int(df["id"].duplicated().sum()),
        "dup_text_exact": int(df["raw"].duplicated().sum()),
        "dup_text_normalized": int(df["key"].duplicated().sum()),
        "mojibake_rows": int(df.mojibake.sum()), "html_rows": int(df.has_html.sum()),
        "url_rows": int(df.has_url.sum()), "emoji_rows": int(df.has_emoji.sum()),
        "kannada_script_rows": int(df.has_kannada.sum()),
        "empty_after_clean": int((df["text"] == "").sum()),
        "words_median": float(df.n_words.median()), "words_p95": float(df.n_words.quantile(.95)),
        "words_max": int(df.n_words.max()),
    })
quality = pd.DataFrame(rows); quality.to_csv(ST / "01_quality_overview.csv", index=False)

# label conflicts among duplicates
def conflicts(df):
    g = df.groupby("key")["Label"].agg(["nunique", "count", lambda s: " | ".join(sorted(set(s)))])
    g.columns = ["n_labels", "n_rows", "labels"]
    return g[g.n_labels > 1].reset_index()
ca, cb = conflicts(bt), conflicts(mt)
pd.concat([ca.assign(task="A"), cb.assign(task="B")]).to_csv(ST / "02_duplicate_label_conflicts.csv", index=False)
report["dup_conflicts"] = {"A": len(ca), "B": len(cb)}

# train/val text overlap (near-duplicate leakage for local validation)
report["val_text_seen_in_train"] = {
    "A": int(bv.key.isin(set(bt.key)).sum()), "B": int(mv.key.isin(set(mt.key)).sum())}

# ---------- 2. label distribution ----------
va = bt.Label.value_counts(); vb = mt.Label.value_counts().reindex(CATS)
report["label_A"] = va.to_dict(); report["label_B"] = vb.to_dict()
report["imbalance_ratio_B"] = round(vb.max() / vb.min(), 2)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 2.2]})
ax = axes[0]
b = ax.bar(va.index, va.values, color=[LAB_A[k] for k in va.index], width=.55)
ax.bar_label(b, labels=[f"{v:,}\n({v/va.sum():.1%})" for v in va.values], color=INK, fontsize=9)
ax.set_title("Task A — nhãn (train)"); ax.set_ylim(0, va.max() * 1.25); ax.grid(axis="x", visible=False)
ax = axes[1]
b = ax.barh(vb.index[::-1], vb.values[::-1], color=BLUE, height=.6)
ax.bar_label(b, labels=[f" {v:,} ({v/vb.sum():.1%})" for v in vb.values[::-1]], color=INK, fontsize=9)
ax.set_title("Task B — hate category (train)"); ax.set_xlim(0, vb.max() * 1.25); ax.grid(axis="y", visible=False)
save(fig, "01_label_distribution.png")

# ---------- 3. length ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axes[0]
bins = np.arange(0, 61, 2)
for lab in ["Non-Hate", "Hate"]:
    ax.hist(bt.loc[bt.Label == lab, "n_words"].clip(upper=60), bins=bins, histtype="step",
            linewidth=2, color=LAB_A[lab], label=lab, density=True)
ax.set_title("Task A — số từ / comment (cắt ở 60)"); ax.set_xlabel("số từ"); ax.set_ylabel("mật độ"); ax.legend(frameon=False)
ax = axes[1]
data = [mt.loc[mt.Label == c, "n_words"] for c in CATS]
bp = ax.boxplot(data, vert=False, tick_labels=CATS, showfliers=False, widths=.5, patch_artist=True,
                medianprops={"color": INK, "linewidth": 2})
for p in bp["boxes"]: p.set(facecolor="#d6e6f8", edgecolor=BLUE)
ax.invert_yaxis(); ax.set_title("Task B — số từ theo category (không outlier)"); ax.set_xlabel("số từ")
save(fig, "02_length_distribution.png")
len_stats = pd.concat([
    bt.groupby("Label").n_words.describe(percentiles=[.5, .9, .95]).assign(task="A"),
    mt.groupby("Label").n_words.describe(percentiles=[.5, .9, .95]).assign(task="B")]).round(1)
len_stats.to_csv(ST / "03_length_by_label.csv")

# ---------- 4. surface features per label ----------
feat_cols = ["has_emoji", "mojibake", "has_kannada", "has_url"]
fa = bt.groupby("Label")[feat_cols + ["upper_ratio"]].mean().round(3)
fb = mt.groupby("Label")[feat_cols + ["upper_ratio"]].mean().reindex(CATS).round(3)
pd.concat([fa.assign(task="A"), fb.assign(task="B")]).to_csv(ST / "04_surface_features_by_label.csv")

# emoji per label
def top_emoji(s, k=12):
    return Counter(e for t in s for e in EMOJI.findall(t) if e != "️").most_common(k)
emo = {lab: top_emoji(bt.loc[bt.Label == lab, "text"]) for lab in ["Hate", "Non-Hate"]}
json.dump(emo, open(ST / "05_top_emoji.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- 5. vocabulary ----------
TOK = re.compile(r"[a-zಀ-೿]{2,}")
def toks(t): return TOK.findall(re.sub(r"(.)\1{2,}", r"\1\1", t.lower()))
bt["toks"] = bt.text.map(toks); mt["toks"] = mt.text.map(toks); bv["toks"] = bv.text.map(toks)
vocab = Counter(w for ts in bt.toks for w in ts)
hapax = sum(1 for c in vocab.values() if c == 1)
val_tokens = [w for ts in bv.toks for w in ts]
report["vocab"] = {"train_types": len(vocab), "hapax_ratio": round(hapax / len(vocab), 3),
                   "val_OOV_token_rate": round(sum(w not in vocab for w in val_tokens) / len(val_tokens), 3)}

# spelling variants example: words sharing a consonant skeleton
def skel(w): return re.sub(r"[aeiouy]+", "", w)
groups = {}
for w, c in vocab.items():
    if c >= 3 and len(w) >= 4: groups.setdefault(skel(w), []).append((w, c))
variants = sorted([(k, sorted(v, key=lambda x: -x[1])) for k, v in groups.items() if len(v) >= 3],
                  key=lambda x: -sum(c for _, c in x[1]))[:25]
pd.DataFrame([{"skeleton": k, "variants": ", ".join(f"{w}({c})" for w, c in v)} for k, v in variants]) \
    .to_csv(ST / "06_spelling_variants.csv", index=False)

# discriminative tokens: smoothed log-odds (Monroe et al. 2008, informative Dirichlet prior)
def log_odds(pos_docs, neg_docs, min_count=5, k=15):
    a = Counter(w for ts in pos_docs for w in ts); b = Counter(w for ts in neg_docs for w in ts)
    prior = a + b; a0, b0, p0 = sum(a.values()), sum(b.values()), sum(prior.values())
    out = []
    for w, pc in prior.items():
        if pc < min_count: continue
        ya, yb = a[w], b[w]
        d = np.log((ya + pc) / (a0 + p0 - ya - pc)) - np.log((yb + pc) / (b0 + p0 - yb - pc))
        var = 1 / (ya + pc) + 1 / (yb + pc)
        out.append((w, d / np.sqrt(var), ya, yb))
    return sorted(out, key=lambda x: -x[1])[:k]

disc = []
for w, z, ya, yb in log_odds(bt.loc[bt.Label == "Hate", "toks"], bt.loc[bt.Label == "Non-Hate", "toks"]):
    disc.append({"task": "A", "class": "Hate", "token": w, "z": round(z, 2), "in_class": ya, "in_rest": yb})
for w, z, ya, yb in log_odds(bt.loc[bt.Label == "Non-Hate", "toks"], bt.loc[bt.Label == "Hate", "toks"]):
    disc.append({"task": "A", "class": "Non-Hate", "token": w, "z": round(z, 2), "in_class": ya, "in_rest": yb})
for c in CATS:
    for w, z, ya, yb in log_odds(mt.loc[mt.Label == c, "toks"], mt.loc[mt.Label != c, "toks"], min_count=4):
        disc.append({"task": "B", "class": c, "token": w, "z": round(z, 2), "in_class": ya, "in_rest": yb})
disc = pd.DataFrame(disc); disc.to_csv(ST / "07_discriminative_tokens.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, c in zip(axes.flat, CATS):
    d = disc[(disc.task == "B") & (disc["class"] == c)].head(10)[::-1]
    ax.barh(d.token, d.z, color=BLUE, height=.6)
    ax.set_title(c); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=9)
fig.suptitle("Task B — token đặc trưng nhất mỗi category (log-odds z-score)", x=.01, ha="left",
             fontweight="bold", color=INK)
fig.tight_layout(); save(fig, "03_taskB_top_tokens.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, lab in zip(axes, ["Hate", "Non-Hate"]):
    d = disc[(disc.task == "A") & (disc["class"] == lab)].head(15)[::-1]
    ax.barh(d.token, d.z, color=LAB_A[lab], height=.6); ax.set_title(f"Task A — token đặc trưng: {lab}")
    ax.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "04_taskA_top_tokens.png")

# ---------- 6. id structure & cross-file overlap ----------
B, BV, M, MV = map(lambda d: set(d.id), (bt, bv, mt, mv))
hate_ids = set(bt.loc[bt.Label == "Hate", "id"])
overlap = {
    "id_range": [int(min(B | BV | M | MV)), int(max(B | BV | M | MV))],
    "unique_ids_all_files": len(B | BV | M | MV),
    "A_train∩A_val": len(B & BV),
    "B_train∩B_val": len(M & MV),
    "B_train ids in A_train": len(M & B), "…of which Hate": len(M & hate_ids),
    "B_train ids in A_val": len(M & BV),
    "B_val ids in A_train": len(MV & B), "B_val ids in A_val": len(MV & BV),
    "B ids not in any A file (likely A-test)": len((M | MV) - B - BV),
    "A_train Hate without B label": len(hate_ids - M - MV),
    "text identical for shared ids": bool((mt.merge(bt, on="id").eval("raw_x == raw_y")).all()),
}
report["id_overlap"] = overlap

# ---------- 7. quick reference baseline (TF-IDF char+word + LR, 5-fold) ----------
def tfidf_lr(balanced):
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True,
                                 token_pattern=r"(?u)\b\w+\b|[\U0001F000-\U0001FAFF]")),
    ])
    return make_pipeline(feats, LogisticRegression(C=4, max_iter=3000,
                                                   class_weight="balanced" if balanced else None))

skf = StratifiedKFold(5, shuffle=True, random_state=42)
base = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.6]})
for ax, (task, df, labels) in zip(axes, [("A", bt, ["Hate", "Non-Hate"]), ("B", mt, CATS)]):
    pred = cross_val_predict(tfidf_lr(task == "B"), df.text, df.Label, cv=skf, n_jobs=-1)
    base[task] = {"macro_f1": round(f1_score(df.Label, pred, average="macro"), 4),
                  "accuracy": round((pred == df.Label).mean(), 4),
                  "per_class": classification_report(df.Label, pred, labels=labels, output_dict=True, zero_division=0)}
    cm = confusion_matrix(df.Label, pred, labels=labels, normalize="true")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if cm[i, j] > .55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right"); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("dự đoán"); ax.set_ylabel("thực tế"); ax.grid(False)
    ax.set_title(f"Task {task} — TF-IDF+LR 5-fold  (macro-F1 {base[task]['macro_f1']:.3f})")
fig.tight_layout(); save(fig, "05_baseline_confusion.png")
report["tfidf_lr_5fold"] = base

json.dump(report, open(ST / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(quality.to_string(index=False))
print(json.dumps({k: v for k, v in report.items() if k != "tfidf_lr_5fold"}, ensure_ascii=False, indent=1, default=str))
for t, r in base.items():
    print(t, r["macro_f1"], r["accuracy"],
          {k: round(v["f1-score"], 3) for k, v in r["per_class"].items() if isinstance(v, dict)})
