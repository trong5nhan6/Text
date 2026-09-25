#!/usr/bin/env python3
"""
Regenerate notebooks/hastika_kaggle.ipynb.

The notebook clones this repository on Kaggle and runs its entrypoints, so it holds
no copy of the source: push a change, re-run the first cell, and the notebook picks
it up. Only the cell text below lives here.

  python notebooks/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "trong5nhan6/Text"
BRANCH = "main"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


# Knobs offered in the notebook's OVERRIDES cell. The values shown there are read out of
# configs/base.yaml at build time, so the cell can never claim a default the config does not have.
KNOBS = [
    ("--- huan luyen ---", None),
    ("training.epochs", "tran, khong phai muc tieu; cung la do dai lich LR"),
    ("training.early_stopping_patience", "dung khi macro-F1 khong cai thien bay nhieu epoch lien"),
    ("training.lr", "learning rate cua backbone (head dung head_lr)"),
    ("training.llrd", "layer-wise LR decay: null = lr phang | 0.9 = moi block duoi giam 0.9 lan"),
    ("training.layer_mix_lr", "CHI model.layers=mix: lr cua 13 logit softmax chon tang"),
    ("training.batch_size", "voi nhieu GPU day la TONG, chia deu cho cac GPU"),
    ("training.single_gpu", "null = dung moi GPU; model >1B tu dong ve 1 GPU | true = ep 1"),
    ("training.grad_checkpointing", "true = it bo nho hon ~sqrt(tang), cham hon ~30%; cho 7B"),
    ("training.grad_accum", "tang len khi giam batch_size, de giu batch hieu dung"),
    ("training.loss", "auto | ce | wce | focal   (auto: task a -> ce, task b -> focal)"),
    ("training.label_smoothing", "ap dung cho ca ce, wce va focal"),
    ("training.fgm", "adversarial training, 1.0 la chuan; KHONG dung duoc voi LoRA/freeze_emb"),
    ("training.ema", "trung binh truot trong so, 0.999 la chuan; gan nhu mien phi"),
    ("training.moe_aux_weight", "CHI sparse_moe: trong so load-balancing; 0 -> router sup ve 1 expert"),
    ("--- du lieu ---", None),
    ("data.text_type", "latin | kn = chu Kannada | en = ban dich may | both = latin+kn"),
    ("data.tta", "CHI transformer: infer ca 2 chu viet roi trung binh (tfidf se bi tu choi)"),
    ("data.max_len", "Religion / Geo-political dai hon, hay bi cat o 96"),
    ("data.use_valdataset", "false = train 100% du lieu, KHONG cham diem duoc -> chi cho ban nop cuoi"),
    ("--- model ---", None),
    ("model.name", "DE len MOI config trong CONFIGS -> chi bo # khi chay dung 1 config"),
    ("model.pooling", "cls | mean | last  (LLM giai ma BAT BUOC dung last)"),
    ("model.dtype", "null = fp32 nhu cu | fp16/bf16: BAT BUOC cho LLM, fp32 se OOM"),
    ("model.dropout", None),
    ("model.multisample_dropout", "null/1 = nhu cu | 4-8 = trung binh head qua nhieu mat na"),
    ("model.unfreeze_last_n_blocks", "null = train tat ca | 4 = chi 4 khoi cuoi | 0 = chi head"),
    ("model.freeze_embeddings", "null = theo khoa tren | true | false"),
    ("model.layers", "null = tang cuoi | [2,12] = noi 2 tang | mix = tong co trong so hoc duoc"),
    ("model.head", "linear | mlp | sparse_moe | soft_moe  (soft_moe bo qua model.pooling)"),
    ("model.mlp_dims", "CHI head=mlp: cac tang an, vd [512] | [512, 128]"),
    ("model.mlp_dropout", "CHI head=mlp: dropout sau moi tang an"),
    ("model.moe_experts", "so expert; head linear chi co 4.614 tham so, MoE E=4 d=64 la ~201k"),
    ("model.moe_expert_dim", None),
    ("model.moe_top_k", "CHI sparse_moe: so expert hoat dong moi mau (1 = Switch routing)"),
    ("model.moe_slots", "CHI soft_moe: so slot moi expert"),
    ("model.moe_dropout", "ben trong moi expert"),
    ("model.hybrid", "null | tfidf = ghep TF-IDF (+phien am) vao vector pooled; ten run them _hyb"),
    ("model.hybrid_max_features", "CHI hybrid: SO CHIEU TF-IDF moi vectoriser; tong = 4x (2x neu tat phien am)"),
    ("model.hybrid_phonetic", "CHI hybrid: true = them view phien am (4 vectoriser), false = 2"),
    ("model.hybrid_dim", "CHI hybrid: so chieu chieu vector TF-IDF xuong truoc khi ghep"),
    ("model.hybrid_dropout", "CHI hybrid: dropout tren vector TF-IDF dau vao"),
    ("model.hybrid_lr", "CHI hybrid: lr cua nhanh TF-IDF (khoi tao moi, can lon hon head_lr)"),
    ("model.side_embedding", "BAT/TAT: null | char | phonetic | char+phonetic -> tron vao embedding dau vao; ten run them _se-..."),
    ("model.side_char_dim", "CHI side: so chieu embedding moi ky tu"),
    ("model.side_char_filters", "CHI side: filter moi do rong CNN (2,3,4,5) -> vector ky tu 4x"),
    ("model.side_key_dim", "CHI side: so chieu embedding khoa phien am"),
    ("model.side_min_count", "CHI side: khoa phien am hiem hon -> dung chung UNK"),
    ("model.side_lr", "CHI side: lr cua nhanh phu + gate"),
    ("seed", None),
    ("--- dia ---", None),
    ("checkpoint.save", "best | none   (none tiet kiem ~0.5 GB moi run)"),
]


# Knobs rendered uncommented, i.e. already in OVERRIDES. They sit at their base.yaml defaults, so
# the cell reports "khong doi gi" until one is edited -- then RUN_SUFFIX becomes mandatory.
ACTIVE = {"model.head", "model.mlp_dims", "model.mlp_dropout", "model.hybrid", "model.hybrid_max_features", "model.hybrid_phonetic",
          "model.hybrid_dim", "model.hybrid_dropout", "model.hybrid_lr",
          "model.side_embedding", "model.side_char_dim", "model.side_char_filters",
          "model.side_key_dim", "model.side_min_count", "model.side_lr"}


# Values the notebook ships with instead of the base.yaml default -- choices made in the
# notebook itself, kept here so rebuilding it does not silently revert them.
NOTEBOOK_VALUES = {"model.hybrid": "tfidf"}


def _overrides_cell() -> str:
    import yaml
    base = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))

    def default_of(dotted):
        node = base
        for part in dotted.split("."):
            node = node[part]
        return node

    keys = [k for k, _ in KNOBS if not k.startswith("---")]
    width = max(len(k) for k in keys) + 4
    lines = ["OVERRIDES = {          # gia tri = mac dinh trong configs/base.yaml, sua roi moi co tac dung"]
    for key, note in KNOBS:
        if key.startswith("---"):
            lines.append(f"    # {key}")
            continue
        v = NOTEBOOK_VALUES.get(key, default_of(key))
        v = f"'{v}'" if isinstance(v, str) else repr(v)
        entry = (f"      '{key}':" if key in ACTIVE else f"    # '{key}':").ljust(width + 8) + f"{v},"
        lines.append(f"{entry.ljust(width + 20)}# {note}" if note else entry)
    lines.append("}")
    return "\n".join(lines) + """
# Checkpoint da fine-tune san tren Kannada code-mixed -- dung config rieng thi tot hon la
# doi model.name o tren, vi run se co ten rieng thay vi de len run cua muril/roberta:
#   configs/cnerg_muril.yaml   Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL   (goc muril)
#   configs/cnerg_xlmr.yaml    Hate-speech-CNERG/deoffxlmr-mono-kannada            (goc xlm-r)
#   configs/muril_large.yaml   google/muril-large-cased      24 block, ~505M
#   configs/roberta_large.yaml xlm-roberta-large             24 block, 561M

RUN_SUFFIX = ''        # vi du '_e8' -> run ten muril_focal_s42_e8. BAT BUOC khi doi gia tri that su.

# doi data.val_ratio / data.split_seed se chia lai split, moi ket qua cu se het so sanh duoc

import yaml
_base = yaml.safe_load(open('configs/base.yaml', encoding='utf-8'))
def _default_of(k):
    node = _base
    for part in k.split('.'):
        node = node[part]
    return node

changed = {k: v for k, v in OVERRIDES.items() if _default_of(k) != v}
# lists without spaces: "[512, 128]" would reach the shell as two arguments
_fmt = lambda v: str(v).replace(" ", "") if isinstance(v, (list, tuple)) else v
SET_ARGS = " ".join(f"{k}={_fmt(v)}" for k, v in OVERRIDES.items())
SET_ARGS = f"--set {SET_ARGS}" if SET_ARGS else ""
ARGS = SET_ARGS + (f" --run_suffix {RUN_SUFFIX}" if RUN_SUFFIX else "")

print("them vao lenh train:", ARGS or "(khong co, dung mac dinh)")
if changed:
    for k, v in changed.items():
        print(f"  doi that su: {k}  {_default_of(k)} -> {v}")
    if not RUN_SUFFIX:
        print("!! RUN_SUFFIX dang trong -> train.py se tu choi chay de khong de len run cu")
elif OVERRIDES:
    print("  (cac dong da bo # deu dang giu nguyen mac dinh -> khong doi gi)")"""


# ----------------------------------------------------------------- 0) setup
md(f"""# HASTIKA @ ICON-2026 — Kaggle training notebook

**Settings (panel bên phải):** Accelerator = **GPU T4 ×2** · Internet = **On**

Notebook này **không chứa code** — nó clone [`{REPO}`](https://github.com/{REPO}) rồi gọi các
entrypoint của repo. Sửa code ở máy → `git push` → chạy lại cell dưới đây là có bản mới,
**không cần upload lại notebook**.

Quy trình: 1) repo → 2) data → 3) TF-IDF → 4) transformers → 5) evaluate → 6) submission.
Kết quả nằm trong `/kaggle/working/repo/` và được giữ lại khi **Save Version (Save & Run All)**.""")

code(f'''import os, subprocess, sys

REPO, BRANCH, WORK = "{REPO}", "{BRANCH}", "/kaggle/working"

# Repo Public -> clone ẩn danh, không cần gì thêm.
# Nếu sau này đổi sang Private: Add-ons ▸ Secrets ▸ New secret, tên GH_TOKEN,
# giá trị = GitHub Personal Access Token (scope "repo"). Cell này tự dò.
TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
    print("dùng GH_TOKEN từ Kaggle Secrets")
except Exception:
    pass

url = f"https://{{TOKEN + '@' if TOKEN else ''}}github.com/{{REPO}}.git"
hide = (lambda s: s.replace(TOKEN, "***")) if TOKEN else (lambda s: s)   # không in token ra log

os.chdir(WORK)
cmd = (["git", "-C", "repo", "pull", "--ff-only"] if os.path.isdir("repo/.git")
       else ["git", "clone", "--depth", "1", "-b", BRANCH, url, "repo"])
r = subprocess.run(cmd, capture_output=True, text=True)
print(hide((r.stdout + r.stderr).strip()))
if r.returncode:
    raise SystemExit("git thất bại — kiểm tra Internet = On, repo Public (hoặc secret GH_TOKEN)")

os.chdir(f"{{WORK}}/repo"); sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())''')

code('''!pip -q install ftfy sentencepiece tiktoken
# Chi can khi chay configs/llm.yaml (LLM giai ma + LoRA):
# !pip -q install peft bitsandbytes accelerate
#   (Kaggle co torchao 0.10 ma peft doi >0.16 -- classifier.py tu vo hieu hoa phep kiem tra do,
#    khong can go torchao bang tay nua.)
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available())
# ModernBERT cần transformers >= 4.48:  !pip -q install -U "transformers>=4.48"''')

# ------------------------------------------------------------------ 1) data
md("""## 1) Dữ liệu

CSV của ban tổ chức nằm sẵn trong repo (`data/raw/`), nên chỉ cần làm sạch + chia split.
Một lát **90% fit / 10% chấm điểm** (`data.val_ratio`), phân tầng theo nhãn và cố định bởi
`data.split_seed` — mọi model dùng chung một lát nên blend được với nhau.""")
code('''!ls data/raw
!python -m src.data.preprocessing''')

md("""**Khi test phát hành (20/9)** — chọn 1 trong 2 cách rồi chạy lại `preprocessing`:""")
code('''# Cách 1 — test đã được push vào repo: chỉ cần chạy lại cell git pull ở trên.
# Cách 2 — upload test thành Kaggle Dataset rồi copy vào data/raw:
# !cp /kaggle/input/<ten-dataset>/*test*.csv data/raw/
# !python -m src.data.preprocessing''')

# ----------------------------------------------------------------- 2) tfidf
md("""## 2) B0 — TF-IDF (CPU, ~10 giây mỗi run)

`C` là tham số điều chuẩn của mô hình tuyến tính, theo nghĩa **nghịch đảo**: `C` nhỏ = phạt trọng số
mạnh = model đơn giản (dễ underfit); `C` lớn = ưu tiên khớp dữ liệu = dễ overfit. Ở đây đặc trưng
TF-IDF có tới 300k chiều trên vài nghìn dòng train, nên `C` là siêu tham số quan trọng nhất của B0.

`train.py` fit một model cho **mỗi giá trị trong lưới** rồi giữ cái có macro-F1 cao nhất trên lát
held-out. Giá trị được chọn hiện trong `results/metrics.csv`, ví dụ `tfidf-lr C=2`.

`model.clf` chọn bộ phân loại; mỗi cái có dải regularization riêng nên **không cần truyền lưới tay**:

| `clf` | Model | Tham số |
|---|---|---|
| `lr` | LogisticRegression | `C` |
| `svm` | LinearSVC | `C` |
| `ridge` | RidgeClassifier | `alpha` |
| `cnb` | ComplementNB | `alpha` |
| `sgd` | SGDClassifier (log loss, elasticnet) | `alpha` |

Chạy cả năm: mỗi run ~10 giây, và **blend của chúng hơn hẳn model đơn tốt nhất** — `cnb` tuy điểm
thấp nhưng chỉ đồng thuận 73–79% với nhóm tuyến tính nên đóng góp nhiều nhất cho ensemble.""")
code('''import itertools, time

CLFS       = ['lr', 'svm', 'sgd', 'mlp']   # da bo: 'ridge', 'cnb'
                                         # 'mlp' = TF-IDF + MLP (configs/tfidf_mlp.yaml), CHAM:
                                         #   ~4-8 phut MOI gia tri alpha x 3, bo di neu voi
TASKS_ML   = ['a', 'b']
TEXT_TYPES = ['latin']        # <- doi o day. them 'kn', 'both' de chay ca ba goc nhin
                              #    latin = chu Latin goc | kn = chu Kannada | both = ca hai
                              #    ['latin', 'kn', 'both'] -> 30 run, ~5 phut

combos = list(itertools.product(CLFS, TASKS_ML, TEXT_TYPES))
t0 = time.time()
for n, (c, t, tt) in enumerate(combos, 1):
    print("")
    print("-" * 72)
    print(f"[{n}/{len(combos)}]  clf = {c:6s} |  task = {t}  |  text_type = {tt:6s} |  "
          f"+{time.time() - t0:.0f}s")
    print("-" * 72, flush=True)
    if c == 'mlp':      # its own config: smaller max_features and an alpha grid for the MLP
        !python train.py --config configs/tfidf_mlp.yaml --task {t} --set data.text_type={tt}
    else:
        !python train.py --config configs/tfidf.yaml --task {t} --set model.clf={c} data.text_type={tt}
print("")
print(f"xong {len(combos)} run trong {time.time() - t0:.0f}s")''')

md("""**Bảng tổng kết + blend ngay** (macro-F1 trên lát held-out; blend dùng greedy forward selection).

> `--tag blend_ml` ghi vào `results/{task}/_blend_ml/`, **tách khỏi** `_blend/` của mục 4.
> Đây mới chỉ là blend của nhóm ML; blend cuối cùng (ML + transformer) nằm ở mục 4, và cell
> nộp bài ở mục 5 chỉ đọc `_blend/`. Nhờ vậy chạy lại cell này sau khi train transformer
> cũng không đè mất blend đầy đủ.""")
code('''import pandas as pd
d = pd.read_csv('results/metrics.csv')
display(d[d.run.str.startswith('tfidf')][['task', 'run', 'macro_f1', 'accuracy', 'model']]
        .sort_values(['task', 'macro_f1'], ascending=[True, False]))
!python evaluate.py --task a --optimize --tag blend_ml
!python evaluate.py --task b --optimize --tag blend_ml''')

code('''# Dò tham số mịn hơn quanh giá trị vừa chọn (nhớ --run_suffix, nếu không script từ chối chạy):
# !python train.py --config configs/tfidf.yaml --task b --set model.clf=ridge "model.param_grid=[1,2,3,5,8]" --run_suffix _fine
# Calibration sigmoid cho svm/ridge (chua co predict_proba that) — cham 5x, do tren du lieu nay KHONG giup:
# !python train.py --config configs/tfidf.yaml --task b --set model.clf=ridge model.calibrate=true --run_suffix _cal''')

# ---------------------------------------------------------- 3) transformers
md("""## 3) B1 — Transformers

### Config đang dùng

`configs/base.yaml` là mặc định chung; mỗi model chỉ ghi đè phần khác biệt qua `_base_: base.yaml`.
Cell dưới in ra **config đã gộp** — đúng những giá trị `train.py` sẽ chạy.""")
code('''import yaml
from src.utils.config import load_config, run_name

SHOW = 'muril'          # đổi để xem config khác: tfidf muril roberta indicbert bert deberta modernbert
SHOW_TASK = 'b'

print(f"--- configs/{SHOW}.yaml (phan ghi de) ".ljust(72, "-"))
print(open(f'configs/{SHOW}.yaml', encoding='utf-8').read())
cfg = load_config(f'configs/{SHOW}.yaml', task=SHOW_TASK)
print(f"--- config da gop, task={SHOW_TASK}, run se ten la '{run_name(cfg)}' ".ljust(72, "-"))
print(yaml.safe_dump({k: cfg[k] for k in ('seed', 'data', 'model', 'training', 'checkpoint')},
                     sort_keys=False, allow_unicode=True))''')

md("""### Chỉnh siêu tham số ngay ở đây

**Giá trị đang thấy trong `OVERRIDES` chính là mặc định trong `configs/base.yaml`** (sinh tự động
lúc build notebook), nên bỏ dấu `#` mà không sửa gì thì **không đổi gì cả**. Sửa giá trị rồi mới có
tác dụng — cell tự so với `base.yaml` lúc chạy và chỉ báo những khoá thật sự khác.

Nhớ đặt `RUN_SUFFIX` khi bạn đổi siêu tham số: run cũ và run mới sẽ có tên khác nhau nên không đè
lên nhau và so sánh được với nhau. Nếu để trống mà siêu tham số đã đổi, `train.py` sẽ **từ chối chạy**
thay vì âm thầm trộn kết quả.""")
code(_overrides_cell())

md("""### Train

`epochs: 6`, `batch_size: 32`, `lr: 2e-5` — công thức chuẩn khi fine-tune BERT trên vài nghìn dòng.
`early_stopping_patience: 5` cố ý ≥ `epochs`: chạy trọn lịch để LR kịp anneal về 0, rồi giữ trọng số
của epoch tốt nhất.

Trên T4 mỗi run ≈ **4–7 phút** (task A 180 step/epoch, task B 89 step/epoch).
6 config × 2 task ≈ 60–80 phút.

- Run đã xong → chạy lại sẽ bỏ qua, không train lại.
- Mỗi run lưu 1 checkpoint fp16 (~0.5 GB với model base); `/kaggle/working` giới hạn ~20 GB.

> **Vì sao 6 chứ không phải 30.** `trainer.py` tính `steps = step_mỗi_epoch × epochs`, warmup 10%,
> rồi LR giảm tuyến tính về 0 ở step cuối — nên `epochs` **vừa là trần vừa là độ dài lịch LR**.
> Lần chạy với `epochs: 30` cho `best_epoch` rơi vào 9–17 và điểm không hơn gì train 2 epoch:
> warmup ngốn 3 epoch đầu, LR tới epoch 6 vẫn còn ~89% đỉnh, và early stopping rốt cuộc chỉ nhặt
> **đỉnh nhiễu** trên lát eval 315–639 dòng. Ở 6 epoch, warmup 0,6 epoch và LR anneal đúng lúc
> model hội tụ.""")
code('''import time

CONFIGS = ['muril', 'roberta', 'bert', 'cnerg_muril', 'cnerg_xlmr']
# da bo bot cho nhanh: 'indicbert', 'deberta', 'modernbert'
# llm = LLM giai ma + LoRA (configs/llm.yaml). Can: pip install peft bitsandbytes accelerate.
#   Huong duy nhat con lai chua do ma co co so: 7 kien truc encoder deu 0,78-0,82, con thu
#   DUY NHAT lam diem nhay la MLM tren van ban dung mien (+0,134 task b) -- tuc rang buoc la
#   model da DOC bao nhieu Kanglish, khong phai no duoc noi day the nao.
# canine = model muc KY TU, khong co tu vung -> khong the OOV, khong the bi xe vun.
#   Them 'canine' vao list de chay. Luu y no dem KY TU: configs/canine.yaml
#   dat data.max_len=256 (cat 3,1% dong; o 128 la 13,5%).
# cnerg_* = checkpoint DA fine-tune san tren abusive/offensive Kannada code-mixed;
#   cung kien truc va cung tham so train voi muril/roberta, chi khac diem xuat phat.
# Ban large (24 block, ~500M): them 'muril_large' / 'roberta_large'. Cham hon ~3 lan va
# hay sup ve mot lop tren du lieu nho -- doc history.json truoc khi tin con so cuoi.
TASKS   = ['a', 'b']

# Bien the chay THEM cho moi config (de ['linear'] / [False] = chi ban goc nhu truoc):
HEADS   = ['linear']      # them 'mlp' -> head MLP (model.mlp_dims trong OVERRIDES), ten run them _mlp
HYBRID  = [False]         # them True  -> ghep TF-IDF vao vector pooled, ten run them _hyb
SIDE    = [None]          # them 'char' / 'phonetic' / 'char+phonetic' -> tron embedding phu vao
                          #   embedding dau vao (gate = 0 luc dau), ten run them _se-<kieu>
# vd HEADS = ['linear', 'mlp'], HYBRID = [False, True] -> 4 bien the x moi config x moi task
# None / False = KHONG them gi -> gia tri trong OVERRIDES van co hieu luc

import itertools
variants = list(itertools.product(HEADS, HYBRID, SIDE))
jobs = [(c, t, h, hy, se) for c in CONFIGS for (h, hy, se) in variants for t in TASKS]
t0 = time.time()
for n, (c, t, h, hy, se) in enumerate(jobs, 1):
    extra = ([f"model.head={h}"] + (["model.hybrid=tfidf"] if hy else [])
             + ([f"model.side_embedding={se}"] if se else []))
    # _mlp phan biet voi run head linear; _hyb do train.py tu them khi bat hybrid
    suffix = RUN_SUFFIX + ("_mlp" if h == "mlp" else "")
    cmd = f"{SET_ARGS} --set {' '.join(extra)}" + (f" --run_suffix {suffix}" if suffix else "")
    print("")
    print("=" * 72)
    print(f"[{n}/{len(jobs)}]  config = {c} | task = {t} | head = {h} | hybrid = {hy} | side = {se}   "
          f"|   {time.strftime('%H:%M:%S')}   |   +{(time.time() - t0) / 60:.1f} phut")
    print("=" * 72, flush=True)
    !python train.py --config configs/{c}.yaml --task {t} {cmd}
print("")
print(f"xong {len(jobs)} run trong {(time.time() - t0) / 60:.1f} phut")''')

md("""> **Loss:** vòng lặp trên đã dùng đúng loss cho từng task qua `training.loss: auto` —
> Task A `ce` (nhãn 51/49, không cần bù), Task B **`focal`** (gamma 2 + class weight `sqrt_inv`).
> Task B lệch **7,3 lần**: Gender 1.362 (43,1%) so với Geo-political 186 (5,9%), mà macro-F1 chấm
> cả 6 lớp ngang nhau. Focal hạ trọng số những mẫu đã dễ và dồn gradient vào mẫu khó.""")

code('''# Ví dụ biến thể:
# !python train.py --config configs/muril.yaml --task b --set training.focal_gamma=3 --run_suffix _g3
# !python train.py --config configs/muril.yaml --task b --set data.max_len=128 --run_name muril_len128
# !python train.py --config configs/muril.yaml --task b --set data.text_type=both data.tta=true
# !python train.py --config configs/roberta.yaml --task b --run_name xlmr_large --set model.name=xlm-roberta-large training.lr=1e-5 training.batch_size=16 training.grad_accum=2''')

code('''!du -sh checkpoints/*/* 2>/dev/null; df -h /kaggle/working | tail -1
# xoá run không cần:  !rm -rf checkpoints/b/<run_name> results/b/<run_name>''')

# -------------------------------------------------------------- 4) evaluate
md("""## 4) Evaluate

Mọi run đều chấm trên cùng một lát held-out (`n_eval` dòng) nên so sánh và blend được trực tiếp.""")
code('''import pandas as pd
display(pd.read_csv('results/metrics.csv'))
!python evaluate.py --task a
!python evaluate.py --task b''')

code('''# blend MOI run cua task do; trong so tim bang greedy forward selection tren lat eval
!python evaluate.py --task a --optimize
!python evaluate.py --task b --optimize''')

code('''from IPython.display import Image, display
for t in ('a', 'b'):
    p = f'results/{t}/_blend/confusion.png'
    if os.path.exists(p):
        print(p); display(Image(p))''')

# ------------------------------------------------------------ 5) submission
md("""## 5) Submission

### Train nhiều model cùng lúc thì file nằm đâu?

**Mỗi run một thư mục riêng, không cái nào đè cái nào.** Tên run mặc định là
`<config>_<loss>_s<seed>`, task là thư mục cha:

```
results/
├── a/                              Task A
│   ├── tfidf_lr/                   eval.npy  val.npy  [test.npy]  metrics.json  config.yaml
│   ├── tfidf_svm/
│   ├── muril_ce_s42/               ← configs/muril.yaml  --task a
│   └── roberta_ce_s42/             ← configs/roberta.yaml --task a
├── b/                              Task B  (loss mặc định là focal nên tên là _focal_)
│   ├── tfidf_lr/  tfidf_svm/  muril_focal_s42/  roberta_focal_s42/
│   ├── _blend/                     blend ĐẦY ĐỦ (mục 4) — cell nộp bài đọc file này
│   └── _blend_ml/                  blend riêng nhóm TF-IDF (mục 2), không ảnh hưởng bản nộp
└── metrics.csv                     1 dòng cho mỗi run, cả 2 task
checkpoints/{task}/{run}/           trọng số fp16 của run đó
```

### Ba tập dữ liệu, đừng nhầm

| File trong run | Là gì | Có nhãn? | Dùng để |
|---|---|---|---|
| `eval.npy` | lát held-out 10% **cắt ra từ train** | ✅ | chấm điểm nội bộ, chọn model, tìm trọng số blend |
| `val.npy` | `*_validation_inputs.csv` **của BTC** | ❌ | **nộp phase Development** |
| `test.npy` | `*_test_inputs.csv` (phát 20/9) | ❌ | **nộp phase Evaluation** |

Chữ "val" xuất hiện ở hai nghĩa khác nhau: lát held-out (có nhãn, để bạn tự chấm) và file
validation của BTC (không nhãn, để nộp). `eval.npy` là cái đầu, `val.npy` là cái sau.

### Nộp thế nào

Codabench có **2 leaderboard riêng biệt**, mỗi task nộp **một** `predictions.csv`. Bạn chỉ có
**20 lượt nộp**, nên đừng nộp từng model — chọn theo điểm held-out ở mục 4 rồi nộp bản tốt nhất.

- Run train *sau* khi có test → `--split test`
- Run train *trước* khi có test → mode 2 (`--checkpoints ... --input ...`), không cần train lại""")

md("""**Cách 1 — từng model riêng: đã có sẵn.** `train.py` tự sinh file nộp cho file validation của
BTC ngay sau khi train xong, đặt tên `{task}_val_{run}`. Cell này chỉ để xem lại danh sách:""")
code('''import glob, os
for z in sorted(glob.glob('results/submissions/*/submission.zip')):
    d = os.path.dirname(z)
    n = sum(1 for _ in open(f'{d}/predictions.csv', encoding='utf-8')) - 1
    print(f"{os.path.basename(d):45s} {n:5d} dong")''')

md("""**Cách 2 — ensemble** (thường tốt hơn). Cell dưới đọc thẳng trọng số mà `evaluate.py --optimize`
vừa tìm được ở mục 4 (`results/{task}/_blend/blend.json`), nên không phải chép tay tên run.""")
code('''import json
for t in ('a', 'b'):
    b = json.load(open(f'results/{t}/_blend/blend.json'))
    keep = [(r, w) for r, w in zip(b['runs'], b['weights']) if w > 0]
    print(f"task {t}: macro-F1 {b['macro_f1']:.4f} | " + ", ".join(f"{r}:{w:g}" for r, w in keep))
    runs = " ".join(r for r, _ in keep)
    ws   = " ".join(str(w) for _, w in keep)
    !python inference.py --task {t} --runs {runs} --weights {ws} --split val --tag ens

# test, tu checkpoint (run train truoc khi co test):
# !python inference.py --task b --checkpoints checkpoints/b/muril_focal_s42 --input data/raw/multiclass_test_inputs.csv --tag ens1''')

code('''import glob, os, shutil
out = '/kaggle/working/submissions'
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
for z in glob.glob('results/submissions/*/submission.zip'):
    shutil.copy(z, f"{out}/{os.path.basename(os.path.dirname(z))}.zip")
print(f"gom {len(os.listdir(out))} file nop")
!ls -la /kaggle/working/submissions''')

md("""Nén cả thư mục thành **một file** để tải về một lần:

> `submissions.zip` là để **tải về**, không phải để nộp — nó là zip chứa các zip. Codabench chỉ
> nhận zip phẳng chứa đúng một `predictions.csv`, tức là từng file `{task}_val_{run}.zip` bên trong.""")
code('''%cd /kaggle/working
!rm -f submissions.zip
!zip -r -q submissions.zip submissions
!ls -lh /kaggle/working/submissions.zip
!unzip -l submissions.zip | head -8
%cd /kaggle/working/repo''')

md("""**Nén cả `results/`** — nhat.npy, metrics.json, history.json, per_class.csv, confusion.png
cua moi run, cong metrics.csv. Day la thu ban can de phan tich o may va viet bai bao; `checkpoints/`
KHONG nam trong do (moi run ~0,5 GB).""")
code('''%cd /kaggle/working
!rm -f results.zip
# bo checkpoint va file nop (da co submissions.zip rieng) cho nhe
!zip -r -q results.zip repo/results -x "repo/results/submissions/*"
!ls -lh /kaggle/working/results.zip
!unzip -l results.zip | tail -5
import glob
print("so run co ket qua:", len(glob.glob('/kaggle/working/repo/results/*/*/metrics.json')))
%cd /kaggle/working/repo''')

def write(filename: str):
    nb = {"cells": list(cells),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / filename
    json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {out} ({len(cells)} cells)")
    cells.clear()


write("hastika_kaggle.ipynb")


# =====================================================================================
#  Notebook 2 -- build the transliteration cache. A one-off job, kept out of the training
#  notebook so that one stays quick to re-run.
# =====================================================================================
md(f"""# Sinh cache chuyển tự — Kanglish (chữ Latin) → chữ Kannada

**Chạy MỘT LẦN.** Kết quả là `data/xlit_kn.json`, commit vào repo; sau đó mọi máy chỉ đọc file
JSON — không cần cài IndicXlit, không cần GPU, không cần mạng.

**Settings (panel bên phải):** Accelerator = **GPU T4** · Internet = **On**

> **Vì sao phải chạy ở Kaggle:** IndicXlit phụ thuộc `fairseq`, mà `fairseq` **không build được
> trên Windows** — header của torch cần `/std:c++17` còn setup của fairseq không truyền cờ đó,
> nên dừng ở `error C2429: nested-namespace-definition`. Trên Linux thì cài bình thường.""")

code(f'''import os, subprocess, sys
REPO, BRANCH, WORK = "{REPO}", "{BRANCH}", "/kaggle/working"

TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except Exception:
    pass
url = f"https://{{TOKEN + '@' if TOKEN else ''}}github.com/{{REPO}}.git"
hide = (lambda s: s.replace(TOKEN, "***")) if TOKEN else (lambda s: s)

os.chdir(WORK)
cmd = (["git", "-C", "repo", "pull", "--ff-only"] if os.path.isdir("repo/.git")
       else ["git", "clone", "--depth", "1", "-b", BRANCH, url, "repo"])
r = subprocess.run(cmd, capture_output=True, text=True)
print(hide((r.stdout + r.stderr).strip()))
os.chdir(f"{{WORK}}/repo"); sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())''')

md("""## 1) Cài IndicXlit

Phải cài theo đúng thứ tự này, **không** cài thẳng `ai4bharat-transliteration`:

1. **`fairseq-fixed`** — bản vá của `fairseq` cho Python 3.11–3.12 (Kaggle đang dùng 3.12).
   `fairseq` gốc không build được ở đó.
2. **`--no-deps`** — `ai4bharat-transliteration` khai báo phụ thuộc `fairseq` **theo đúng tên
   đó**, nên pip sẽ cố cài bản gốc và chết, kể cả khi `fairseq-fixed` đã có. `--no-deps` cắt
   đường đó, đồng thời bỏ luôn `flask`, `gevent`, `tensorboardX` — thừa hoàn toàn với ta.
3. Cài tay đúng 4 gói nó thật sự cần lúc chạy.

`urduhack` cố tình **không** cài: nó chỉ dùng để chuẩn hoá chữ Shahmukhi (Urdu) nhưng lại kéo
theo **TensorFlow**, và còn hạ cấp `click` làm hỏng gói khác. `src/data/transliterate.py` đăng
ký sẵn một module giả mang tên đó trước khi import, nên nhánh Urdu không bao giờ chạy tới.""")
code('''!pip -q install ftfy
!pip -q install fairseq-fixed
!pip -q install --no-deps ai4bharat-transliteration
!pip -q install pydload indic-nlp-library ujson sacremoses

from src.data.transliterate import _stub_urduhack, _allow_fairseq_checkpoint
_stub_urduhack()              # chan truoc khi import, tranh keo theo TensorFlow
_allow_fairseq_checkpoint()   # torch 2.6+ mac dinh weights_only=True, fairseq chua biet dieu do

from ai4bharat.transliteration import XlitEngine
e = XlitEngine("kn", beam_width=4, src_script_type="roman")
print(e.translit_sentence("nin sule maga"))        # mong doi: {'kn': 'ನಿನ್ ಸುಲೇ ಮಗ'}''')

md("""> Hai dòng `_stub_urduhack()` / `_allow_fairseq_checkpoint()` ở trên chỉ vá cho **kernel của
> notebook**. Cell sinh cache chạy `!python -m ...` ở **tiến trình riêng**, nên hai hàm đó cũng
> được gọi sẵn bên trong `build_cache()` — không cần làm gì thêm.""")

md("""## 2) Sinh cache

Gom mọi comment trong `data/raw` (4 file, **7.565 câu duy nhất** sau khi làm sạch), chuyển tự
từng câu rồi ghi ra `data/xlit_kn.json`.

Khoá của cache là **text đã làm sạch** — đúng chuỗi mà `preprocessing.py` đặt vào cột `text`,
nên lúc tra là khớp tuyệt đối.

Script **lưu sau mỗi 500 câu**: session bị ngắt thì chạy lại sẽ tiếp tục từ chỗ dừng.""")
code('''!python -m src.data.transliterate --lang kn --beam 4''')

md("""## 3) Kiểm tra bằng mắt

Điều cần thấy: các **biến thể chính tả khác nhau của cùng một từ** phải cho ra **cùng một chuỗi
Kannada**. Đó chính là cơ chế làm giảm 25,6% OOV — nếu không thấy điều này thì chuyển tự sẽ
không giúp được gì.""")
code('''import json
from src.data.transliterate import CACHE

cache = json.load(open(CACHE, encoding='utf-8'))
print(f"{len(cache)} cau trong cache\\n")

# cac bien the cua cung mot tu co ve cung mot chuoi Kannada khong?
probe = ["channel", "chanela", "channele", "chaannel", "aadre", "adre", "adru", "sule", "sulee"]
for w in probe:
    outs = {v.split()[i] for k, v in cache.items()
            for i, t in enumerate(k.lower().split()) if t == w and i < len(v.split())}
    print(f"  {w:10s} -> {sorted(outs)[:3] if outs else '(khong gap)'}")

print("\\n--- vai cau day du ---")
for k, v in list(cache.items())[:5]:
    print(f"  {k[:55]:57s} -> {v[:55]}")''')

md("""## 4) Tải về rồi commit

Tải `xlit_kn.json` từ panel **Output** bên phải, đặt vào `data/` ở máy, rồi:

```bash
git add data/xlit_kn.json
git commit -m "Add Roman->Kannada transliteration cache"
git push
```

Sau đó bật bằng `--set data.transliterate=true`. **Khi file test phát hành (20/9)**, chạy lại
notebook này một lần nữa — nó chỉ chuyển tự phần câu mới.""")
code('''import shutil
from src.data.transliterate import CACHE

shutil.copy(CACHE, "/kaggle/working/xlit_kn.json")
print(f"tai ve: /kaggle/working/xlit_kn.json  ({CACHE.stat().st_size/1e6:.1f} MB)")''')

write("build_xlit_cache.ipynb")


# =====================================================================================
#  Notebook 3 -- build the English-translation cache (the TRAA arm). Also a one-off, and
#  it needs notebook 2's output, because MT models want native script, not romanised text.
# =====================================================================================
md(f"""# Sinh cache dịch máy — Kanglish → English

**Chạy MỘT LẦN**, và **chạy SAU** `build_xlit_cache.ipynb`. Kết quả là `data/mt_en.json`,
commit vào repo; sau đó mọi máy chỉ đọc JSON.

**Settings:** Accelerator = **GPU T4** · Internet = **On**

### Luồng xử lý

```
Kanglish (chữ Latin)  ──IndicXlit──►  chữ Kannada  ──NLLB-200──►  English
  "nin sule maga"                     "ನಿನ್ ಸುಲೇ ಮಗ"              "you are a bastard"
```

Model dịch **cần chữ bản địa**, không nhận chữ Latin — nên phải chuyển tự trước. Bài
*"Transliterate or translate?"* (Puranik et al., FIRE 2021) cũng làm đúng thế: họ dịch
**từ tập đã chuyển tự**.

### Cảnh báo trước khi chạy

Bài trên đo trên **chính tiếng Kannada** và thấy dịch **không giúp**:

| Kannada, weighted F1 | BERT | ULMFiT |
|---|---|---|
| Gốc (TRA) | 0,6040 | **0,6389** |
| Chuyển tự (TRAI) | 0,5831 | 0,6150 |
| **Dịch (TRAA)** | 0,6231 | 0,6031 |

Hai lý do cơ chế: dịch máy **làm dịu hoặc bỏ từ chửi** (`sule`, `nayi`, `thu` — đúng những
token mang tín hiệu Hate mạnh nhất theo EDA mục 6), và đầu vào code-mixed sai chính tả nằm
ngoài phân bố huấn luyện của model dịch.

Vẫn đáng chạy để **tự kiểm chứng trên dữ liệu của mình** và để có đủ ba nhánh TRA/TRAI/TRAA
cho paper.""")

code(f'''import os, subprocess, sys
REPO, BRANCH, WORK = "{REPO}", "{BRANCH}", "/kaggle/working"

TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except Exception:
    pass
url = f"https://{{TOKEN + '@' if TOKEN else ''}}github.com/{{REPO}}.git"
hide = (lambda s: s.replace(TOKEN, "***")) if TOKEN else (lambda s: s)

os.chdir(WORK)
cmd = (["git", "-C", "repo", "pull", "--ff-only"] if os.path.isdir("repo/.git")
       else ["git", "clone", "--depth", "1", "-b", BRANCH, url, "repo"])
r = subprocess.run(cmd, capture_output=True, text=True)
print(hide((r.stdout + r.stderr).strip()))
os.chdir(f"{{WORK}}/repo"); sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())''')

md("""## 1) Kiểm tra điều kiện

Cần `data/xlit_kn.json` đã có trong repo. Nếu chưa, chạy `build_xlit_cache.ipynb` trước rồi
commit kết quả.""")
code('''!pip -q install ftfy
import torch
from src.data.transliterate import CACHE as XLIT, load_cache

xlit = load_cache()
print(f"cache chuyen tu: {len(xlit)} cau" if xlit else "!! THIEU data/xlit_kn.json")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "khong co")''')

md("""## 2) Dịch

`facebook/nllb-200-distilled-600M` — transformers thuần, không cần fairseq, và Kannada
(`kan_Knda`) là một trong 200 ngôn ngữ nó hỗ trợ. Lưu sau **mỗi batch** nên bị ngắt thì chạy
lại sẽ tiếp tục.""")
code('''!python -m src.data.translate --source kn --model nllb --batch_size 24 --num_beams 4''')

code('''# Bien the: dich THANG tu chu Latin, khong qua buoc chuyen tu (yeu hon, nhung de doi chung)
# !python -m src.data.translate --source roman --out data/mt_en_roman.json
# Model to hon, cham hon, chat luong hon:
# !python -m src.data.translate --source kn --model nllb-1.3b --batch_size 12''')

md("""## 3) Kiểm tra bằng mắt — **quan trọng nhất**

Điều cần soi: **từ chửi có sống sót qua bản dịch không?** Nếu `sule`, `nayi`, `thu`, `maga`
bị dịch thành từ trung tính hoặc biến mất, thì TRAA đã phá đúng tín hiệu mà Task A cần, và
bạn biết ngay là nó sẽ không giúp.""")
code('''import json, re
import pandas as pd
from src.data.transliterate import load_cache as load_xlit
from src.data.translate import CACHE as MT

mt, xlit = json.load(open(MT, encoding='utf-8')), load_xlit()
print(f"{len(mt)} cau da dich\\n")

# nhung cau chua token Hate manh nhat (EDA muc 6) -- xem ban dich con giu duoc khong
SLURS = ["sule", "nayi", "thu", "maga", "magane", "dagar", "muduka"]
rows = []
for k in mt:
    if any(re.search(rf"\\b{s}\\b", k.lower()) for s in SLURS):
        rows.append({"goc": k[:45], "kannada": xlit.get(k, "?")[:30], "english": mt[k][:55]})
    if len(rows) >= 12:
        break
pd.set_option("display.max_colwidth", 60)
display(pd.DataFrame(rows))''')

md("""## 4) Tải về rồi commit

```bash
git add data/mt_en.json
git commit -m "Add Kannada->English translation cache"
git push
```""")
code('''import shutil
from src.data.translate import CACHE

shutil.copy(CACHE, "/kaggle/working/mt_en.json")
print(f"tai ve: /kaggle/working/mt_en.json  ({CACHE.stat().st_size/1e6:.1f} MB)")''')

write("build_mt_cache.ipynb")


# =====================================================================================
#  Notebook 4 -- domain-adaptive MLM. Produces an adapted checkpoint that train.py can be
#  pointed at; kept separate because it runs once and its output is reused by every run.
# =====================================================================================
md(f"""# Domain-adaptive pretraining — MLM trên Kanglish

**Settings:** Accelerator = **GPU T4 ×2** · Internet = **On** · ~25 phút

Từ vựng của MuRIL được xây cho tiếng Ấn viết bằng **chữ bản địa**, nên Kanglish chữ Latin bị xé
vụn: **2,05 mảnh/từ** so với **1,04** của tiếng Anh, và chỉ **36,7%** từ còn nguyên một mảnh.

```
government    ->  ['government']                        (tiếng Anh, 1 mảnh)
Bajetigu      ->  ['Ba', '##jet', '##ig', '##u']        (Kanglish, 4 mảnh)
```

Embedding của `##jet`, `##ikk` được học trong ngữ cảnh **chẳng liên quan gì** tới tiếng Kannada.
MLM kéo chúng về đúng chỗ — và vì nó **không cần nhãn**, nó tránh được đúng vấn đề đã giết chết
phương án nối dữ liệu ngoài vào train (đo được +0,0012 ± 0,0087, tức bằng không): *"offensive"*
khác *"hate"*, nhưng văn bản thì vẫn cùng một ngôn ngữ.

**Nói thẳng về quy mô:** kho của bạn ~**0,31 triệu token**, trong khi MuRIL pretrain trên ~16 **tỷ**
và các bài DAPT thường dùng 1–100 triệu. Thứ cứu vớt là chỉ **8.405 mục từ vựng (4,3%)** thực sự
xuất hiện, nên toàn bộ ngân sách dồn đúng vào những embedding đang sai. Kỳ vọng **+0,01 đến
+0,03**, không phải bước nhảy.""")

code(f'''import os, subprocess, sys

REPO, BRANCH, WORK = "{REPO}", "{BRANCH}", "/kaggle/working"
TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except Exception:
    pass
url = f"https://{{TOKEN + '@' if TOKEN else ''}}github.com/{{REPO}}.git"
hide = (lambda s: s.replace(TOKEN, "***")) if TOKEN else (lambda s: s)

os.chdir(WORK)
cmd = (["git", "-C", "repo", "pull", "--ff-only"] if os.path.isdir("repo/.git")
       else ["git", "clone", "--depth", "1", "-b", BRANCH, url, "repo"])
r = subprocess.run(cmd, capture_output=True, text=True)
print(hide((r.stdout + r.stderr).strip()))
if r.returncode:
    raise SystemExit("git that bai -- kiem tra Internet = On, repo Public")

os.chdir(f"{{WORK}}/repo"); sys.path.insert(0, os.getcwd())
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())''')

code('''!pip -q install ftfy sentencepiece
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader''')

md("""## 1) Kho văn bản

Gom **mọi** file có Kanglish, khử trùng lặp. Không cần nhãn nào.

> **Lát held-out bị loại khỏi kho, mặc định.** Văn bản đó không mang nhãn nên giữ lại cũng không
> rò rỉ nhãn — nhưng model sẽ đã đọc đúng những câu ấy, và mọi macro-F1 đo trên lát đó sau này
> sẽ **lạc quan một cách âm thầm**. Giữ trung thực chỉ tốn 919 dòng trên 13.995.
>
> Ngược lại, `*_validation_inputs.csv` và file test **được** đưa vào: đó là transductive learning
> thông thường, và đúng là tình huống bản nộp sẽ chạy. Nhớ khai báo điều này trong bài báo.""")
code('''from pretrain_mlm import build_corpus
from src.utils.config import load_config
from src.data.preprocessing import ensure_processed

cfg = load_config("configs/base.yaml", task="a")
ensure_processed(cfg)
corpus = build_corpus(cfg, include_eval=False)
print(f"-> {len(corpus):,} dong")
for t in corpus[:5]:
    print("   ", t[:80])''')

md("""## 2) Chạy MLM

`--epochs 15` trên ~13k dòng là khoảng 20–25 phút trên T4. Theo dõi **perplexity**: nó bắt đầu
cao vì các mảnh từ vựng đang sai với văn bản này, và **giảm xuống chính là sự thích nghi** mà
script này tồn tại để làm. Nếu nó không giảm, có gì đó sai.""")
code('''MODEL      = 'google/muril-base-cased'
MLM_EPOCHS = 15     # so epoch cua MLM. Fine-tune co so epoch RIENG, o muc 3.
BATCH      = 32     # TONG, chia deu cho cac GPU (2 x T4 -> 16/GPU). Giam neu OOM.
GRAD_ACCUM = 1      # BATCH x GRAD_ACCUM = batch hieu dung (32 x 1 = 32)
MAX_LEN    = 128    # do dai theo TOKEN; giam con 96 cung tiet kiem nhieu VRAM

!python pretrain_mlm.py --model {MODEL} --epochs {MLM_EPOCHS} --batch_size {BATCH} --grad_accum {GRAD_ACCUM} --max_len {MAX_LEN}

# Bien the:
# !python pretrain_mlm.py --model Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL --epochs 15
# !python pretrain_mlm.py --model xlm-roberta-base --epochs 15
# !python pretrain_mlm.py --epochs 20 --lr 1e-4        # kho nho -> co the can lr cao hon
# !python pretrain_mlm.py --include_eval               # CHI cho ban nop cuoi, diem noi bo se lac quan
#
# NEU OOM: ha BATCH va tang GRAD_ACCUM de giu batch hieu dung (vd 16 x 2, hoac 8 x 4).
# CHI dung 1 GPU du co 2:  them --single_gpu
#
# THU NHANH truoc khi bo 25 phut GPU -- chay het ca phan luu trong ~1 phut:
# !python pretrain_mlm.py --max_rows 200 --epochs 1 --out /tmp/mlm_test''')

md("""> **Về OOM.** Logits của MLM có shape `[batch, len, vocab]`, mà vocab của MuRIL là
> **197.285** — nên riêng một tensor đó ở `batch 32 × len 128` đã chiếm **3,2 GB**, và phải giữ
> hai bản (xuôi + ngược). Đó là thứ làm nổ T4, không phải model.
>
> | batch × len | logits (xuôi+ngược) | + model/AdamW | ~tổng |
> |---|---|---|---|
> | 32 × 128 | 6,46 GB | 3,81 GB | ~11,5 GB → **OOM** |
> | 16 × 128 | 3,23 GB | 3,81 GB | ~8,2 GB |
> | **8 × 128** | **1,62 GB** | 3,81 GB | **~6,6 GB** ✅ mặc định |
> | 4 × 128 | 0,81 GB | 3,81 GB | ~5,8 GB |
>
> **Trên 2×T4 script tự dùng cả hai GPU** (`DataParallel`), nên `BATCH` là **tổng** và mỗi GPU
> chỉ giữ một nửa:
>
> | BATCH tổng | /GPU | logits/GPU | **GPU 0** | GPU 1 |
> |---|---|---|---|---|
> | 16 | 8 | 1,62 GB | 6,4 GB | 3,6 GB |
> | **32** | **16** | **3,23 GB** | **8,0 GB** | 5,2 GB |
> | 48 | 24 | 4,85 GB | 9,7 GB | 6,8 GB |
> | 64 | 32 | 6,46 GB | 11,3 GB | 8,4 GB |
>
> GPU 0 luôn chật hơn vì chỉ nó giữ trọng số gốc, gradient và hai trạng thái AdamW (3,8 GB);
> GPU kia chỉ giữ một bản sao trọng số.
>
> **Chạy trên máy 1 GPU vẫn bình thường** — script tự dò `torch.cuda.device_count()` và chỉ bật
> `DataParallel` khi thấy nhiều hơn một. Checkpoint lưu ra giống hệt nhau trong cả hai trường hợp.
>
> Script in ước lượng VRAM **trước khi** train, nên bạn biết trước chứ không phải đợi nó crash.""")

md("""## 3) Fine-tune từ checkpoint vừa thích nghi

Không cần code mới — chỉ trỏ `model.name` vào thư mục vừa lưu. Chạy **cả bản gốc lẫn bản MLM**
thì mới biết nó có giúp không.""")
code('''# (config, checkpoint MLM tuong ung). Them dong moi sau khi da chay pretrain_mlm.py cho model do.
PAIRS = [
    ('muril',   'checkpoints/mlm/muril-base-cased'),
  # ('roberta', 'checkpoints/mlm/xlm-roberta-base'),
  # ('cnerg_muril', 'checkpoints/mlm/kannada-codemixed-abusive-MuRIL'),
]
FT_EPOCHS = 6     # so epoch khi FINE-TUNE -- KHAC voi MLM_EPOCHS o tren (cai do la cua MLM).
                  # base.yaml dang de 20; doi gia tri thi BAT BUOC co suffix, nen no nam trong SUF.
SUF = f'_e{FT_EPOCHS}'
# patience = epochs tuc TAT early stopping: trainer van giu epoch tot nhat, con lich LR duoc
# anneal het. Voi epochs=20 thi model dat dinh o epoch ~5 luc LR con ~83% -- phi doan anneal.
FT = f'--set training.epochs={FT_EPOCHS} training.early_stopping_patience={FT_EPOCHS}'

# Ten thu muc TU PHAN BIET: run_name gan them '_mlm' khi model.name bi ghi de, nen ban goc va
# ban MLM khong bao gio dung chung mot thu muc, va khong can them --run_suffix bang tay.
for cfg, ckpt in PAIRS:
    for t in ('a', 'b'):
        print("=" * 70)
        !python train.py --config configs/{cfg}.yaml --task {t} {FT} --run_suffix {SUF}
        !python train.py --config configs/{cfg}.yaml --task {t} {FT} model.name={ckpt} --run_suffix {SUF}''')

code('''import pandas as pd
d = pd.read_csv('results/metrics.csv')
display(d[d.run.str.contains('muril')][['task', 'run', 'macro_f1', 'accuracy', 'best_epoch']]
        .sort_values(['task', 'macro_f1'], ascending=[True, False]))''')

md("""## 4) Tải checkpoint về

**~0,9 GB** — chỉ tải nếu bạn muốn dùng lại ở máy hoặc ở session Kaggle khác. Nếu không, cứ chạy
lại notebook này (25 phút) thì nhanh hơn là tải lên tải xuống.""")
code('''!cd /kaggle/working/repo && zip -r -q /kaggle/working/mlm_muril.zip checkpoints/mlm
!ls -lh /kaggle/working/mlm_muril.zip''')

write("pretrain_mlm.ipynb")


# =====================================================================================
# =====================================================================================
#  Notebook 5 -- embedding mix. One tokenizer, several word-embedding tables mixed before
#  one encoder (model.embed_mix), measured against the plain encoder and against the MLM
#  encoder itself. Needs the MLM checkpoint from pretrain_mlm.ipynb, fetched from Drive.
# =====================================================================================
MLM_DRIVE_ID = "1XgEo2UzdOw8sX4ofd0mmzbOVVhfYnDo7"

md(f"""# Embedding mix — 1 tokenizer · nhiều bảng embedding · 1 encoder

**Settings:** Accelerator = **GPU T4 ×2** · Internet = **On**

```
"Bajetigu thu nin …"
      │  tokenizer của ENCODER (1 cái)  → input_ids
      ├──► E_0 : bảng embedding của encoder (cnerg_muril)            ← vẫn được train
      ├──► E_1 : bảng của google/muril-base-cased                    ← đóng băng
      └──► E_2 : bảng của MuRIL sau MLM (checkpoints/mlm/…)           ← đóng băng
                 e = E_0[id] + Σ a_k · (E_k[id] − E_0[id])      a_k khởi tạo = 0
      │
      ▼  + position + LayerNorm → 1 encoder (cnerg_muril, 12 block) → pooling → head
```

- **`embed_mix_mode: token`**: mỗi token có trọng số `a_k` riêng (router `Linear(768→K)` khởi tạo 0).
  **`global`**: một trọng số chung cho mỗi nguồn.
- Nguồn phải **dùng chung vocab** với encoder. MuRIL, cnerg_muril và MuRIL-MLM đều dùng cùng một
  file vocab (197.285 mục). mBERT/XLM-R bị **từ chối**, vì chỉ khớp ~80% token và phần lệch
  chính là các từ Kanglish.
- Log mỗi run in `embed_mix a[nguồn]`: gần **0** nghĩa là nguồn đó không được dùng.

> **Kỳ vọng thực tế.** Đo trước: trên token Kanglish, bảng MuRIL-MLM gần như **trùng** bảng MuRIL
> gốc (cosine ≈ 0,998). MLM thay đổi chủ yếu **encoder**, không phải bảng. Vì vậy notebook chạy
> kèm một **run đối chứng dùng thẳng encoder MLM**. Nếu run đó thắng mà mix không thắng, câu trả
> lời là dùng encoder MLM, không phải trộn bảng.""")

code(f'''import os, subprocess, sys

REPO, BRANCH, WORK = "{REPO}", "{BRANCH}", "/kaggle/working"
TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
except Exception:
    pass
url = f"https://{{TOKEN + '@' if TOKEN else ''}}github.com/{{REPO}}.git"
hide = (lambda s: s.replace(TOKEN, "***")) if TOKEN else (lambda s: s)

os.chdir(WORK)
cmd = (["git", "-C", "repo", "pull", "--ff-only"] if os.path.isdir("repo/.git")
       else ["git", "clone", "--depth", "1", "-b", BRANCH, url, "repo"])
r = subprocess.run(cmd, capture_output=True, text=True)
print(hide((r.stdout + r.stderr).strip()))
if r.returncode:
    raise SystemExit("git that bai -- kiem tra Internet = On, repo Public")

os.chdir(f"{{WORK}}/repo"); sys.path.insert(0, os.getcwd())
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip())''')

code('''!pip -q install ftfy sentencepiece gdown
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader''')

md("""## 1) Tải checkpoint MLM

File `mlm_muril.zip` (~1,4 GB) do `pretrain_mlm.ipynb` sinh ra, để trên Google Drive (chế độ
*Anyone with the link*). Bên trong là `checkpoints/mlm/muril-base-cased/` gồm `config.json`
(BertForMaskedLM, vocab 197.285), `model.safetensors` (fp32, có kèm đầu MLM, `train.py` tự bỏ
qua), `tokenizer.json` và `tokenizer_config.json`. Giải nén **ở gốc repo** thì đường dẫn khớp
với các config.

- Đã có thư mục đó thì cell tự bỏ qua, không tải lại.
- Drive báo *quota exceeded*: tải file về máy, rồi *Add Input ▸ Upload* thành một Kaggle Dataset
  và đặt `KAGGLE_ZIP` bên dưới trỏ vào nó (hoặc để trống: cell tự tìm `mlm_muril.zip` trong
  `/kaggle/input`).""")

code(f'''import os, zipfile, glob

MLM_DIR    = "checkpoints/mlm/muril-base-cased"
DRIVE_ID   = "{MLM_DRIVE_ID}"
KAGGLE_ZIP = ""        # vd "/kaggle/input/mlm-muril/mlm_muril.zip" neu tai qua Kaggle Dataset
ZIP        = "/kaggle/working/mlm_muril.zip"

if os.path.isfile(f"{{MLM_DIR}}/model.safetensors"):
    print("da co", MLM_DIR, "-> bo qua tai")
else:
    src = KAGGLE_ZIP or (glob.glob("/kaggle/input/**/mlm_muril.zip", recursive=True) or [None])[0]
    if src:
        print("dung zip co san:", src); ZIP = src
    else:
        import gdown
        gdown.download(id=DRIVE_ID, output=ZIP, quiet=False)
    with zipfile.ZipFile(ZIP) as z:
        print("\\n".join(f"  {{i.file_size / 1e6:9.1f}} MB  {{i.filename}}" for i in z.infolist()))
        z.extractall(".")          # -> ./checkpoints/mlm/muril-base-cased/
    if ZIP.startswith("/kaggle/working/"):
        os.remove(ZIP)             # 1.4 GB it khong can nua; /kaggle/working gioi han ~20 GB
!ls -la {{MLM_DIR}}''')

md("""**Kiểm tra checkpoint** (nhanh, không cần GPU): nạp config và tokenizer, so vocab với encoder.
Tokenizer trong zip được lưu bằng transformers 5. Nếu bản trên Kaggle không đọc được, cell sẽ ghi
đè bằng tokenizer của `google/muril-base-cased`. Việc này an toàn vì cùng file vocab và cùng
kiểu giữ chữ hoa/thường; MLM cũng được train bằng chính tokenizer đó.""")

code('''import transformers
from transformers import AutoConfig, AutoTokenizer
print("transformers", transformers.__version__)
cfg = AutoConfig.from_pretrained(MLM_DIR)
print("config:", cfg.architectures, "vocab", cfg.vocab_size, "hidden", cfg.hidden_size)
try:
    tok = AutoTokenizer.from_pretrained(MLM_DIR)
    tok.tokenize("Bajetigu thu nin ajji")
except Exception as e:
    print("tokenizer trong zip khong doc duoc:", str(e)[:150], "\\n-> thay bang google/muril-base-cased")
    AutoTokenizer.from_pretrained("google/muril-base-cased").save_pretrained(MLM_DIR)
    tok = AutoTokenizer.from_pretrained(MLM_DIR)
enc = AutoTokenizer.from_pretrained("Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL")
print("vocab MLM == vocab cnerg_muril:", tok.get_vocab() == enc.get_vocab())
print("MLM   :", tok.tokenize("Bajetigu thu nin ajji 😡"))
print("cnerg :", enc.tokenize("Bajetigu thu nin ajji 😡"), " (cnerg chuyen chu thuong)")''')

md("""## 2) Bảng điều khiển

**Mọi thứ chỉnh ở MỘT cell dưới đây.** Cell ngay sau nó chỉ in **kế hoạch** (tên run và đầy đủ
override), không train gì. Xem kế hoạch rồi mới chạy cell train.

Các run được sinh ra:

| run | bật khi | khác mốc ở đâu |
|---|---|---|
| `base` | `RUN_BASE = True` | mốc: encoder thuần, không trộn |
| `mix:<mode>` | mỗi mode trong `MIX_MODES` | + trộn các bảng trong `MIX_SOURCES` |
| `mlm_enc` | `RUN_MLM_ENC = True` | đối chứng: dùng thẳng MuRIL-MLM làm encoder |

`SIDE`, `HYBRID` và `HEAD` được áp cho **mọi** run, kể cả `base` và `mlm_enc`. Như vậy các run
chỉ khác nhau ở phần trộn, và so sánh vẫn công bằng. Chênh lệch dưới ~0,02 macro-F1 là trong
mức nhiễu của lát eval.""")

code('''# ============================== BANG DIEU KHIEN ==============================
TASKS = ['a', 'b']                  # 'a' = Hate/Non-Hate, 'b' = 6 nhom doi tuong

# ---- encoder (kiem luon tokenizer) -------------------------------------------------
ENCODER = 'cnerg_muril'             # ten file trong configs/: cnerg_muril | muril | roberta | cnerg_xlmr ...

# ---- embed_mix: tron bang embedding -----------------------------------------------
# Phai DUNG CHUNG VOCAB voi ENCODER (khac vocab -> train.py tu choi va giai thich).
#   voi cnerg_muril / muril:  'google/muril-base-cased', 'checkpoints/mlm/muril-base-cased',
#                             'Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL' (neu ENCODER khac no)
#   voi roberta / cnerg_xlmr: 'xlm-roberta-base', 'Hate-speech-CNERG/deoffxlmr-mono-kannada'
MIX_SOURCES = ['google/muril-base-cased', 'checkpoints/mlm/muril-base-cased']   # [] = khong tron
MIX_MODES   = ['token', 'global']   # moi mode -> 1 run.  token = trong so rieng moi token | global = chung
MIX_LR      = 1e-3                  # lr cua router / trong so tron

# ---- ap cho MOI run -----------------------------------------------------------------
SIDE     = None       # None | 'char' | 'phonetic' | 'char+phonetic'   (vector phu muc tu, gate = 0)
HYBRID   = None       # None | 'tfidf'                                   (ghep TF-IDF truoc head)
HEAD     = 'linear'   # 'linear' | 'mlp'
MLP_DIMS = [512]      # CHI HEAD='mlp'

# ---- huan luyen -------------------------------------------------------------------
EPOCHS   = 6          # cung la do dai lich LR
PATIENCE = 6          # >= EPOCHS: chay tron lich roi giu epoch tot nhat
LR       = 2e-5       # lr cua encoder
BATCH    = 32         # tong tren moi GPU
MAX_LEN  = 96         # token; 128 giam 1/2 so cau bi cat (Religion/Geo-political bi cat nhieu nhat)
LOSS     = 'auto'     # auto (a: ce, b: focal) | ce | wce | focal
SEED     = 42

# ---- run doi chung ----------------------------------------------------------------
RUN_BASE    = True    # encoder khong tron -- moc de so
RUN_MLM_ENC = True    # encoder = checkpoints/mlm/muril-base-cased, khong tron

SUFFIX = None         # None = tu sinh tu cac tham so huan luyen o tren (vd '_e6')
# ===================================================================================''')

code('''# ---- KE HOACH: dung bang dieu khien -> danh sach run. Khong train gi o day. ----
import os
from src.utils.config import load_config, run_name

MLM_DIR = 'checkpoints/mlm/muril-base-cased'
fmt = lambda v: str(v).replace(" ", "")              # list -> khong co dau cach cho shell

if SUFFIX is None:                                    # chi ghi nhung gi khac mac dinh
    SUFFIX = f"_e{EPOCHS}"
    for val, dflt, tag in ((LR, 2e-5, 'lr'), (BATCH, 32, 'bs'), (MAX_LEN, 96, 'len'), (PATIENCE, EPOCHS, 'p')):
        if val != dflt:
            SUFFIX += f"_{tag}{val:g}"
    if HEAD == 'mlp':                                 # side/hybrid tu them _se/_hyb, head thi khong
        SUFFIX += "_mlp" + "x".join(map(str, MLP_DIMS))

common = {'training.epochs': EPOCHS, 'training.early_stopping_patience': PATIENCE,
          'training.lr': LR, 'training.batch_size': BATCH, 'data.max_len': MAX_LEN,
          'training.loss': LOSS, 'model.head': HEAD}
if HEAD == 'mlp':
    common['model.mlp_dims'] = MLP_DIMS
if SIDE:
    common['model.side_embedding'] = SIDE
if HYBRID:
    common['model.hybrid'] = HYBRID

PLAN = []   # (nhan, config, overrides)
if RUN_BASE:
    PLAN.append(('base', ENCODER, {}))
if MIX_SOURCES:
    for mode in MIX_MODES:
        PLAN.append((f'mix:{mode}', ENCODER, {'model.embed_mix': MIX_SOURCES,
                                              'model.embed_mix_mode': mode,
                                              'model.embed_mix_lr': MIX_LR}))
if RUN_MLM_ENC:
    if os.path.isfile(f'{MLM_DIR}/config.json'):
        PLAN.append(('mlm_enc', 'muril', {'model.name': MLM_DIR}))
    else:
        print(f"!! RUN_MLM_ENC bo qua: chua co {MLM_DIR} (chay cell tai MLM o muc 1)")
if any(MLM_DIR in str(s) for s in MIX_SOURCES) and not os.path.isfile(f'{MLM_DIR}/config.json'):
    print(f"!! MIX_SOURCES co {MLM_DIR} nhung thu muc chua co -> cac run mix se loi")

JOBS = []
print(f"SUFFIX = {SUFFIX!r}   |   {len(PLAN)} run x {len(TASKS)} task = {len(PLAN) * len(TASKS)} lan train\\n")
for label, conf, extra in PLAN:
    sets = {**common, **extra}
    for t in TASKS:
        cfg = load_config(f'configs/{conf}.yaml', [f"{k}={fmt(v)}" for k, v in sets.items()],
                          task=t, seed=SEED, run_suffix=SUFFIX)
        JOBS.append((label, conf, t, sets, run_name(cfg)))
        print(f"  {label:13s} task {t}  ->  {run_name(cfg)}")
print("\\noverride chung:", {k: v for k, v in common.items()})''')

code('''# ---- TRAIN: chay dung KE HOACH o tren ----
import time
t0 = time.time()
for n, (label, conf, t, sets, name) in enumerate(JOBS, 1):
    args = " ".join(f"{k}={fmt(v)}" for k, v in sets.items())
    print("\\n" + "=" * 72)
    print(f"[{n}/{len(JOBS)}]  {label} | {name} | +{(time.time() - t0) / 60:.1f} phut")
    print("=" * 72, flush=True)
    !python train.py --config configs/{conf}.yaml --task {t} --seed {SEED} --set {args} --run_suffix {SUFFIX}
print(f"\\nxong {len(JOBS)} run trong {(time.time() - t0) / 60:.1f} phut")''')

md("""## 3) Kết quả

macro-F1 của đúng các run trong kế hoạch, kèm trọng số trộn và gate mà mỗi run học được (đọc từ
log). Nhớ rằng lát eval nhiễu ±0,02.""")

code('''import pandas as pd
names = {name: label for label, _, _, _, name in JOBS}
d = pd.read_csv('results/metrics.csv')
d = d[d.run.isin(names)].assign(nhan=lambda x: x.run.map(names))
display(d[['task', 'nhan', 'run', 'macro_f1', 'accuracy', 'best_epoch']]
        .sort_values(['task', 'macro_f1'], ascending=[True, False]))
for label, conf, t, _, name in JOBS:
    f = f'logs/{t}_{name}.log'
    if os.path.isfile(f):
        lines = [l.split('| INFO | ')[-1].strip() for l in open(f, encoding='utf-8')
                 if 'embed_mix a[' in l or 'side_gate' in l]
        if lines:
            print(f"\\n{label} / task {t}:"); print("\\n".join("  " + l for l in lines))''')

code('''# Blend cac run trong ke hoach (trong so toi uu tren lat eval -> lac quan):
for t in TASKS:
    runs = " ".join(d[d.task == t].run)
    if runs:
        !python evaluate.py --task {t} --runs {runs} --optimize --tag blend_mix''')

md("""## 4) Tải kết quả về

Nén `results/` và `logs/`. **Không** nén checkpoint: mỗi checkpoint embed_mix nặng thêm khoảng
0,3 GB cho mỗi bảng trộn.""")

code('''%cd /kaggle/working/repo
!zip -r -q /kaggle/working/results_embed_mix.zip results logs
!ls -lh /kaggle/working/results_embed_mix.zip''')

write("embed_mix.ipynb")
