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
| `test.npy` | `hastika_*_test.csv` của BTC | ❌ | **nộp phase Evaluation** |

Chữ "val" xuất hiện ở hai nghĩa khác nhau: lát held-out (có nhãn, để bạn tự chấm) và file
validation của BTC (không nhãn, để nộp). `eval.npy` là cái đầu, `val.npy` là cái sau.

### Nộp thế nào

Codabench có **2 leaderboard riêng biệt**, mỗi task nộp **một** `predictions.csv`. Bạn chỉ có
**20 lượt nộp**, nên đừng nộp từng model — chọn theo điểm held-out ở mục 4 rồi nộp bản tốt nhất.

- Mọi run giờ sinh **cả hai** file nộp: `{task}_val_{run}` (phase Development) và
  `{task}_test_{run}` (phase Evaluation).
- Run train **trước khi có file test**: chạy lại đúng lệnh train của nó là đủ. `train.py` thấy
  thiếu `test.npy` thì tự dự đoán từ checkpoint (transformer, không train lại), hoặc train lại
  (TF-IDF, vài giây; hoặc transformer không lưu checkpoint).""")

md("""**Cách 1 — từng model riêng: đã có sẵn.** `train.py` tự sinh file nộp ngay sau khi train xong:
`{task}_val_{run}` cho file validation và `{task}_test_{run}` cho file test của BTC. Cell này chỉ
để xem lại danh sách:""")
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
    # both = {t}_val_ens (phase Development) va {t}_test_ens (phase Evaluation), cung trong so
    !python inference.py --task {t} --runs {runs} --weights {ws} --split both --tag ens

# Neu bao "test.npy missing": run do train truoc khi co file test -> chay lai lenh train cua no
# (train.py tu bo sung test.npy), hoac du doan thang tu checkpoint:
# !python inference.py --task b --checkpoints checkpoints/b/muril_focal_s42 --input data/raw/hastika_multiclass_test.csv --tag ens1''')

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
> sẽ **lạc quan một cách âm thầm**. Giữ trung thực chỉ tốn 919 dòng (kho còn khoảng 13,5 nghìn dòng).
>
> Ngược lại, `*_validation_inputs.csv` và file test (`hastika_*_test.csv`) **được** đưa vào: đó là transductive learning
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

md("""## 2) Chạy MLM — bản 2

So với bản 1 (`checkpoints/mlm/muril-base-cased`):

| | bản 1 | **bản 2** |
|---|---|---|
| Cách che | từng mảnh (`th [MASK]` → nhìn `th` là đoán ra) | **cả từ** `--wwm` (`[MASK] [MASK]` → phải đọc câu) |
| Kho | train + val | **+ file test** (`hastika_*_test.csv`, không dùng nhãn) |
| Xuất phát | `google/muril` | **`cnerg_muril`** (+ bản `google/muril` để so) |
| Đầu MLM | có sẵn | cnerg **không có** → chép từ `google/muril` (`--mlm_head_from`) |
| Khi nào dừng | lưu epoch cuối | tách **5%** làm tập đánh giá (`--eval_ratio`), **lưu epoch tốt nhất** |

**Đọc log:** mỗi epoch in `loss/ppl` (trên kho train) và `eval loss/ppl` (trên 5% tách riêng).
`*` = epoch tốt nhất đến lúc đó, và checkpoint được lưu ngay lúc ấy. Nếu `eval ppl` bắt đầu
**tăng** trong khi `ppl` train vẫn giảm, model đang học thuộc kho.

> **Đừng hoảng với số đầu của cnerg.** Đo ở máy: trước khi train, `cnerg_muril` có eval loss
> **14,9**, còn tệ hơn đoán đều (12,2), vì các tầng trên của nó đã bị fine-tune cho phân loại nên
> đầu MLM chép sang không khớp. Nhưng chỉ sau 24 bước nó xuống **8,4**, dưới cả `google/muril` gốc
> (8,6 với cùng cách che cả từ). Với vài nghìn bước trên Kaggle nó hồi phục hẳn.
>
> Che cả từ **khó hơn** che từng mảnh (google/muril: loss 8,6 so với 7,8). Vì vậy ppl của bản 2
> **không so được** với ppl của bản 1. Chỉ so được bằng macro-F1 sau fine-tune, ở mục 3.""")

code('''# ============================ MLM ban 2 ============================
MLM_RUNS = [   # (checkpoint xuat phat, chep dau MLM tu, thu muc luu) -- moi dong ~30 phut tren 2xT4
    ('Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL', 'google/muril-base-cased', 'checkpoints/mlm_v2/cnerg-muril'),
    ('google/muril-base-cased',                           None,                      'checkpoints/mlm_v2/muril'),
]
MLM_EPOCHS = 15     # tran; epoch tot nhat theo eval ppl moi duoc luu
WWM        = True   # che ca tu
EVAL_RATIO = 0.05   # 5% kho lam tap danh gia MLM (0 = tat, luu epoch cuoi nhu ban 1)
BATCH      = 32     # TONG, chia deu cho cac GPU (2 x T4 -> 16/GPU). Giam neu OOM.
GRAD_ACCUM = 1      # BATCH x GRAD_ACCUM = batch hieu dung
MAX_LEN    = 128    # token
LR         = 5e-5

import time
for model, head, out in MLM_RUNS:
    flags = (f"--model {model} --out {out} --epochs {MLM_EPOCHS} --batch_size {BATCH} "
             f"--grad_accum {GRAD_ACCUM} --max_len {MAX_LEN} --lr {LR} --eval_ratio {EVAL_RATIO}"
             + (" --wwm" if WWM else "") + (f" --mlm_head_from {head}" if head else ""))
    print("=" * 72, f"\\n{model}  ->  {out}\\n" + "=" * 72, flush=True)
    t0 = time.time()
    !python pretrain_mlm.py {flags}
    print(f"-> {(time.time() - t0) / 60:.1f} phut")

# THU NHANH truoc khi bo 1 gio GPU (chay het ca phan luu, ~1-2 phut):
# !python pretrain_mlm.py --model Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL --mlm_head_from google/muril-base-cased --wwm --eval_ratio 0.2 --max_rows 200 --epochs 1 --out /tmp/mlm_test
# NEU OOM: ha BATCH va tang GRAD_ACCUM (vd 16 x 2). CHI 1 GPU: them --single_gpu vao flags.''')

md("""> **Về OOM.** Logits của MLM có shape `[batch, len, vocab]`, mà vocab của MuRIL là
> **197.285**, nên riêng một tensor đó ở `batch 32 × len 128` đã chiếm **3,2 GB**, và phải giữ
> hai bản (xuôi + ngược). Đó là thứ làm nổ T4, không phải model.
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
> GPU 0 luôn chật hơn vì chỉ nó giữ trọng số gốc, gradient và hai trạng thái AdamW (3,8 GB).
> Che cả từ và tập đánh giá **không** tốn thêm VRAM. Script in ước lượng VRAM **trước khi** train.""")

md("""## 3) Fine-tune từ checkpoint vừa thích nghi

Không cần code mới: chỉ trỏ `model.name` vào thư mục vừa lưu. Mỗi dòng trong `PAIRS` chạy **bản
gốc** và **bản MLM** cho cả hai task. Bản gốc đã chạy rồi thì tự bỏ qua. `run_name` tự gắn đuôi
theo thư mục (`_mlm`, `_mlm-v2`) nên các bản không bao giờ đè lên nhau.""")
code('''import os
PAIRS = [   # (config, checkpoint MLM tuong ung)
    ('cnerg_muril', 'checkpoints/mlm_v2/cnerg-muril'),   # MLM ban 2 tu cnerg
    ('muril',       'checkpoints/mlm_v2/muril'),         # MLM ban 2 tu google/muril
    ('muril',       'checkpoints/mlm/muril-base-cased'), # MLM ban 1 (neu co) -- de so ban 1 va ban 2
]
FT_EPOCHS = 6     # so epoch khi FINE-TUNE -- KHAC voi MLM_EPOCHS o tren
SUF = f'_e{FT_EPOCHS}'
# patience = epochs: chay tron lich LR roi giu epoch tot nhat
FT = f'--set training.epochs={FT_EPOCHS} training.early_stopping_patience={FT_EPOCHS}'

for cfg, ckpt in PAIRS:
    if not os.path.isfile(f'{ckpt}/config.json'):
        print(f"bo qua {ckpt}: chua co"); continue
    for t in ('a', 'b'):
        print("=" * 70, flush=True)
        !python train.py --config configs/{cfg}.yaml --task {t} {FT} --run_suffix {SUF}
        !python train.py --config configs/{cfg}.yaml --task {t} {FT} model.name={ckpt} --run_suffix {SUF}''')

code('''import pandas as pd
d = pd.read_csv('results/metrics.csv')
display(d[d.run.str.contains('muril') & d.run.str.endswith(SUF)][['task', 'run', 'macro_f1', 'accuracy', 'best_epoch']]
        .sort_values(['task', 'macro_f1'], ascending=[True, False]))''')

md("""## 4) Tải checkpoint về

Mỗi checkpoint MLM khoảng **1,5 GB** (fp32, kèm đầu MLM). Nén **riêng từng cái** để tải từng file
và để dùng trong `embed_mix.ipynb` hoặc session Kaggle khác (upload thành Kaggle Dataset hoặc
Google Drive). Cấu trúc bên trong zip giữ nguyên đường dẫn, nên giải nén ở gốc repo là dùng được.""")
code('''%cd /kaggle/working/repo
import glob, os
for d in sorted(glob.glob('checkpoints/mlm_v2/*')):
    z = f"/kaggle/working/{d.replace('/', '_')}.zip"
    !zip -r -q {z} {d}
    print(f"{z}: {os.path.getsize(z) / 1e9:.2f} GB")''')

md("""## File nộp (val + test)

Mỗi run fine-tune ở trên đã sinh **hai** file nộp trong `results/submissions/`:
`{task}_val_{run}` cho phase Development và `{task}_test_{run}` cho phase Evaluation.
Cell dưới liệt kê chúng và nén `results/` + `logs/` (gồm cả file nộp, **không** gồm checkpoint)
thành một file để tải về.""")
code('''%cd /kaggle/working/repo
import glob, os
subs = sorted(glob.glob('results/submissions/*/submission.zip'))
for split in ('val', 'test'):
    names = [os.path.basename(os.path.dirname(z)) for z in subs if f'_{split}_' in os.path.basename(os.path.dirname(z))]
    print(f"{split}: {len(names)} file nop")
    for n in names:
        print("   ", n)
!zip -r -q /kaggle/working/results_pretrain_mlm.zip results logs
!ls -lh /kaggle/working/results_pretrain_mlm.zip''')

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


# =====================================================================================
# =====================================================================================
#  Notebook 6 -- vocabulary extension + MLM (pretrain_mlm.py --extend_vocab), then
#  fine-tune and compare against the same MLM without the new words.
# =====================================================================================
md(f"""# Hướng 2 — Mở rộng vocab + MLM

**Settings:** Accelerator = **GPU T4 ×2** · Internet = **On** · khoảng 1,5–2 giờ nếu chạy hết

```
kho Kanglish ──► tìm từ hay gặp mà bị cắt vụn:  maklu → ma ##k ##lu  (96 lần)
                                                   │
tokenizer cnerg_muril  + ~180–210 từ mới  ──────►  maklu → [maklu]      makluge → [maklu] ##ge
bảng embedding 197.285 → +N dòng, mỗi dòng mới = trung bình các mảnh cũ
                                                   │
PHA 1 (3 epoch): encoder ĐÓNG BĂNG, chỉ train N dòng mới (lr 1e-3)
PHA 2 (15 epoch): MLM che cả từ, train toàn bộ, lưu epoch có eval ppl tốt nhất
                                                   │
fine-tune Task A/B  ──►  so với: cnerg_muril thuần  và  MLM bản 2 CÙNG cài đặt nhưng KHÔNG thêm từ
```

**Để so sánh công bằng:** run đối chứng (`RUN_CONTROL`) dùng **đúng** các cài đặt MLM (che cả từ,
cùng epoch, cùng seed, cùng tập đánh giá) và chỉ khác ở chỗ không thêm từ. Chênh lệch macro-F1
giữa hai bản khi đó là do vocab.

> **Kỳ vọng thực tế.** Mỗi từ mới chỉ xuất hiện khoảng 20 lần trong kho, nên mỗi token mới được
> "luyện" khoảng 100 lần trong cả quá trình. Đó là rất ít để học một vector 768 chiều. Mức tăng
> có thể từ 0 đến +1 điểm, nằm trong nhiễu ±0,02 của lát eval. **ppl của bản này không so được**
> với bản không thêm từ (đơn vị token khác nhau). Chỉ so bằng macro-F1 ở mục 4.""")

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
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!ls data/raw''')

md("""## 1) Bảng điều khiển

Mọi cài đặt nằm ở cell dưới. Các cell sau chỉ đọc từ đây.""")

code('''# ============================== BANG DIEU KHIEN ==============================
MODEL     = 'Hate-speech-CNERG/kannada-codemixed-abusive-MuRIL'   # diem xuat phat MLM
CONFIG    = 'cnerg_muril'               # config fine-tune tuong ung trong configs/
HEAD_FROM = 'google/muril-base-cased'   # cnerg khong co dau MLM -> chep tu day (None neu model co san)

# ---- mo rong vocab ----
MIN_COUNT     = 10      # tu phai gap >= so lan nay trong kho
MIN_PIECES    = 3       # ... va bi cat >= so manh nay (2 -> ~950 tu, rui ro hon)
MAX_WORDS     = 0       # tran so tu moi (0 = khong gioi han)
WARMUP_EPOCHS = 3       # PHA 1: chi train dong moi, encoder dong bang (0 = bo pha 1)
EXT_LR        = 1e-3    # lr cua pha 1

# ---- MLM (PHA 2) -- giong het run doi chung ----
MLM_EPOCHS = 15;  WWM = True;  EVAL_RATIO = 0.05
BATCH = 32;  GRAD_ACCUM = 1;  MAX_LEN = 128;  LR = 5e-5

OUT         = 'checkpoints/mlm_vx/cnerg-muril'   # co vocab mo rong
RUN_CONTROL = True                               # MLM cung cai dat, KHONG them tu
CONTROL_OUT = 'checkpoints/mlm_v2/cnerg-muril'   # bo qua neu da co (vd tu pretrain_mlm.ipynb)

# ---- fine-tune de so sanh ----
FT_EPOCHS = 6
TASKS     = ['a', 'b']
# ===================================================================================''')

md("""## 2) Xem trước từ mới (không train, vài giây)

Dùng đúng kho và đúng cách tách tập đánh giá như `pretrain_mlm.py` (seed 42). Từ mới chỉ được
chọn từ phần **train** của kho MLM, không lấy từ tập đánh giá.""")

code('''import random, pandas as pd
from transformers import AutoTokenizer
from pretrain_mlm import build_corpus
from src.utils.config import load_config
from src.models.vocab_ext import find_new_words

corpus = build_corpus(load_config("configs/base.yaml", task="a"), include_eval=False, log=lambda *a: None)
idx = list(range(len(corpus))); random.Random(42).shuffle(idx)
n_ev = max(1, int(round(EVAL_RATIO * len(corpus)))) if EVAL_RATIO > 0 else 0
train_part = [corpus[i] for i in idx[n_ev:]]

tok = AutoTokenizer.from_pretrained(MODEL)
cand, st = find_new_words(train_part, tok, MIN_COUNT, MIN_PIECES, MAX_WORDS)
print(f"kho MLM: {len(train_part):,} cau (train) + {n_ev:,} cau danh gia")
print(f"tu moi: {st['new_words']}  |  moi tu xuat hien trung binh {st['mean_count_of_new']:.0f} lan")
print(f"manh/tu: {st['pieces_per_word_before']:.2f} -> {st['pieces_per_word_after']:.2f}")
display(pd.DataFrame([(w, n, ' '.join(p)) for w, n, p in cand], columns=['tu moi', 'so lan', 'dang bi cat thanh']).head(40))''')

md("""## 3) Chạy MLM

Log in `pha1 ep …` rồi `pha2 ep …`. `*` = epoch có eval ppl tốt nhất đến lúc đó, được lưu ngay.
Danh sách từ đã thêm được ghi vào `OUT/added_words.tsv`.

**Phần cứng** (2×T4, batch 32, max_len 128): GPU 0 khoảng 8 GB. Thêm vài trăm từ gần như không
tốn thêm bộ nhớ. Pha 1 nhanh hơn một epoch thường vì encoder không cập nhật.""")

code('''import time, os
common = (f"--model {MODEL} --epochs {MLM_EPOCHS} --batch_size {BATCH} --grad_accum {GRAD_ACCUM} "
          f"--max_len {MAX_LEN} --lr {LR} --eval_ratio {EVAL_RATIO}"
          + (" --wwm" if WWM else "") + (f" --mlm_head_from {HEAD_FROM}" if HEAD_FROM else ""))

t0 = time.time()
print("=" * 72, "\\nMO RONG VOCAB ->", OUT, "\\n" + "=" * 72, flush=True)
!python pretrain_mlm.py {common} --out {OUT} --extend_vocab --ext_min_count {MIN_COUNT} --ext_min_pieces {MIN_PIECES} --ext_max_words {MAX_WORDS} --ext_warmup_epochs {WARMUP_EPOCHS} --ext_lr {EXT_LR}
print(f"-> {(time.time() - t0) / 60:.1f} phut")

if RUN_CONTROL:
    if os.path.isfile(f"{CONTROL_OUT}/config.json"):
        print("doi chung da co:", CONTROL_OUT, "-> bo qua")
    else:
        t0 = time.time()
        print("=" * 72, "\\nDOI CHUNG (khong them tu) ->", CONTROL_OUT, "\\n" + "=" * 72, flush=True)
        !python pretrain_mlm.py {common} --out {CONTROL_OUT}
        print(f"-> {(time.time() - t0) / 60:.1f} phut")''')

md("""## 4) Fine-tune và so sánh

Ba run cho mỗi task, cùng cài đặt fine-tune:

| run | model |
|---|---|
| gốc | `CONFIG` thuần |
| đối chứng | MLM bản 2, không thêm từ (`CONTROL_OUT`) |
| **mở rộng vocab** | `OUT` |

Tên run tự phân biệt theo thư mục (`_mlm-v2`, `_mlm-vx`).""")

code('''SUF = f'_e{FT_EPOCHS}'
FT  = f'--set training.epochs={FT_EPOCHS} training.early_stopping_patience={FT_EPOCHS}'
MODELS = [None] + [d for d in ([CONTROL_OUT] if RUN_CONTROL else []) + [OUT] if os.path.isfile(f'{d}/config.json')]
for t in TASKS:
    for m in MODELS:
        print("=" * 72, f"\\ntask {t} | {m or CONFIG + ' (goc)'}", flush=True)
        extra = f" model.name={m}" if m else ""
        !python train.py --config configs/{CONFIG}.yaml --task {t} {FT}{extra} --run_suffix {SUF}''')

code('''from src.utils.config import load_config, run_name
names = {}
for t in TASKS:
    for m in MODELS:
        sets = [f"training.epochs={FT_EPOCHS}", f"training.early_stopping_patience={FT_EPOCHS}"] + ([f"model.name={m}"] if m else [])
        names[run_name(load_config(f"configs/{CONFIG}.yaml", sets, task=t, run_suffix=SUF))] = m or "goc"
d = pd.read_csv('results/metrics.csv')
d = d[d.run.isin(names)].assign(model=lambda x: x.run.map(names))
display(d[['task', 'model', 'run', 'macro_f1', 'accuracy', 'best_epoch']].sort_values(['task', 'macro_f1'], ascending=[True, False]))''')

md("""## 5) Tải checkpoint về

Khoảng 1,5 GB mỗi checkpoint. Zip giữ nguyên đường dẫn, nên giải nén ở gốc repo là dùng được.
Tokenizer mở rộng nằm **trong** thư mục checkpoint, nên `train.py` và `inference.py` tự dùng
đúng tokenizer đó.""")

code('''%cd /kaggle/working/repo
for d in [OUT] + ([CONTROL_OUT] if RUN_CONTROL else []):
    if os.path.isdir(d):
        z = f"/kaggle/working/{d.replace('/', '_')}.zip"
        !zip -r -q {z} {d}
        print(f"{z}: {os.path.getsize(z) / 1e9:.2f} GB")''')

md("""## File nộp (val + test)

Mỗi run fine-tune ở trên đã sinh **hai** file nộp trong `results/submissions/`:
`{task}_val_{run}` cho phase Development và `{task}_test_{run}` cho phase Evaluation.
Cell dưới liệt kê chúng và nén `results/` + `logs/` (gồm cả file nộp, **không** gồm checkpoint)
thành một file để tải về.""")
code('''%cd /kaggle/working/repo
import glob, os
subs = sorted(glob.glob('results/submissions/*/submission.zip'))
for split in ('val', 'test'):
    names = [os.path.basename(os.path.dirname(z)) for z in subs if f'_{split}_' in os.path.basename(os.path.dirname(z))]
    print(f"{split}: {len(names)} file nop")
    for n in names:
        print("   ", n)
!zip -r -q /kaggle/working/results_vocab_ext.zip results logs
!ls -lh /kaggle/working/results_vocab_ext.zip''')

write("vocab_ext.ipynb")


# =====================================================================================
# =====================================================================================
#  Notebook 7 -- FINAL, task A: 3 transformers x 3 seeds + 3 TF-IDF, fixed-weight ensemble,
#  fit slice extended with the leak-labelled validation inputs, plus 100%-data versions.
# =====================================================================================
md(f"""# FINAL — Task A (Hate / Non-Hate)

**Settings:** Accelerator = **GPU T4 ×2** · Internet = **On** · khoảng 3–3,5 giờ nếu chạy hết

## Dữ liệu
```
binary_train.csv ── làm sạch, bỏ trùng ──► 6.386 câu ─┬─► HELD-OUT 639 câu   (chấm điểm; KHÔNG BAO GIỜ đổi)
                                                      └─► FIT 5.747 câu
binary_validation_inputs.csv (806, không nhãn)                   │
   gán nhãn theo cấu trúc A–B: id có trong file B → Hate (395)   │
                               ngược lại          → Non-Hate (411)│
   bỏ 14 câu trùng train ──────────────── thêm 792 câu ───────────┘──► FIT 6.539 câu
hastika_binary_test.csv (806) ──► CHỈ để dự đoán. Nhãn test luôn do MODEL đoán.
```

## Model và ensemble
```
            ┌ ① cnerg_muril gốc              × 3 seed ─ TB ┐
câu ──┬──►  ├ ② cnerg_muril + MLM bản 2      × 3 seed ─ TB ┼─ TB ─► P_trans ─┐
      │     └ ③ cnerg_muril + MLM + vocab    × 3 seed ─ TB ┘                 ├─ W·P_trans + (1−W)·P_tfidf ─► nhãn
      └──►    ④ TF-IDF LR   ⑤ TF-IDF SVM   ⑥ TF-IDF SGD ──────── TB ─► P_tfidf ┘
```
- **Tầng 1:** trung bình 3 seed của mỗi transformer. **Tầng 2:** trung bình trong mỗi nhóm.
  **Tầng 3:** trộn hai nhóm với trọng số **cố định** `W` (không tối ưu trên held-out).
- **Bản 90%:** chấm điểm được trên held-out. **Bản 100% (`_full`):** gộp cả held-out vào fit,
  dùng đúng số epoch / C mà bản 90% đã chọn, nên không chấm điểm được.

> ⚠️ **Phải khai báo trong system paper:** nhãn của 806 câu val được **suy ra** từ việc id có mặt
> trong file Task B (gồm cả file B test), không phải nhãn do ban tổ chức công bố. Quy tắc này đúng
> 100% trên A train (3.160/3.160 Hate, 0/3.286 Non-Hate). File nộp **val** của notebook này **vô
> nghĩa để đánh giá** vì model đã học chính các câu đó. File nộp **test** mới là bài nộp.""")

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
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!ls data/raw''')

md("""## 1) Bảng điều khiển""")

code('''# ============================== BANG DIEU KHIEN ==============================
SEEDS    = [42, 43, 44]
EPOCHS   = 10            # fine-tune; luu epoch tot nhat theo held-out
VAL_LEAK = True          # them 792 cau val (nhan suy ra tu file B) vao phan fit
RUN_FULL = True          # them ban 100% (gop held-out vao fit) -- khong cham diem duoc

TRANSFORMERS = [   # (ten, config, model.name -- None = mac dinh cua config)
    ('cnerg',  'cnerg_muril', None),
    ('mlm_v2', 'cnerg_muril', 'checkpoints/mlm_v2/cnerg-muril'),
    ('mlm_vx', 'cnerg_muril', 'checkpoints/mlm_vx/cnerg-muril'),
]
TFIDF   = ['lr', 'svm', 'sgd']
W_TRANS = 0.6            # trong so nhom transformer; TF-IDF = 1 - W_TRANS

# checkpoint MLM: thu muc dich -> (ten file zip, Google Drive: DAN NGUYEN LINK chia se hoac chi file id)
# Link dang https://drive.google.com/file/d/<ID>/view?usp=sharing, che do "Anyone with the link".
MLM_CKPTS = {
    'checkpoints/mlm_v2/cnerg-muril': ('checkpoints_mlm_v2_cnerg-muril.zip', ''),   # <- dan link Drive vao ''
    'checkpoints/mlm_vx/cnerg-muril': ('checkpoints_mlm_vx_cnerg-muril.zip', ''),   # <- dan link Drive vao ''
}
SUFFIX = f'_e{EPOCHS}'

# ---- TANG 2: LLM ----------------------------------------------------------------
USE_LLM    = True            # them LLM (configs/llm_qwen7b.yaml) va ghep 2 tang
LLM_CONFIG = 'llm_qwen7b'    # Qwen2.5-7B-Instruct, 4-bit + LoRA, khung prompt + head "doi tuong"
LLM_DELTA  = 0.2             # cau co |P_ensemble(Hate) - 0,5| < LLM_DELTA = "vung khong chac" -> hoi LLM
LLM_MIX    = 0.5             # trong vung do: P = LLM_MIX * P_ensemble + (1 - LLM_MIX) * P_LLM (0 = chi LLM)
# Ket qua cu (tranh train lai): zip results_final_*.zip tu lan chay truoc, Add Input vao notebook.
# '' = tu tim moi results*.zip trong /kaggle/input.
RESULTS_ZIP = ''
# ===================================================================================''')

md("""## 2) Lấy checkpoint MLM

Hai checkpoint (mỗi cái khoảng 0,9 GB): `mlm_v2/cnerg-muril` (MLM bản 2) và `mlm_vx/cnerg-muril`
(MLM + mở rộng vocab). Cell dưới thử lần lượt:

1. **Kaggle Dataset đã giải nén** (*Add Input*): tìm `…/mlm_v2/cnerg-muril/config.json` ở **mọi độ
   sâu** trong `/kaggle/input`, rồi trỏ tới (symlink, không chép).
2. **File zip** trong `/kaggle/input`, hoặc **tải từ Google Drive** (dán link vào `MLM_CKPTS`).

File zip có thể có **lớp thư mục bọc ngoài** hoặc không, ví dụ
`checkpoints_mlm_v2_cnerg-muril/checkpoints/mlm_v2/cnerg-muril/…` (nén lại sau khi giải nén trên
Windows) hoặc `checkpoints/mlm_v2/cnerg-muril/…` (zip gốc từ Kaggle). Cell giải nén vào thư mục
tạm, **tìm** đúng thư mục chứa `config.json`, rồi chuyển về `checkpoints/mlm_v2/cnerg-muril`. Cả hai
dạng đều chạy được.""")

code('''import glob, zipfile, shutil, json

def find_ckpt(root, tail):
    """thu muc chua config.json co duong dan ket thuc bang `tail`, o bat ky do sau nao"""
    hits = sorted(glob.glob(f"{root}/**/{tail}/config.json", recursive=True), key=len)
    return os.path.dirname(hits[0]) if hits else None

def drive_download(link, out):
    import gdown
    if link.startswith("http"):
        gdown.download(url=link, output=out, quiet=False, fuzzy=True)   # fuzzy: nhan ca link /view?usp=sharing
    else:
        gdown.download(id=link, output=out, quiet=False)
    if not zipfile.is_zipfile(out):
        raise SystemExit(f"{out} khong phai file zip -- kiem tra link Drive da de 'Anyone with the link' chua")

for d, (zname, drive) in MLM_CKPTS.items():
    if os.path.isfile(f"{d}/config.json"):
        print("da co:", d); continue
    tail = d.split("checkpoints/", 1)[1]                       # vd mlm_v2/cnerg-muril
    src = find_ckpt("/kaggle/input", tail)
    if src:                                                    # 1) dataset da giai nen
        os.makedirs(os.path.dirname(d), exist_ok=True)
        os.symlink(src, d)
        print(f"{d} -> {src} (symlink)"); continue
    zips = glob.glob(f"/kaggle/input/**/{zname}", recursive=True)
    if zips:                                                   # 2a) zip trong /kaggle/input
        z, downloaded = zips[0], False
    elif drive:                                                # 2b) tai tu Drive
        z, downloaded = f"/kaggle/working/{zname}", True
        print(f"tai {zname} tu Google Drive ...", flush=True)
        drive_download(drive, z)
    else:
        print(f"!! THIEU {d}: Add Input mot Kaggle Dataset co {zname}, hoac dan link Drive vao MLM_CKPTS")
        continue
    tmp = f"/kaggle/working/_unzip/{tail.replace('/', '_')}"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        print("  trong zip:", sorted({n.split('/')[0] for n in zf.namelist()}))
        zf.extractall(tmp)
    src = find_ckpt(tmp, tail)
    if not src:
        raise SystemExit(f"khong tim thay .../{tail}/config.json trong {z}")
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.move(src, d)
    shutil.rmtree(tmp, ignore_errors=True)
    if downloaded:
        os.remove(z)                                           # 0,9 GB khong can nua
    print(f"{d} <- {src.replace(tmp, '<zip>')}")

# kiem tra: config + tokenizer doc duoc, vocab khop bang embedding
from transformers import AutoTokenizer
for _, _, m in TRANSFORMERS:
    if not m:
        continue
    if not os.path.isfile(f"{m}/config.json"):
        raise SystemExit(f"chua co {m} -- xem cell tren")
    v = json.load(open(f"{m}/config.json"))["vocab_size"]
    tok = AutoTokenizer.from_pretrained(m)
    print(f"OK {m}: vocab {v:,} | tokenizer {len(tok):,} | 'maklu' -> {tok.tokenize('maklu')}")
print("du checkpoint")''')

md("""## 3) Kiểm tra dữ liệu

Chạy đúng hàm mà `train.py` dùng, để thấy trước bao nhiêu câu val được thêm vào và nhãn ra sao.""")

code('''import pandas as pd
from src.utils.config import load_config
from src.data.preprocessing import ensure_processed, add_val_leak
from src.data.dataset import load_split
cfg = load_config("configs/base.yaml", ["data.val_leak_labels=true"], task="a")
ensure_processed(cfg)
tr, va, te = load_split(cfg, "train"), load_split(cfg, "val"), load_split(cfg, "test")
aug = add_val_leak(cfg, tr, va) if VAL_LEAK else tr.assign(source="train")
print(f"\\ntrain goc: fit {int((tr.is_val == 0).sum())} + held-out {int((tr.is_val == 1).sum())}")
print(f"sau khi them val: fit {int((aug.is_val == 0).sum())} + held-out {int((aug.is_val == 1).sum())}  (held-out phai giu nguyen)")
print("nguon cua phan fit:", aug[aug.is_val == 0].source.value_counts().to_dict())
print("test:", len(te), "cau -- chi de du doan")''')

md("""## 4) Kế hoạch

In danh sách run (đúng tên `train.py` sẽ tạo) trước khi train. Run đã xong sẽ tự bỏ qua.""")

code('''from src.utils.config import run_name
fmt = lambda v: str(v).replace(" ", "")
VL = ["data.val_leak_labels=true"] if VAL_LEAK else []

def name_of(conf, sets, seed=None, suffix=None):
    return run_name(load_config(f"configs/{conf}.yaml", sets, task="a", seed=seed, run_suffix=suffix))

JOBS = []      # (nhom, ten model, config, sets, seed, suffix, run name)
for c in TFIDF:
    sets = [f"model.clf={c}"] + VL
    JOBS.append(("tfidf", c, "tfidf", sets, None, None, name_of("tfidf", sets)))
for label, conf, mname in TRANSFORMERS:
    for s in SEEDS:
        sets = [f"training.epochs={EPOCHS}", f"training.early_stopping_patience={EPOCHS}"] + VL \\
               + ([f"model.name={mname}"] if mname else [])
        JOBS.append(("trans", label, conf, sets, s, SUFFIX, name_of(conf, sets, s, SUFFIX)))
for g, label, conf, sets, s, suf, n in JOBS:
    print(f"  {g:6s} {label:7s} seed {s if s else '-':>3}  ->  {n}")
print(f"\\n{len(JOBS)} run (ban 90%)" + (" + ban 100% sau do" if RUN_FULL else ""))''')

md("""## 4b) Kiểm tra kết quả đã có (tránh train lại)

Nếu bạn *Add Input* file zip kết quả của lần chạy trước (vd `results_final_noleak.zip`), cell
dưới **khôi phục** các run trong đó vào `results/a/`. `train.py` tự **bỏ qua** run đã xong (cùng
cấu hình), nên các cell train phía sau **chỉ train phần còn thiếu**:
- đủ hết → chỉ train LLM (nếu `USE_LLM`);
- thiếu run nào → train bù run đó trước, rồi mới train LLM.""")

code('''import os, glob, re, zipfile
zips = [RESULTS_ZIP] if RESULTS_ZIP else sorted(glob.glob("/kaggle/input/**/results*.zip", recursive=True))
restored = 0
for zp in zips:
    with zipfile.ZipFile(zp) as z:
        for member in z.namelist():
            m = re.match(r"(?:.*/)?results/a/([^/]+)/([^/]+)$", member)
            if not m or os.path.exists(f"results/a/{m.group(1)}/{m.group(2)}"):
                continue                                   # khong de len ket qua dang co
            os.makedirs(f"results/a/{m.group(1)}", exist_ok=True)
            with open(f"results/a/{m.group(1)}/{m.group(2)}", "wb") as f:
                f.write(z.read(member))
            restored += 1
print(f"khoi phuc {restored} file tu {len(zips)} zip: {[os.path.basename(z) for z in zips]}")

def is_done(name, scored=True):
    d = f"results/a/{name}"
    return os.path.isfile(f"{d}/test.npy") and os.path.isfile(f"{d}/config.yaml") and \\
           (not scored or os.path.isfile(f"{d}/eval.npy"))
have = [n for *_, n in JOBS if is_done(n)]
miss = [n for *_, n in JOBS if not is_done(n)]
print(f"ban 90%: da co {len(have)}/{len(JOBS)} run" + (f" | THIEU: {miss}" if miss else " -> khong can train lai"))
full_have = sorted(os.path.basename(d) for d in glob.glob("results/a/*_full*") if is_done(os.path.basename(d), False))
print(f"ban 100% da co: {len(full_have)} run")
print("=> " + ("chi train LLM" if USE_LLM and not miss else
               "train bu run thieu" + (" roi train LLM" if USE_LLM else "")))''')

md("""## 5) Train bản 90% (có điểm held-out)

TF-IDF chạy trên CPU trong vài giây. Mỗi run transformer khoảng 12–13 phút (Task A, thêm 13% dữ
liệu), 9 run tổng cộng khoảng 2 giờ. Mỗi run tự sinh file nộp riêng cho val và test.""")

code('''import time
t0 = time.time()
for n, (g, label, conf, sets, s, suf, name) in enumerate(JOBS, 1):
    args = " ".join(sets)
    extra = (f" --seed {s}" if s else "") + (f" --run_suffix {suf}" if suf else "")
    print("=" * 72, f"\\n[{n}/{len(JOBS)}] {name} | +{(time.time() - t0) / 60:.1f} phut", flush=True)
    !python train.py --config configs/{conf}.yaml --task a --set {args}{extra}
print(f"\\nxong trong {(time.time() - t0) / 60:.1f} phut")''')

md("""## 6) Train bản 100% (`_full`)

Gộp cả 639 câu held-out vào phần fit. Không còn gì để chấm điểm, nên mỗi run dùng lại lựa chọn
của bản 90% tương ứng: **C/alpha** tốt nhất (TF-IDF) và **số epoch tốt nhất** (trung vị theo 3
seed của từng transformer). Lịch LR chạy đúng số epoch đó rồi giữ epoch cuối.""")

code('''import json, statistics
FULL_JOBS = []
if RUN_FULL:
    for g, label, conf, sets, s, suf, name in JOBS:
        m = json.load(open(f"results/a/{name}/metrics.json"))
        if g == "tfidf":
            fsets = sets + ["data.use_valdataset=false", f"model.param_grid=[{m['best_C']}]"]
        else:
            eps = [json.load(open(f"results/a/{n2}/metrics.json"))["best_epoch"]
                   for g2, l2, *_, n2 in JOBS if g2 == "trans" and l2 == label]
            ep = max(1, int(round(statistics.median(eps))))
            fsets = [x for x in sets if not x.startswith("training.")] + \\
                    [f"training.epochs={ep}", f"training.early_stopping_patience={ep}", "data.use_valdataset=false"]
        FULL_JOBS.append((g, label, conf, fsets, s, suf, name_of(conf, fsets, s, suf)))
    t0 = time.time()
    for n, (g, label, conf, sets, s, suf, name) in enumerate(FULL_JOBS, 1):
        args = " ".join(sets)
        extra = (f" --seed {s}" if s else "") + (f" --run_suffix {suf}" if suf else "")
        print("=" * 72, f"\\n[{n}/{len(FULL_JOBS)}] {name} | +{(time.time() - t0) / 60:.1f} phut", flush=True)
        !python train.py --config configs/{conf}.yaml --task a --set {args}{extra}
    print(f"\\nxong trong {(time.time() - t0) / 60:.1f} phut")''')

md("""## 6b) Tầng 2 — LLM (nếu `USE_LLM`)

Qwen2.5-7B-Instruct nạp **4-bit**, chỉ train **LoRA** (khoảng 40M tham số) + 2 head: **Hate/Non-Hate**
và **"đối tượng bị tấn công"** (nhóm của Task B; chỉ dùng khi train). Mỗi bình luận được bọc trong
khung prompt cố định (xem `configs/llm_qwen7b.yaml`). Chạy trên 1 T4, khoảng 1–1,5 giờ. Đã có kết
quả (từ zip) thì tự bỏ qua.""")

code('''LLM_NAME = None
if USE_LLM:
    !pip -q install peft bitsandbytes accelerate
    llm_sets = VL
    LLM_NAME = name_of(LLM_CONFIG, llm_sets)
    print("LLM run:", LLM_NAME, "(da co -> bo qua)" if is_done(LLM_NAME) else "")
    t0 = time.time()
    args = " ".join(llm_sets)
    !python train.py --config configs/{LLM_CONFIG}.yaml --task a {"--set " + args if args else ""}
    print(f"-> {(time.time() - t0) / 60:.1f} phut")
    if not is_done(LLM_NAME):
        raise SystemExit("LLM chua co ket qua -- xem log o tren (OOM? thieu peft/bitsandbytes?)")''')

md("""## 7) Kết quả trên held-out (chỉ bản 90%)

Từng run, trung bình seed của từng model, từng nhóm, ensemble cuối, và các biến thể bỏ bớt một
model. Lát held-out có 639 câu, nên chênh lệch dưới khoảng 0,02 là trong mức nhiễu. **Đừng chọn
`W` theo bảng này**: đó chính là tối ưu trên held-out mà ta đang tránh.""")

code('''import numpy as np
from sklearn.metrics import f1_score
y = tr.y.to_numpy()[tr.is_val.to_numpy() == 1]
f1 = lambda p: f1_score(y, p.argmax(1), average="macro")
P = lambda name, split="eval": np.load(f"results/a/{name}/{split}.npy")

rows = [{"run": n, "nhom": g, "model": l, "macro_f1": round(f1(P(n)), 4)} for g, l, *_, n in JOBS]
display(pd.DataFrame(rows))

def model_probs(split, jobs, label):
    return np.mean([P(n, split) for g, l, *_, n in jobs if l == label], axis=0)
def group_probs(split, jobs, group, drop=()):
    labels = [l for l, *_ in TRANSFORMERS] if group == "trans" else TFIDF
    return np.mean([model_probs(split, jobs, l) for l in labels if l not in drop], axis=0)
def ensemble(split, jobs, w=W_TRANS, drop=()):
    return w * group_probs(split, jobs, "trans", drop) + (1 - w) * group_probs(split, jobs, "tfidf", drop)

summary = [{"to hop": f"{l} (TB {len(SEEDS)} seed)", "macro_f1": f1(model_probs("eval", JOBS, l))} for l, *_ in TRANSFORMERS]
summary += [{"to hop": "nhom transformer", "macro_f1": f1(group_probs("eval", JOBS, "trans"))},
            {"to hop": "nhom TF-IDF", "macro_f1": f1(group_probs("eval", JOBS, "tfidf"))},
            {"to hop": f"ENSEMBLE (W={W_TRANS})", "macro_f1": f1(ensemble("eval", JOBS))}]
summary += [{"to hop": f"ensemble bo {d}", "macro_f1": f1(ensemble("eval", JOBS, drop=(d,)))}
            for d in [l for l, *_ in TRANSFORMERS] + TFIDF]
summary += [{"to hop": f"(tham khao) W={w}", "macro_f1": f1(ensemble("eval", JOBS, w))} for w in (0.4, 0.5, 0.7, 0.8)]
display(pd.DataFrame(summary).round(4))''')

md("""**Ghép 2 tầng** (chỉ khi `USE_LLM`). Câu nào ensemble **chắc chắn** (|P(Hate) − 0,5| ≥ `LLM_DELTA`)
giữ nguyên dự đoán của ensemble; câu nào **không chắc** thì trộn với LLM theo `LLM_MIX`.

Bảng quan trọng nhất là **độ chính xác TRONG vùng không chắc**: ensemble và LLM, trên cùng các
câu. LLM chỉ có ích nếu nó đúng nhiều hơn ensemble ở đúng vùng đó.""")

code('''def cascade(split, jobs, delta=LLM_DELTA, mix=LLM_MIX):
    p1 = ensemble(split, jobs)
    p2 = P(LLM_NAME, split)
    band = np.abs(p1[:, 1] - 0.5) < delta
    p = p1.copy()
    p[band] = mix * p1[band] + (1 - mix) * p2[band]
    return p, band

if USE_LLM:
    p1, p2 = ensemble("eval", JOBS), P(LLM_NAME)
    acc = lambda p, m: (p[m].argmax(1) == y[m]).mean() if m.any() else float("nan")
    rows = [{"to hop": "ENSEMBLE (tang 1)", "macro_f1": f1(p1)},
            {"to hop": "LLM mot minh", "macro_f1": f1(p2)},
            {"to hop": f"ensemble + LLM nhu thanh vien thu 7 (TB 0,5/0,5)", "macro_f1": f1(0.5 * p1 + 0.5 * p2)}]
    for d in (0.1, 0.2, 0.3, 0.5):
        pc, band = cascade("eval", JOBS, delta=d)
        rows.append({"to hop": f"CASCADE delta={d}" + ("  <- dang dung" if d == LLM_DELTA else ""),
                     "macro_f1": f1(pc), "cau vao vung khong chac": int(band.sum()),
                     "acc ensemble trong vung": round(acc(p1, band), 3), "acc LLM trong vung": round(acc(p2, band), 3)})
    display(pd.DataFrame(rows).round(4))
    print("Luu y: chon delta theo bang nay la toi uu tren held-out (639 cau) -> chi nen doi neu chenh ro.")''')

md("""## 8) File nộp

Ensemble với cùng trọng số cho:
- **bản 90%** → `a_test_final` (và `a_val_final`)
- **bản 100%** → `a_test_final_full` (và `a_val_final_full`)
- nếu `USE_LLM`: **2 tầng** → `a_test_final_llm` và `a_test_final_full_llm`

**Nộp file `test`.** File `val` chỉ sinh ra cho đủ bộ: model đã học chính các câu đó.""")

code('''from src.evaluation.submission import write_submission
from src.data.dataset import label_names
labels = label_names("a")
outs = {}
for tag, jobs in [("final", JOBS)] + ([("final_full", FULL_JOBS)] if RUN_FULL else []):
    for split, df in (("val", va), ("test", te)):
        p = ensemble(split, jobs)
        outs[(tag, split)] = p
        write_submission(df.id.values, p, labels, f"results/submissions/a_{split}_{tag}")
if USE_LLM:
    # 2 tang: ensemble (90% hoac 100%) + LLM (ban 90%) trong vung khong chac
    for tag, jobs in [("final_llm", JOBS)] + ([("final_full_llm", FULL_JOBS)] if RUN_FULL else []):
        for split, df in (("val", va), ("test", te)):
            p, band = cascade(split, jobs)
            outs[(tag, split)] = p
            write_submission(df.id.values, p, labels, f"results/submissions/a_{split}_{tag}")
            if split == "test":
                changed = int((p.argmax(1) != ensemble(split, jobs).argmax(1)).sum())
                print(f"  {tag}: {int(band.sum())} cau test vao vung khong chac, LLM doi {changed} nhan")
if RUN_FULL:
    a, b = outs[("final", "test")].argmax(1), outs[("final_full", "test")].argmax(1)
    print(f"\\nban 90% va ban 100% dong y tren {(a == b).mean():.1%} cau test")''')

md("""## 9) Tải về""")

code('''%cd /kaggle/working/repo
!mkdir -p /kaggle/working/final_submissions
!cp -r results/submissions/a_test_final* results/submissions/a_val_final* /kaggle/working/final_submissions/
!ls -R /kaggle/working/final_submissions | head -30
!zip -r -q /kaggle/working/results_final.zip results logs
!ls -lh /kaggle/working/results_final.zip''')

write("final.ipynb")
