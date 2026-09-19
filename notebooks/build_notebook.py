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
    ("training.batch_size", None),
    ("training.grad_accum", "tang len khi giam batch_size, de giu batch hieu dung"),
    ("training.loss", "auto | ce | wce | focal   (auto: task a -> ce, task b -> focal)"),
    ("training.label_smoothing", "ap dung cho ca ce, wce va focal"),
    ("--- du lieu ---", None),
    ("data.text_type", "null/latin = chu Latin | kn = chu Kannada | both = ca hai"),
    ("data.tta", "CHI transformer: infer ca 2 chu viet roi trung binh (tfidf se bi tu choi)"),
    ("data.max_len", "Religion / Geo-political dai hon, hay bi cat o 96"),
    ("--- model ---", None),
    ("model.pooling", "cls | mean"),
    ("model.dropout", None),
    ("model.unfreeze_last_n_blocks", "null = train tat ca | 4 = chi 4 khoi cuoi | 0 = chi head"),
    ("model.freeze_embeddings", "null = theo khoa tren | true | false"),
    ("model.head", "linear | sparse_moe | soft_moe  (soft_moe bo qua model.pooling)"),
    ("model.moe_experts", "so expert; head linear chi co 4.614 tham so, MoE E=4 d=64 la ~201k"),
    ("model.moe_expert_dim", None),
    ("seed", None),
    ("--- dia ---", None),
    ("checkpoint.save", "best | none   (none tiet kiem ~0.5 GB moi run)"),
]


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
        v = default_of(key)
        v = f"'{v}'" if isinstance(v, str) else repr(v)
        entry = f"    # '{key}':".ljust(width + 8) + f"{v},"
        lines.append(f"{entry.ljust(width + 20)}# {note}" if note else entry)
    lines.append("}")
    return "\n".join(lines) + """
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
ARGS = " ".join(f"{k}={v}" for k, v in OVERRIDES.items())
ARGS = (f"--set {ARGS}" if ARGS else "") + (f" --run_suffix {RUN_SUFFIX}" if RUN_SUFFIX else "")

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

CLFS       = ['lr', 'svm', 'ridge', 'cnb', 'sgd']
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

CONFIGS = ['muril', 'roberta', 'indicbert', 'bert', 'deberta', 'modernbert']  # bo bot neu thieu gio
# Ban large (24 block, ~500M): them 'muril_large' / 'roberta_large'. Cham hon ~3 lan va
# hay sup ve mot lop tren du lieu nho -- doc history.json truoc khi tin con so cuoi.
TASKS   = ['a', 'b']

t0 = time.time()
for i, c in enumerate(CONFIGS):
    for j, t in enumerate(TASKS):
        n = i * len(TASKS) + j + 1
        print("")
        print("=" * 72)
        print(f"[{n}/{len(CONFIGS) * len(TASKS)}]  config = {c}   |   task = {t}   "
              f"|   {time.strftime('%H:%M:%S')}   |   +{(time.time() - t0) / 60:.1f} phut")
        print("=" * 72, flush=True)
        !python train.py --config configs/{c}.yaml --task {t} {ARGS}
print("")
print(f"xong {len(CONFIGS) * len(TASKS)} run trong {(time.time() - t0) / 60:.1f} phut")''')

md("""> **Loss:** vòng lặp trên đã dùng đúng loss cho từng task qua `training.loss: auto` —
> Task A `ce` (nhãn 51/49, không cần bù), Task B **`focal`** (gamma 2 + class weight `sqrt_inv`).
> Task B lệch **7,3 lần**: Gender 1.362 (43,1%) so với Geo-political 186 (5,9%), mà macro-F1 chấm
> cả 6 lớp ngang nhau. Focal hạ trọng số những mẫu đã dễ và dồn gradient vào mẫu khó.""")

code('''# Ví dụ biến thể:
# !python train.py --config configs/muril.yaml --task b --set training.focal_gamma=3 --run_suffix _g3
# !python train.py --config configs/muril.yaml --task b --set data.max_len=128 --run_name muril_len128
# !python train.py --config configs/muril.yaml --task b --set data.text_type=both --set data.tta=true
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
