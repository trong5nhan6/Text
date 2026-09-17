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
    ("training.batch_size", None),
    ("training.grad_accum", "tang len khi giam batch_size, de giu batch hieu dung"),
    ("training.loss", "auto | ce | wce | focal   (auto: task a -> ce, task b -> wce)"),
    ("training.label_smoothing", "bi bo qua khi loss=focal"),
    ("--- du lieu ---", None),
    ("data.max_len", "Religion / Geo-political dai hon, hay bi cat o 96"),
    ("--- model ---", None),
    ("model.pooling", "cls | mean"),
    ("model.dropout", None),
    ("model.unfreeze_last_n_blocks", "null = train tat ca | 4 = chi 4 khoi cuoi | 0 = chi head"),
    ("model.freeze_embeddings", "null = theo khoa tren | true | false"),
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
RUN_SUFFIX = ''        # vi du '_e8' -> run ten muril_wce_s42_e8. BAT BUOC khi doi gia tri that su.

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
md("""## 2) B0 — TF-IDF (CPU, ~30 giây mỗi run)

`C` là tham số điều chuẩn của mô hình tuyến tính, theo nghĩa **nghịch đảo**: `C` nhỏ = phạt trọng số
mạnh = model đơn giản (dễ underfit); `C` lớn = ưu tiên khớp dữ liệu = dễ overfit. Ở đây đặc trưng
TF-IDF có tới 300k chiều trên vài nghìn dòng train, nên `C` là siêu tham số quan trọng nhất của B0.

`train.py` fit một model cho **mỗi giá trị trong `model.C_grid`** rồi giữ cái có macro-F1 cao nhất
trên lát held-out. Giá trị được chọn hiện trong `results/metrics.csv`, ví dụ `tfidf-lr C=2`.

LR và SVM **không cùng thang `C`** (loss khác nhau), nên SVM dùng dải nhỏ hơn. Chạy cả hai vì blend
của chúng thường tốt hơn từng cái một.""")
code('''!python train.py --config configs/tfidf.yaml --task a
!python train.py --config configs/tfidf.yaml --task b''')

code('''# LinearSVC — run tự đặt tên tfidf_svm, dải C nhỏ hơn LR
!python train.py --config configs/tfidf.yaml --task a --set model.clf=svm "model.C_grid=[0.05,0.1,0.25,0.5,1]"
!python train.py --config configs/tfidf.yaml --task b --set model.clf=svm "model.C_grid=[0.05,0.1,0.25,0.5,1]"''')

code('''# Dò C mịn hơn quanh giá trị vừa chọn (nhớ đổi --run_name, nếu không script từ chối chạy):
# !python train.py --config configs/tfidf.yaml --task b --run_name tfidf_lr_fine --set "model.C_grid=[0.25,0.5,1,1.5,2,3,4]"''')

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

md("""**Task B — thử `ce` thay cho `wce`.** Mặc định Task B dùng weighted CE để bù lệch lớp 7,3 lần.
Nhưng ở lần chạy trước `roberta_wce_s42` chỉ đạt accuracy 0,6444 — thấp bất thường, dấu hiệu class
weight đẩy quá tay sang lớp hiếm và bào mòn lớp lớn. `ce` là đối chứng cần có.

Không cần `--run_suffix`: tên run đã chứa loss, nên chúng nằm ở `results/b/<config>_ce_s42`,
tách hẳn với `<config>_wce_s42`.""")
code('''for c in CONFIGS:
    print("")
    print("=" * 72); print(f"task b | {c} | loss=ce"); print("=" * 72, flush=True)
    !python train.py --config configs/{c}.yaml --task b --set training.loss=ce''')

code('''# Ví dụ biến thể:
# !python train.py --config configs/muril.yaml --task b --set training.loss=focal
# !python train.py --config configs/muril.yaml --task b --set data.max_len=128 --run_name muril_len128
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
├── b/                              Task B  (loss mặc định là wce nên tên là _wce_)
│   ├── tfidf_lr/  tfidf_svm/  muril_wce_s42/  roberta_wce_s42/
│   └── _blend/                     kết quả blend gần nhất
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
# !python inference.py --task b --checkpoints checkpoints/b/muril_wce_s42 --input data/raw/multiclass_test_inputs.csv --tag ens1''')

code('''import glob, shutil
out = '/kaggle/working/submissions'
os.makedirs(out, exist_ok=True)
for z in glob.glob('results/submissions/*/submission.zip'):
    shutil.copy(z, f"{out}/{os.path.basename(os.path.dirname(z))}.zip")
!ls -la /kaggle/working/submissions''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "hastika_kaggle.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {out} ({len(cells)} cells, repo {REPO}@{BRANCH})")
