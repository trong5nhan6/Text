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

Mỗi config × task ≈ **2–4 phút** trên T4 (4 epoch, early stopping patience 2).
- Run đã xong → chạy lại sẽ bỏ qua, không train lại.
- Đổi siêu tham số → thêm `--run_name <tên mới>` (hoặc `--overwrite`), nếu không script sẽ từ chối chạy.
- Mỗi run lưu 1 checkpoint fp16 (~0.5 GB với model base); `/kaggle/working` giới hạn ~20 GB.""")
code('''import time

CONFIGS = ['muril', 'roberta']        # thêm: 'indicbert', 'bert', 'deberta', 'modernbert'
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
        !python train.py --config configs/{c}.yaml --task {t}
print("")
print(f"xong {len(CONFIGS) * len(TASKS)} run trong {(time.time() - t0) / 60:.1f} phut")''')

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

code('''# blend + tối ưu trọng số trên lát eval (đổi tên run theo bảng trên)
!python evaluate.py --task a --runs tfidf_lr tfidf_svm muril_ce_s42 roberta_ce_s42 --optimize
!python evaluate.py --task b --runs tfidf_lr tfidf_svm muril_wce_s42 roberta_wce_s42 --optimize''')

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

md("**Cách 1 — mỗi run một file nộp riêng** (để đối chiếu, đặt tên theo task + run):")
code('''import glob, os
for t in ('a', 'b'):
    for f in sorted(glob.glob(f'results/{t}/*/val.npy')):
        run = os.path.basename(os.path.dirname(f))
        !python inference.py --task {t} --runs {run} --split val --tag {run}''')

md("**Cách 2 — ensemble** (thường tốt hơn; đổi tên run và trọng số theo bảng ở mục 4):")
code('''!python inference.py --task a --runs tfidf_lr tfidf_svm muril_ce_s42 roberta_ce_s42 --split val --tag ens3
!python inference.py --task b --runs tfidf_lr tfidf_svm muril_wce_s42 roberta_wce_s42 --split val --tag ens3
# test, từ checkpoint:
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
