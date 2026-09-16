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
md("## 2) B0 — TF-IDF (CPU, ~30 giây mỗi task)")
code('''!python train.py --config configs/tfidf.yaml --task a
!python train.py --config configs/tfidf.yaml --task b''')

# ---------------------------------------------------------- 3) transformers
md("""## 3) B1 — Transformers

Mỗi config × task ≈ **2–4 phút** trên T4 (4 epoch, early stopping patience 2).
- Run đã xong → chạy lại sẽ bỏ qua, không train lại.
- Đổi siêu tham số → thêm `--run_name <tên mới>` (hoặc `--overwrite`), nếu không script sẽ từ chối chạy.
- Mỗi run lưu 1 checkpoint fp16 (~0.5 GB với model base); `/kaggle/working` giới hạn ~20 GB.""")
code('''CONFIGS = ['muril', 'roberta']        # thêm: 'indicbert', 'bert', 'deberta', 'modernbert'
TASKS   = ['a', 'b']
for c in CONFIGS:
    for t in TASKS:
        !python train.py --config configs/{c}.yaml --task {t}''')

code('''# Ví dụ biến thể:
# !python train.py --config configs/muril.yaml --task b --set training.loss=focal
# !python train.py --config configs/muril.yaml --task b --set data.max_len=128 --run_name muril_len128
# !python train.py --config configs/roberta.yaml --task b --run_name xlmr_large \\
#       --set model.name=xlm-roberta-large training.lr=1e-5 training.batch_size=16 training.grad_accum=2''')

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
!python evaluate.py --task a --runs tfidf_lr muril_ce_s42 roberta_ce_s42 --optimize
!python evaluate.py --task b --runs tfidf_lr muril_wce_s42 roberta_wce_s42 --optimize''')

code('''from IPython.display import Image, display
for t in ('a', 'b'):
    p = f'results/{t}/_blend/confusion.png'
    if os.path.exists(p):
        print(p); display(Image(p))''')

# ------------------------------------------------------------ 5) submission
md("""## 5) Submission

- **Development phase (val):** mode 1, dùng xác suất đã lưu.
- **Evaluation phase (test):** run train *sau* khi có test → `--split test`;
  run train *trước* khi có test → mode 2 dùng checkpoint, không cần train lại.""")
code('''!python inference.py --task a --runs tfidf_lr muril_ce_s42 roberta_ce_s42 --split val --tag ens3
!python inference.py --task b --runs tfidf_lr muril_wce_s42 roberta_wce_s42 --split val --tag ens3
# test, từ checkpoint:
# !python inference.py --task b --checkpoints checkpoints/b/muril_wce_s42 \\
#       --input data/raw/multiclass_test_inputs.csv --tag ens1''')

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
