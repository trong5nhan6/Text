#!/usr/bin/env python3
"""
Regenerate notebooks/hastika_kaggle.ipynb from the current source tree.
Every project file is embedded with %%writefile, so the notebook runs on Kaggle
without uploading anything. Run this again after editing code/configs:

  python notebooks/build_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = {"src/eda"}                       # EDA runs locally, not on Kaggle
FILES = sorted(
    [p for p in (ROOT / "src").rglob("*.py")
     if not any(p.relative_to(ROOT).as_posix().startswith(e) for e in EXCLUDE)]
    + sorted((ROOT / "configs").glob("*.yaml"))
    + [ROOT / f for f in ("train.py", "evaluate.py", "inference.py", "requirements.txt")]
)

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# HASTIKA @ ICON-2026 — Kaggle training notebook
**Settings (panel bên phải):** Accelerator = **GPU T4** · Internet = **On** · Persistence = *Files only* (nếu có).

Quy trình: 1) setup → 2) data → 3) TF-IDF → 4) transformers → 5) evaluate → 6) submission.
Toàn bộ kết quả nằm trong `/kaggle/working/hastika` và được lưu lại khi bạn **Save Version (Save & Run All)**.

> Code trong notebook được sinh tự động từ repo bằng `notebooks/build_notebook.py` — sửa code ở repo rồi build lại, đừng sửa trực tiếp ở đây.""")

code("""import os, subprocess, sys
WORK = '/kaggle/working/hastika' if os.path.exists('/kaggle') else os.path.abspath('hastika')
os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
for d in ['configs', 'data/raw', 'data/processed', 'src/data', 'src/models', 'src/training',
          'src/evaluation', 'src/utils', 'checkpoints', 'logs', 'results']:
    os.makedirs(d, exist_ok=True)
print(WORK)""")

md("## 1) Source code")
for f in FILES:
    rel = f.relative_to(ROOT).as_posix()
    code(f"%%writefile {rel}\n" + f.read_text(encoding="utf-8"))

code("""!pip -q install ftfy sentencepiece tiktoken
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())""")

md("""## 2) Data
Lấy dữ liệu từ repo của ban tổ chức. **Khi test được phát hành (20/9):** chạy lại cell này (git pull) hoặc upload
file `*test*.csv` vào `data/raw/` (ví dụ từ một Kaggle Dataset: `!cp /kaggle/input/<dataset>/*test*.csv data/raw/`).""")
code("""if not os.path.exists('_organiser'):
    !git clone -q https://github.com/shankarb14/Hastika-ICON2026.git _organiser
else:
    !git -C _organiser pull -q
!cp _organiser/data/*.csv data/raw/
!ls data/raw
!python -m src.data.preprocessing""")

md("## 3) B0 — TF-IDF (CPU, vài phút)")
code("""!python train.py --config configs/tfidf.yaml --task a
!python train.py --config configs/tfidf.yaml --task b""")

md("""## 4) B1 — Transformers
Mặc định là **holdout 90/10** (`data.n_folds: 1`): mỗi config × task ≈ **2–4 phút** trên T4
(1 lát × 4 epoch, early stopping). Muốn 5-fold thì thêm `--set data.n_folds=5` (≈ 10–20 phút).
- Bị ngắt giữa chừng → chạy lại đúng lệnh, lát/fold đã xong được bỏ qua.
- Đổi siêu tham số → thêm `--run_name <tên mới>` (hoặc `--overwrite`).
- Mỗi lát lưu 1 checkpoint fp16 (~0.5 GB với model base). `/kaggle/working` giới hạn ~20 GB → xoá run không dùng (cell cuối mục này).""")
code("""CONFIGS = ['muril', 'roberta']            # thêm: 'indicbert', 'bert', 'deberta', 'modernbert'
TASKS = ['a', 'b']
for c in CONFIGS:
    for t in TASKS:
        !python train.py --config configs/{c}.yaml --task {t}""")
code("""# Ví dụ biến thể:
# !python train.py --config configs/muril.yaml --task b --set training.loss=focal
# !python train.py --config configs/muril.yaml --task a --seed 7
# !python train.py --config configs/roberta.yaml --task b --run_name xlmr_large_wce \\
#       --set model.name=xlm-roberta-large training.lr=1e-5 training.batch_size=16 training.grad_accum=2""")
code("""!du -sh checkpoints/*/* 2>/dev/null; df -h /kaggle/working | tail -1
# xoá checkpoint của run không cần:  !rm -rf checkpoints/b/<run_name>""")

md("## 5) Evaluate (OOF)")
code("""import pandas as pd
display(pd.read_csv('results/metrics.csv'))
!python evaluate.py --task a
!python evaluate.py --task b""")
code("""# blend + tối ưu trọng số trên OOF (đổi tên run theo bảng trên)
!python evaluate.py --task a --runs tfidf_lr_h10 muril_ce_s42_h10 roberta_ce_s42_h10 --optimize
!python evaluate.py --task b --runs tfidf_lr_h10 muril_wce_s42_h10 roberta_wce_s42_h10 --optimize""")
code("""from IPython.display import Image
Image('results/b/_blend/confusion.png')""")

md("""## 6) Submission
- **Development phase (val):** mode 1 dùng xác suất đã lưu.
- **Evaluation phase (test):**
  - run train *sau* khi có test → `--split test`
  - run train *trước* khi có test → mode 2 dùng checkpoint (`--checkpoints ... --input ...`), không cần train lại.""")
code("""!python inference.py --task a --runs tfidf_lr_h10 muril_ce_s42_h10 roberta_ce_s42_h10 --split val --tag ens3
!python inference.py --task b --runs tfidf_lr_h10 muril_wce_s42_h10 roberta_wce_s42_h10 --split val --tag ens3
# test, từ checkpoint:
# !python inference.py --task b --checkpoints checkpoints/b/muril_wce_s42_h10 checkpoints/b/roberta_wce_s42_h10 \\
#       --input data/raw/multiclass_test_inputs.csv --tag ens2
!find results/submissions -name submission.zip""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "hastika_kaggle.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {out} ({len(cells)} cells, {len(FILES)} embedded files)")
