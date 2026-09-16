# HASTIKA @ ICON-2026 — Kanglish hate speech classification

Pipeline cho hai subtask của [HASTIKA @ ICON-2026](https://www.codabench.org/competitions/17784/):

- **Task A:** phân loại nhị phân Hate / Non-Hate.
- **Task B:** phân loại 6 nhóm đối tượng bị nhắm tới: Gender, Political, Religion, Geo-political, Violence, Others.

Metric xếp hạng là **macro-F1**.
Thể lệ gốc của ban tổ chức nằm trong `docs/shared_task_README.md` và `docs/FORMAT.md`.

## Cấu trúc

Bốn tầng tách bạch: **code** (`src/`, entrypoint ở gốc) · **cấu hình** (`configs/`) · **dữ liệu** (`data/`) ·
**tài liệu** (`docs/`). Mọi thứ do máy sinh ra (`data/processed/`, `results/`, `checkpoints/`, `logs/`)
đều bị git bỏ qua vì tái tạo được từ code + config.

```
train.py  evaluate.py  inference.py     entrypoint
configs/                base.yaml (mặc định) + mỗi model 1 file (_base_: base.yaml)
  tfidf.yaml  muril.yaml  bert.yaml (mBERT)  roberta.yaml (XLM-R)  deberta.yaml (mDeBERTa-v3)
  indicbert.yaml  modernbert.yaml (English-only, dùng làm ablation)
src/
  data/                 preprocessing.py (làm sạch, gộp câu trùng, StratifiedGroupKFold) · dataset.py
  models/               factory.py · classifier.py (encoder + pooling + head, lưu/nạp checkpoint) · tfidf.py
  training/             trainer.py (AMP, grad-accum, early stopping) · losses.py (ce / wce / focal)
  evaluation/           metrics.py (macro-F1, per-class, confusion, results/metrics.csv)
  eda/                  eda.py — sinh lại báo cáo EDA (chỉ chạy local)
  utils/                config.py (YAML + --set) · logger.py · seed.py
data/
  raw/                  CSV của ban tổ chức (+ *test*.csv khi phát hành)     [tracked]
  processed/            text đã làm sạch + cột is_val (sinh tự động)          [ignored]
docs/
  shared_task_README.md  FORMAT.md  LICENSE_NOTE.md   thể lệ gốc của ban tổ chức
  eda/README.md + figures/ + stats/                   báo cáo EDA
  starting_kit/                                       sample submission + baseline của BTC
notebooks/              hastika_kaggle.ipynb (clone repo + chạy trên Kaggle) + build_notebook.py
results/                                                                      [ignored]
  metrics.csv                       bảng tổng hợp mọi run
  {task}/{run}/                     eval.npy val.npy [test.npy] metrics.json config.yaml per_class.csv confusion.png
  submissions/{task}_{split}_{tag}/ predictions.csv + submission.zip (nộp Codabench)
checkpoints/{task}/{run}/           trọng số của epoch tốt nhất (fp16)          [ignored]
logs/{task}_{run}.log                                                         [ignored]
```

## Cài đặt
```bash
pip install -r requirements.txt
python -m src.data.preprocessing          # tạo data/processed (train.py cũng tự gọi nếu thiếu)
python -m src.eda.eda                     # (tuỳ chọn) sinh lại báo cáo EDA ở docs/eda/
```

## Chia dữ liệu

Một lát duy nhất, chia một lần trong `src/data/preprocessing.py` và dùng chung cho mọi model:

| | |
|---|---|
| **Fit** (`is_val == 0`) | 90% — A 5.747 dòng · B 2.827 dòng |
| **Eval** (`is_val == 1`) | 10% — A 639 dòng · B 315 dòng, phân tầng theo nhãn |

```yaml
data:
  val_ratio: 0.1      # kích thước lát held-out
  split_seed: 42      # giữ cố định -> mọi run chung một lát -> blend được
```

Mọi run lưu `eval.npy` cùng số dòng và cùng thứ tự, nên `evaluate.py` so sánh và blend trực tiếp
được; nếu một run được train trên lát khác thì script báo lỗi thay vì so nhầm.
Đổi `val_ratio` hoặc `split_seed` thì `data/processed/` **tự sinh lại** (đối chiếu với `{task}_split.json`).

> **Lát eval khá mỏng.** Task B chỉ có 315 dòng để chấm, trong đó Geo-political 18 dòng và
> Violence 22 dòng — sai 1 mẫu ở lớp hiếm đã làm macro-F1 xê dịch ~0,5 điểm. Chênh lệch
> dưới ~0,02 giữa hai model không phân biệt được; đừng đuổi theo con số thập phân thứ ba.

## Train
```bash
python train.py --config configs/tfidf.yaml --task a
python train.py --config configs/muril.yaml --task b
python train.py --config configs/roberta.yaml --task b --set training.loss=focal --run_name roberta_focal
```
- **Tên run mặc định** là `<config>_<loss>_s<seed>`, ví dụ `muril_wce_s42`; riêng TF-IDF là `tfidf_lr`.
- **Run đã xong** thì chạy lại sẽ bỏ qua, không train lại.
- **Đổi siêu tham số:** nếu giữ nguyên tên run, script sẽ **từ chối chạy**. Hãy dùng `--run_name` khác hoặc thêm `--overwrite`.
- **Ghi đè cấu hình từ dòng lệnh:** mọi khoá trong YAML đều đổi được bằng `--set khoa.con=gia_tri`.
- **Loss mặc định:** Task A dùng `ce`, Task B dùng `wce` (class weight = căn bậc hai của nghịch đảo tần suất).

## Đánh giá
```bash
python evaluate.py --task b                                              # mọi run của task b
python evaluate.py --task b --runs tfidf_lr muril_wce_s42 roberta_wce_s42 --optimize   # blend + tìm trọng số
```

## Tạo file nộp
```bash
# mode 1: xác suất đã lưu (val cho phase Development, test nếu run train sau khi có test)
python inference.py --task b --runs tfidf_lr muril_wce_s42 --weights 1 2 --split val
# mode 2: từ checkpoint, cho run train trước khi có test (không cần train lại)
python inference.py --task b --checkpoints checkpoints/b/muril_wce_s42 --input data/raw/multiclass_test_inputs.csv
```

## Kaggle
`notebooks/hastika_kaggle.ipynb` **clone repo này rồi gọi các entrypoint** — nó không chứa bản sao
code nào, nên sửa code ở máy chỉ cần `git push` là xong.

1. kaggle.com/code ▸ New Notebook ▸ *File ▸ Import Notebook* ▸ chọn `notebooks/hastika_kaggle.ipynb`
2. Panel phải: **Accelerator = GPU T4 ×2**, **Internet = On** (bắt buộc: cần cho `git clone` và tải model)
3. *Run All*. Muốn chạy nền và giữ output thì **Save Version ▸ Save & Run All**

- **Cập nhật code:** `git push` ở máy → chạy lại cell đầu tiên trên Kaggle (`git pull`). Không phải upload lại notebook.
- **Repo phải Public**, hoặc nếu để Private thì tạo GitHub PAT (scope `repo`) và lưu vào
  *Add-ons ▸ Secrets* với tên `GH_TOKEN` — cell đầu tự dò và che token khỏi log.
- Repo được clone vào `/kaggle/working/repo`, nên `results/` và `checkpoints/` nằm trong output của notebook.
  Cell cuối copy các `submission.zip` ra `/kaggle/working/submissions/` cho dễ tải về.
- Sửa nội dung notebook: sửa `notebooks/build_notebook.py` rồi chạy `python notebooks/build_notebook.py`.

## Khi có test (20/9)
1. Đặt `*test*.csv` vào `data/raw/`, rồi chạy `python -m src.data.preprocessing`. Fold train giữ nguyên.
2. Run TF-IDF: chạy lại `train.py` (vài phút).
3. Run transformer đã có checkpoint: dùng `inference.py --checkpoints ... --input ...`.

## Kết quả hiện tại (lát held-out 10%)
| Task | Run | Macro-F1 | Acc | n_eval |
|---|---|---|---|---|
| A | tfidf_lr (C=2) | 0,8278 | 0,8279 | 639 |
| B | tfidf_lr (C=2, balanced) | 0,6485 | 0,7270 | 315 |
| B | tfidf_svm (C=0,1, balanced) | 0,6439 | 0,7270 | 315 |
| B | blend lr 0,6 + svm 0,4 | **0,6587** | 0,7365 | 315 |

Transformer: chưa chạy, cần train trên Kaggle.

## Ghi chú
- **Bias theo `id`:** không dùng `id` hay sự trùng lặp `id` giữa các file làm đặc trưng. Xem phân tích ở `docs/eda/README.md`, mục 8.
- **Giới hạn đĩa Kaggle:** mỗi run lưu 1 checkpoint fp16, khoảng 0,5 GB với model base, trong khi `/kaggle/working` chỉ khoảng 20 GB. Muốn tắt lưu checkpoint thì dùng `--set checkpoint.save=none`.
- **mDeBERTa-v3:** mặc định chạy fp32 vì fp16 dễ bị tràn số.
- **ModernBERT:** chỉ được pretrain trên tiếng Anh và code, nên chỉ dùng để so sánh.
