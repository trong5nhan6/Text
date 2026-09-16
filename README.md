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
  processed/            text đã làm sạch + cột fold (sinh tự động)            [ignored]
docs/
  shared_task_README.md  FORMAT.md  LICENSE_NOTE.md   thể lệ gốc của ban tổ chức
  eda/README.md + figures/ + stats/                   báo cáo EDA
  starting_kit/                                       sample submission + baseline của BTC
notebooks/              hastika_kaggle.ipynb (chạy trên Kaggle) + build_notebook.py
results/                                                                      [ignored]
  metrics.csv                       bảng tổng hợp mọi run
  {task}/{run}/                     oof.npy val.npy [test.npy] metrics.json config.yaml per_class.csv confusion.png
  submissions/{task}_{split}_{tag}/ predictions.csv + submission.zip (nộp Codabench)
checkpoints/{task}/{run}/fold{k}/   trọng số fold tốt nhất (fp16)              [ignored]
logs/{task}_{run}.log                                                         [ignored]
```

## Cài đặt
```bash
pip install -r requirements.txt
python -m src.data.preprocessing          # tạo data/processed (train.py cũng tự gọi nếu thiếu)
python -m src.eda.eda                     # (tuỳ chọn) sinh lại báo cáo EDA ở docs/eda/
```

## Train
```bash
python train.py --config configs/tfidf.yaml --task a
python train.py --config configs/muril.yaml --task b
python train.py --config configs/roberta.yaml --task b --set training.loss=focal --run_name roberta_focal
python train.py --config configs/muril.yaml --task a --folds 0 1     # train một phần, chạy lại để tiếp tục
```
- **Tên run mặc định** là `<config>_<loss>_s<seed>`, ví dụ `muril_wce_s42`; riêng TF-IDF là `tfidf_lr`.
- **Tiếp tục khi bị ngắt:** mỗi fold xong được lưu cache, nên chạy lại đúng lệnh sẽ bỏ qua các fold đã có.
- **Đổi siêu tham số:** nếu giữ nguyên tên run, script sẽ **từ chối chạy**. Hãy dùng `--run_name` khác hoặc thêm `--overwrite`.
- **Ghi đè cấu hình từ dòng lệnh:** mọi khoá trong YAML đều đổi được bằng `--set khoa.con=gia_tri`.
- **Loss mặc định:** Task A dùng `ce`, Task B dùng `wce` (class weight = căn bậc hai của nghịch đảo tần suất).

## Đánh giá (OOF)
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
Mở `notebooks/hastika_kaggle.ipynb` trên Kaggle, bật **GPU T4** và **Internet**, rồi *Run All*.
- Notebook đã chứa toàn bộ code nên không cần upload repo.
- Sau khi sửa code hoặc config ở máy, chạy `python notebooks/build_notebook.py` để cập nhật notebook.

## Khi có test (20/9)
1. Đặt `*test*.csv` vào `data/raw/`, rồi chạy `python -m src.data.preprocessing`. Fold train giữ nguyên.
2. Run TF-IDF: chạy lại `train.py` (vài phút).
3. Run transformer đã có checkpoint: dùng `inference.py --checkpoints ... --input ...`.

## Kết quả hiện tại (OOF 5-fold)
| Task | Run | Macro-F1 | Acc |
|---|---|---|---|
| A | tfidf_lr (C=16) | 0,8165 | 0,8166 |
| B | tfidf_lr (C=1, balanced) | 0,6088 | 0,6859 |
| B | tfidf_svm (C=0,5, balanced) | 0,6004 | 0,6935 |

Transformer: chưa chạy, cần train trên Kaggle.

## Ghi chú
- **Bias theo `id`:** không dùng `id` hay sự trùng lặp `id` giữa các file làm đặc trưng. Xem phân tích ở `docs/eda/README.md`, mục 8.
- **Giới hạn đĩa Kaggle:** checkpoint lưu dạng fp16, khoảng 0,5 GB/fold với model base, trong khi `/kaggle/working` chỉ khoảng 20 GB. Muốn tắt lưu checkpoint thì dùng `--set checkpoint.save=none`.
- **mDeBERTa-v3:** mặc định chạy fp32 vì fp16 dễ bị tràn số.
- **ModernBERT:** chỉ được pretrain trên tiếng Anh và code, nên chỉ dùng để so sánh.
