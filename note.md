# NST Project - Fixed Version for GitHub + Google Colab

## 1. Data assumption fixed

Project này **không yêu cầu folder labeled thủ công** nữa.

Data gốc chỉ cần có:

```text
project/
├── source_data/
│   ├── overlap_raw/           # ảnh NST chồng/chạm cần giải quyết/classify
│   └── single_chromosomes/    # ảnh NST đơn lẻ
```

Nếu repo đang dùng tên folder `single_chromosome` thay vì `single_chromosomes`, code vẫn tự nhận.

## 2. Ý tưởng pipeline

Vì data ban đầu chỉ có ảnh đơn lẻ và ảnh overlap_raw chưa gán nhãn, pipeline được chỉnh lại như sau:

```text
single_chromosomes
    ↓
Grayscale + binary + Zhang-Suen skeleton check
    ↓
Generate synthetic labeled dataset
    ├── touching
    ├── overlapping
    └── touching_overlapping
    ↓
Train Teacher: CCI-Net style
    ↓
Teacher pseudo-label overlap_raw
    ↓
Train Student: Swin Transformer + ResNet50 FPN v2
    ↓
Evaluate synthetic test + classify real overlap_raw
```

## 3. Folder structure

```text
project/
├── source_data/
│   ├── overlap_raw/
│   └── single_chromosomes/
│
├── generated_data/
│   ├── overlap_raw/
│   │   ├── *_gray.png
│   │   ├── *_binary.png
│   │   ├── *_skeleton.png
│   │   ├── *_overlay.png
│   │   └── skeleton_stats.csv
│   ├── single_chromosomes/
│   └── all_skeleton_stats.csv
│
├── dataset/
│   ├── train/
│   │   ├── touching/
│   │   ├── overlapping/
│   │   └── touching_overlapping/
│   ├── val/
│   ├── test/
│   ├── pseudo_labeled/
│   └── student_train/
│
├── result/
│   ├── checkpoints/
│   ├── teacher_ccinet_confusion_matrix.png
│   ├── student_swin_resnet50fpnv2_confusion_matrix.png
│   ├── pseudo_labels_overlap_raw.csv
│   ├── overlap_raw_predictions_student.csv
│   └── classified_overlap_raw_student/
│
├── src/
│   ├── datasets.py
│   ├── models.py
│   ├── skeleton_utils.py
│   ├── synthetic_data.py
│   ├── train_utils.py
│   └── utils.py
│
├── 1v1_create.py
├── 2v1_preprocess_zhang_suen.py
├── 3v1_build_dataset.py
├── 4v1_models.py
├── 5v1_train_teacher.py
├── 6v1_train_student_ssl.py
├── 7v1_evaluate_compare.py
├── 8v1_classify_overlap_raw.py
├── config.py
├── main.py
├── requirements.txt
└── note.md
```

## 4. Chức năng từng file

### `1v1_create.py`
Tạo/check folder structure của project.

### `2v1_preprocess_zhang_suen.py`
Xử lý ảnh:

- đọc ảnh gốc;
- chuyển grayscale;
- binarize;
- padding viền bằng `np.pad`;
- làm mỏng bằng thuật toán Zhang-Suen;
- lưu ảnh gray/binary/skeleton/overlay;
- thống kê endpoint, junction, skeleton component;
- xác định:
  - `valid_single_line`: ảnh NST đơn có đúng 1 đường skeleton, không phân nhánh, 2 endpoint;
  - `valid_two_line_candidate`: ảnh có khả năng gồm 2 đường NST đơn.

### `3v1_build_dataset.py`
Tạo dataset train/val/test tự động từ ảnh `single_chromosomes`.

Vì không có label thủ công, script tự generate ảnh synthetic gồm 2 NST đơn với 3 class:

- `touching`
- `overlapping`
- `touching_overlapping`

### `4v1_models.py`
In thông tin model và số parameter.

### `5v1_train_teacher.py`
Train Teacher model: CCI-Net style.

Cấu hình mặc định:

```text
Epoch: 200
Batch size: 64
Learning rate: 0.0001
Loss: CrossEntropyLoss
Dropout: 0.5
Early stopping: 10
```

### `6v1_train_student_ssl.py`
Semi-Supervised Learning Teacher-Student:

1. Load Teacher checkpoint.
2. Teacher pseudo-label ảnh thật trong `source_data/overlap_raw`.
3. Ảnh pseudo-label có confidence đủ cao được copy vào `dataset/pseudo_labeled`.
4. Train Student bằng synthetic dataset + pseudo-label dataset.

Student model:

```text
Swin Transformer + ResNet50 FPN v2
```

### `7v1_evaluate_compare.py`
Đánh giá Teacher và Student trên synthetic test set, xuất:

- confusion matrix PNG;
- confusion matrix CSV;
- classification report CSV;
- test predictions CSV.

### `8v1_classify_overlap_raw.py`
Dùng checkpoint để classify ảnh thật trong `source_data/overlap_raw`.

Output:

```text
result/overlap_raw_predictions_student.csv
result/classified_overlap_raw_student/
```

### `main.py`
Chạy toàn bộ pipeline.

## 5. Cách chạy local hoặc Colab

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy full pipeline:

```bash
python main.py --epochs 200 --batch-size 64 --lr 0.0001 --device cuda
```

Chạy test nhanh:

```bash
python main.py --epochs 2 --batch-size 8 --synthetic-per-class 30 --device cuda
```

Chỉ preprocess + build dataset, chưa train:

```bash
python main.py --skip-train --synthetic-per-class 100
```

Chạy từng bước:

```bash
python 1v1_create.py
python 2v1_preprocess_zhang_suen.py
python 3v1_build_dataset.py --synthetic-per-class 600
python 5v1_train_teacher.py --epochs 200 --batch-size 64 --lr 0.0001 --device cuda
python 6v1_train_student_ssl.py --epochs 200 --batch-size 64 --lr 0.0001 --device cuda
python 7v1_evaluate_compare.py --batch-size 64 --device cuda
python 8v1_classify_overlap_raw.py --model student --batch-size 64 --device cuda
```

## 6. Lưu ý quan trọng

- Project này là **image classification**, nên output chính là class/prediction của ảnh cluster, không phải mask pixel-level.
- Nếu cần tách chính xác từng NST thành mask A/B/overlap C thì phải chuyển sang segmentation như U-Net, Mask R-CNN hoặc DaCSeg.
- Ảnh train được đọc và xử lý theo grayscale. Khi đưa vào Swin/ResNet, gray channel được lặp thành 3 channel để tương thích backbone, nhưng không thêm thông tin màu.
- Nếu data thật trong `overlap_raw` chưa có ground-truth label, confusion matrix chỉ đánh giá được trên synthetic test set. `overlap_raw` sẽ có file prediction/pseudo-label riêng.
