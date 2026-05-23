# NST Project — Semi-supervised A/B/C Chromosome Segmentation

## Mục tiêu đúng của project

Project này **không phân loại ảnh thành touching/overlapping/single**.

Output đúng là nhận dạng trên từng ảnh trong `source_data/overlap_raw`:

- **NST A**
- **NST B**
- **Vùng C = vùng overlap giữa A và B**

Bài toán được code lại thành **pixel-level segmentation**.

Label map:

```text
0 = background
1 = A-only
2 = B-only
3 = C-overlap
```

Khi xuất output:

```text
mask_A = label 1 + label 3
mask_B = label 2 + label 3
mask_C = label 3
```

---

## Cấu trúc folder

```text
project/
├── source_data/
│   ├── overlap_raw/              # ảnh thật cần nhận dạng A/B/C
│   └── single_chromosomes/       # ảnh NST đơn lẻ để tạo synthetic train data
├── generated_data/
│   ├── resized/
│   │   ├── overlap_raw/
│   │   └── single_chromosomes/
│   └── skeletons/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── pseudo/
│       ├── images/
│       └── labels/
├── result/
│   ├── models/
│   ├── figures/
│   └── overlap_raw/
│       ├── labels/
│       ├── masks_A/
│       ├── masks_B/
│       ├── masks_C/
│       ├── skeleton_qc/
│       └── overlays/
├── 1v1_create.py
├── 2v1_preprocess_zhang_suen.py
├── 3v1_generate_synthetic_masks.py
├── 4v1_models.py
├── 5v1_train_teacher.py
├── 6v1_train_student_ssl.py
├── 7v1_evaluate_compare.py
├── 8v1_segment_overlap_raw.py
├── main.py
├── config.py
└── note.md
```

---

## Flow xử lý

### 1. Preprocess

File:

```text
2v1_preprocess_zhang_suen.py
```

Chức năng:

- resize toàn bộ ảnh về `224x224`
- chuyển grayscale
- tùy chọn bật Zhang-Suen skeleton QC
- padding viền trước khi skeleton để giảm chân giả ở mép ảnh
- lưu CSV kiểm tra skeleton

Chạy nhanh, không skeleton:

```bash
python 2v1_preprocess_zhang_suen.py --image-size 224
```

Chạy có Zhang-Suen skeleton:

```bash
python 2v1_preprocess_zhang_suen.py --image-size 224 --use-skeleton --save-skeleton-debug
```

Skeleton hợp lệ cho NST đơn:

```text
components = 1
endpoints = 2
branch_points = 0
```

---

### 2. Generate synthetic A/B/C masks

File:

```text
3v1_generate_synthetic_masks.py
```

Vì data gốc chỉ có:

```text
overlap_raw/          # ảnh thật không có label
single_chromosomes/   # ảnh NST đơn
```

nên code sẽ lấy 2 ảnh NST đơn, random rotate/scale/translate rồi ghép thành ảnh chồng giả lập.

Khi tự ghép, ta biết được:

```text
NST A nằm ở đâu
NST B nằm ở đâu
vùng C overlap nằm ở đâu
```

=> tạo được mask train tự động.

---

### 3. Teacher model

File:

```text
5v1_train_teacher.py
```

Teacher dùng CCI-Net style model để học synthetic A/B/C segmentation.

Loss:

```text
CrossEntropyLoss
```

Config:

```text
epoch = 200
batch size = 64
learning rate = 0.0001
dropout = 0.5
early stopping = 10
```

---

### 4. Semi-supervised Teacher-Student

File:

```text
6v1_train_student_ssl.py
```

Flow:

```text
Teacher đã train
↓
Teacher predict overlap_raw thật
↓
tạo pseudo-label A/B/C
↓
Student học synthetic labels + pseudo labels
```

Student model:

```text
Swin-like Transformer + ResNet-FPN-v2 style segmentation model
```

Đây là bước semi-supervised chính.

---

### 5. Evaluate

File:

```text
7v1_evaluate_compare.py
```

Confusion matrix ở đây là **pixel-level confusion matrix**:

```text
background / A-only / B-only / C-overlap
```

Không phải confusion matrix classification touching/overlap/single.

---

### 6. Segment overlap_raw

File:

```text
8v1_segment_overlap_raw.py
```

Output cuối:

```text
result/overlap_raw/
├── labels/
├── masks_A/
├── masks_B/
├── masks_C/
├── skeleton_qc/
├── overlays/
└── overlap_raw_ABC_predictions_student.csv
```

CSV có thêm skeleton QC cho A và B:

```text
A_skeleton_status
A_endpoints
A_branch_points
B_skeleton_status
B_endpoints
B_branch_points
```

---

## Chạy Colab nhanh

Test nhanh:

```bash
python main.py \
  --epochs 2 \
  --batch-size 4 \
  --train-count 30 \
  --val-count 10 \
  --test-count 10 \
  --image-size 224 \
  --device cuda \
  --use-skeleton
```

Train thật:

```bash
python main.py \
  --epochs 200 \
  --batch-size 64 \
  --train-count 600 \
  --val-count 120 \
  --test-count 120 \
  --lr 0.0001 \
  --image-size 224 \
  --device cuda \
  --use-skeleton
```

Nếu skeleton quá chậm, có thể bỏ `--use-skeleton` khi train, rồi chạy skeleton QC sau:

```bash
python 2v1_preprocess_zhang_suen.py --image-size 224 --use-skeleton --save-skeleton-debug
python 8v1_segment_overlap_raw.py --model student --image-size 224 --device cuda --keep-largest
```

---

## Lưu ý quan trọng

Classification chỉ trả lời ảnh thuộc loại nào. Project này cần nhận dạng NST A/B/C nên phải dùng segmentation.

Semi-supervised learning vẫn được giữ:

```text
synthetic mask supervised learning
+
teacher pseudo-labeling overlap_raw
+
student training
```

Skeleton vẫn được dùng cho:

```text
preprocess QC
lọc single_chromosomes khi cần strict_skeleton
kiểm tra output A/B cuối cùng
```

## Update: shape-aware A/B/C output repair

Bài toán cuối là nhận dạng pixel-level NST A, NST B và vùng C overlap trong `source_data/overlap_raw`, không phải classification touching/overlapping/single.

Để tránh lỗi segment bị lỗ chỗ ở vùng NST A/B, bước `8v1_segment_overlap_raw.py` đã thêm post-processing mặc định:

1. Lấy foreground từ ảnh grayscale gốc bằng Otsu để giới hạn mask trên thân NST thật.
2. Fill holes trong A/B bằng `remove_small_holes`.
3. Morphological closing để nối các vùng bị đứt nhỏ.
4. Remove small islands để bỏ nhiễu.
5. `--keep-largest` để giữ component chính của từng NST.
6. Zhang-Suen skeleton QC trên A và B.
7. Skeleton-guided candidate selection: thử nhiều mức close/fill và chọn mask có skeleton gần nhất với một đường đơn không phân nhánh.

Output sau postprocess:

```text
result/overlap_raw/
├── masks_A/          # NST A đã fill holes + shape repair
├── masks_B/          # NST B đã fill holes + shape repair
├── masks_C/          # vùng overlap C
├── raw_masks/        # mask thô trước shape repair để so sánh
├── skeleton_qc/      # skeleton A/B sau repair
├── overlays/         # overlay A/B/C
└── overlap_raw_ABC_predictions_student.csv
```

Lệnh train/predict mặc định nên dùng:

```bash
python main.py --epochs 200 --batch-size 64 --train-count 600 --val-count 120 --test-count 120 --lr 0.0001 --image-size 224 --device cuda --use-skeleton --keep-largest
```

Nếu mask vẫn còn lỗ hoặc răng cưa, tăng nhẹ:

```bash
python 8v1_segment_overlap_raw.py --model student --image-size 224 --device cuda --keep-largest --close-radius 3 --hole-area 1200
```

Nếu muốn xem model thô chưa repair để debug:

```bash
python 8v1_segment_overlap_raw.py --model student --image-size 224 --device cuda --no-shape-postprocess
```
