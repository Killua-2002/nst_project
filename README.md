# NST Project — Hybrid Classif + Semi-supervised A/B/C Segmentation

## Mục tiêu đúng

Project này **không phân loại ảnh thành touching / overlapping / single**.

Output cuối cho từng ảnh trong `source_data/overlap_raw` là:

```text
NST A
NST B
vùng C = vùng A và B chồng chéo
```

Label map nội bộ:

```text
0 = background
1 = A-only
2 = B-only
3 = C-overlap
```

Khi xuất mask cuối:

```text
mask_A = A-only + C
mask_B = B-only + C
mask_C = C-overlap
```

---

## Vì sao thêm classification model?

Bản segmentation thuần có thể tạo mask A/B bị lỗ chỗ, đứt đoạn hoặc không còn hình thái NST.

Bản mới dùng hướng hybrid:

```text
Student segmentation dự đoán A/B/C
+
classification shape model kiểm tra mask có giống một NST đơn không
+
Zhang-Suen skeleton QC kiểm tra NST có dạng một đường đơn không phân nhánh
+
watershed/foreground từ raw image để ép mask bám vào thân NST thật
```

Nói đơn giản: segmentation tìm vùng, classification + skeleton giữ hình dạng NST.

---

## Semi-supervised vẫn giữ

Flow vẫn là Teacher–Student:

```text
source_data/single_chromosomes
    ↓
generate synthetic ảnh chồng + label A/B/C tự động
    ↓
train Teacher CCI-Net segmentation
    ↓
Teacher pseudo-label source_data/overlap_raw
    ↓
train Student Swin-like + ResNet-FPN-v2 segmentation
    ↓
Student predict overlap_raw
    ↓
Classification shape validator + skeleton repair chọn mask A/B tốt nhất
```

---

## Cấu trúc folder

```text
project/
├── source_data/
│   ├── overlap_raw/              # ảnh thật cần nhận dạng A/B/C
│   └── single_chromosomes/       # ảnh NST đơn, dùng tạo synthetic + train shape classifier
├── generated_data/
│   ├── resized/
│   │   ├── overlap_raw/
│   │   └── single_chromosomes/
│   └── skeletons/
├── dataset/
│   ├── train/images, train/labels
│   ├── val/images, val/labels
│   ├── test/images, test/labels
│   └── pseudo/images, pseudo/labels
├── result/
│   ├── models/
│   ├── figures/
│   └── overlap_raw/
│       ├── labels/
│       ├── masks_A/
│       ├── masks_B/
│       ├── masks_C/
│       ├── raw_masks/
│       ├── overlays/
│       ├── visualizations/
│       └── skeleton_qc/
├── 1v1_create.py
├── 2v1_preprocess_zhang_suen.py
├── 3v1_generate_synthetic_masks.py
├── 4v1_models.py
├── 4v2_train_shape_classifier.py
├── 5v1_train_teacher.py
├── 6v1_train_student_ssl.py
├── 7v1_evaluate_compare.py
├── 8v1_segment_overlap_raw.py
├── shape_classifier_model.py
├── utils_shape_guided.py
├── utils_image.py
├── utils_dataset.py
├── utils_train.py
├── main.py
└── note.md
```

---

## Chức năng file

### `1v1_create.py`

Tạo toàn bộ folder cần thiết.

### `2v1_preprocess_zhang_suen.py`

- grayscale
- resize all ảnh về `image_size`
- optional Zhang-Suen skeleton QC
- lưu ảnh resize vào `generated_data/resized`

### `3v1_generate_synthetic_masks.py`

Từ `single_chromosomes`, ghép 2 NST đơn để tạo ảnh chồng synthetic.

Sinh label:

```text
0 background
1 A-only
2 B-only
3 C-overlap
```

### `4v1_models.py`

Model segmentation:

```text
Teacher = CCI-Net style segmentation
Student = Swin-like + ResNet-FPN-v2 style segmentation
```

### `4v2_train_shape_classifier.py`

Train classification model để học hình dạng NST đơn hợp lệ.

Positive:

```text
mask foreground từ single_chromosomes
```

Negative:

```text
mask bị cắt, lỗ, vỡ, island, noise, erode/dilate sai
```

Model này không thay output A/B/C, mà dùng để score candidate mask A/B:

```text
mask này có giống NST đơn không?
```

### `5v1_train_teacher.py`

Train Teacher CCI-Net trên synthetic A/B/C labels.

### `6v1_train_student_ssl.py`

Teacher tạo pseudo-label cho `overlap_raw`, sau đó Student học:

```text
synthetic labeled data + pseudo-labeled overlap_raw
```

### `7v1_evaluate_compare.py`

Tạo pixel-level confusion matrix:

```text
background / A-only / B-only / C-overlap
```

### `8v1_segment_overlap_raw.py`

Final inference.

Output:

```text
result/overlap_raw/masks_A
result/overlap_raw/masks_B
result/overlap_raw/masks_C
result/overlap_raw/overlays
result/overlap_raw/visualizations
result/overlap_raw/skeleton_qc
```

Bước này dùng hybrid:

```text
segmentation probabilities
+
foreground raw image
+
shape classifier score
+
Zhang-Suen skeleton score
+
watershed candidate
```

---

## Chạy nhanh Colab smoke test

```bash
python main.py \
  --epochs 2 \
  --batch-size 4 \
  --train-count 30 \
  --val-count 10 \
  --test-count 10 \
  --shape-epochs 2 \
  --shape-max-count 300 \
  --image-size 224 \
  --device cuda \
  --keep-largest \
  --close-radius 2 \
  --hole-area 768 \
  --visualize-limit 12
```

---

## Train thật

```bash
python main.py \
  --epochs 200 \
  --batch-size 64 \
  --train-count 600 \
  --val-count 120 \
  --test-count 120 \
  --shape-epochs 25 \
  --shape-max-count 4000 \
  --lr 0.0001 \
  --patience 10 \
  --image-size 224 \
  --device cuda \
  --keep-largest \
  --close-radius 2 \
  --hole-area 768 \
  --visualize-limit 80
```

---

## Chạy lại final segmentation không train lại

Dùng khi đã có checkpoint và chỉ muốn chỉnh mask/visualization:

```bash
python 8v1_segment_overlap_raw.py \
  --model student \
  --image-size 224 \
  --device cuda \
  --keep-largest \
  --close-radius 3 \
  --hole-area 1200 \
  --visualize-limit -1
```

---

## Output cần xem đầu tiên

Mở folder:

```text
result/overlap_raw/visualizations/
```

Mỗi ảnh visualization gồm:

```text
raw grayscale NST
raw model overlay trước repair
final A/B/C overlay
mask C
mask A
mask B
skeleton A
skeleton B
```

Nếu mask A/B vẫn chưa đẹp:

- tăng `--close-radius 3`
- tăng `--hole-area 1200`
- chạy lại `8v1_segment_overlap_raw.py`, không cần train lại

---

## Lưu ý

Không có ground-truth mask thật cho `overlap_raw`, nên chất lượng cuối phụ thuộc vào synthetic generation + pseudo-label. Bản này sửa logic để không còn chỉ vẽ blob segmentation rỗ, mà ép mask A/B phải giống hình NST thật hơn bằng classification shape model và skeleton QC.
