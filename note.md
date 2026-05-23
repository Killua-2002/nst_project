# NST Overlap Segmentation Project — A/B/C Output

## Mục tiêu đã sửa

Project này **không còn là image classification** kiểu `touching / overlapping / single` nữa.
Output đúng là segmentation cho ảnh trong `source_data/overlap_raw`:

- **NST A**: vùng NST thứ nhất.
- **NST B**: vùng NST thứ hai.
- **Vùng C**: vùng hai NST chồng lên nhau.

Để vẫn dùng `CrossEntropyLoss`, mask huấn luyện được mã hóa thành 4 class pixel-level:

| Label | Ý nghĩa |
|---|---|
| 0 | background |
| 1 | A-only |
| 2 | B-only |
| 3 | C-overlap |

Khi xuất kết quả:

- `mask_A = label == 1 hoặc label == 3`
- `mask_B = label == 2 hoặc label == 3`
- `mask_C = label == 3`

Như vậy vùng C vẫn thuộc cả A và B, nhưng loss vẫn dùng được `CrossEntropyLoss` vì label training là 4-class exclusive.

## Cấu trúc folder

```text
project/
├── source_data/
│   ├── overlap_raw/              # ảnh thật cần nhận dạng A/B/C
│   └── single_chromosomes/       # ảnh NST đơn lẻ, dùng tạo synthetic mask
├── generated_data/
│   └── resized/
│       ├── overlap_raw/
│       └── single_chromosomes/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   ├── masks_A/
│   │   ├── masks_B/
│   │   └── masks_C/
│   ├── val/
│   ├── test/
│   └── pseudo/
├── result/
│   ├── models/
│   ├── plots/
│   └── overlap_raw/
│       ├── labels/
│       ├── masks_A/
│       ├── masks_B/
│       ├── masks_C/
│       └── overlays/
├── 1v1_create.py
├── 2v1_preprocess_resize_skeleton.py
├── 3v1_generate_synthetic_masks.py
├── 4v1_models.py
├── 5v1_train_teacher.py
├── 6v1_train_student_ssl.py
├── 7v1_evaluate_compare.py
├── 8v1_segment_overlap_raw.py
├── main.py
├── config.py
├── models.py
├── train_utils.py
├── utils_image.py
└── note.md
```

## Hướng làm

### 1. Dữ liệu gốc giữ nguyên

Chỉ cần 2 folder data gốc:

```text
source_data/overlap_raw
source_data/single_chromosomes
```

`dataset/` không phải data gốc thứ hai. Nó được sinh tự động từ `single_chromosomes`.

### 2. Preprocess

File: `2v1_preprocess_resize_skeleton.py`

Thực hiện:

1. Đọc ảnh.
2. Chuyển grayscale.
3. Resize tất cả về `224x224`.
4. Zhang-Suen skeleton là optional, mặc định tắt cho nhanh.

Chạy nhanh:

```bash
python 2v1_preprocess_resize_skeleton.py --image-size 224
```

Bật skeleton khi cần report/debug:

```bash
python 2v1_preprocess_resize_skeleton.py --image-size 224 --use-skeleton --save-skeleton-debug
```

### 3. Generate synthetic segmentation dataset

File: `3v1_generate_synthetic_masks.py`

Từ ảnh NST đơn lẻ, code tạo ảnh chồng synthetic:

```text
single chromosome 1 + single chromosome 2
        ↓
synthetic overlap image
        ↓
label 0/1/2/3
        ↓
mask_A, mask_B, mask_C
```

Quy ước để A/B có nghĩa ổn định:

- A được xoay thiên về hướng ngang.
- B được xoay thiên về hướng dọc.
- C là phần giao nhau của A và B.

Chạy:

```bash
python 3v1_generate_synthetic_masks.py --train 600 --val 120 --test 120 --image-size 224
```

### 4. Teacher model

File: `5v1_train_teacher.py`

Teacher dùng kiến trúc **CCI-Net-inspired segmentation model**:

- CNN backbone.
- SE block.
- Multi-scale feature fusion.
- Decoder xuất 4 class pixel-level.

Train bằng synthetic labels thật.

```bash
python 5v1_train_teacher.py --epochs 200 --batch-size 64 --lr 0.0001 --device cuda
```

### 5. Semi-Supervised Teacher–Student

File: `6v1_train_student_ssl.py`

Teacher predict ảnh thật trong `overlap_raw` để tạo pseudo-label:

```text
overlap_raw image
    ↓
Teacher CCI-Net
    ↓
pseudo label 0/1/2/3
    ↓
Student training
```

Student dùng mô hình **Swin Transformer + ResNet50 FPN v2 inspired**:

- ResNet-style encoder.
- FPN multi-scale fusion.
- Swin-style window self-attention block.
- Decoder xuất 4 class pixel-level.

Student học từ:

```text
synthetic labeled data + pseudo-labeled overlap_raw
```

Chạy:

```bash
python 6v1_train_student_ssl.py --epochs 200 --batch-size 64 --lr 0.0001 --device cuda
```

### 6. Evaluate

File: `7v1_evaluate_compare.py`

So sánh Teacher và Student trên synthetic test set có label thật.
Kết quả gồm:

- pixel confusion matrix 4 class.
- mean IoU.
- mean Dice.
- pixel accuracy.

```bash
python 7v1_evaluate_compare.py --device cuda
```

### 7. Segment overlap_raw thành A/B/C

File: `8v1_segment_overlap_raw.py`

Output chính nằm ở:

```text
result/overlap_raw/
├── labels/       # label 0/1/2/3
├── masks_A/      # NST A
├── masks_B/      # NST B
├── masks_C/      # vùng C overlap
├── overlays/     # ảnh overlay để nhìn nhanh
└── overlap_raw_ABC_predictions_student.csv
```

Chạy:

```bash
python 8v1_segment_overlap_raw.py --model student --device cuda
```

## Chạy full pipeline

Test nhanh trên Colab:

```bash
python main.py --epochs 2 --batch-size 4 --train-count 30 --val-count 10 --test-count 10 --image-size 224 --device cuda
```

Train thật:

```bash
python main.py --epochs 200 --batch-size 64 --train-count 600 --val-count 120 --test-count 120 --lr 0.0001 --image-size 224 --device cuda
```

Nếu Colab bị OOM, giảm batch:

```bash
python main.py --epochs 200 --batch-size 16 --train-count 600 --val-count 120 --test-count 120 --lr 0.0001 --image-size 224 --device cuda
```

## Lưu ý quan trọng

Vì data thật `overlap_raw` không có mask thật, pseudo-label của nó phụ thuộc vào Teacher. Semi-supervised learning giúp Student học thêm từ ảnh thật, nhưng chất lượng vẫn phụ thuộc synthetic generation và confidence threshold.

A/B là nhãn theo quy ước trong synthetic data, không phải tên sinh học cố định. Trong project này:

- A thường là NST theo trục chính ngang.
- B thường là NST theo trục chính dọc.
- C là vùng giao/overlap.
