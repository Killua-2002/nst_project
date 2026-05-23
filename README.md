# NST Project - Resize All + Optional Zhang-Suen Skeleton

## 1. Data gốc

Project chỉ cần 2 folder ảnh gốc:

```text
project/
├── source_data/
│   ├── overlap_raw/           # ảnh NST chồng/chạm thật cần classify
│   └── single_chromosomes/    # ảnh NST đơn lẻ để generate synthetic train data
```

Nếu repo dùng tên `single_chromosome` thì code vẫn tự nhận, nhưng nên dùng `single_chromosomes` cho thống nhất.

## 2. Ý tưởng xử lý mới

Zhang-Suen skeleton chạy lâu vì phải làm thinning và phân tích endpoint/branch point trên CPU. Vì vậy pipeline đã tách thành 2 chế độ:

```text
FAST MODE mặc định:
source_data
    ↓
grayscale + resize all ảnh về 224x224
    ↓
generated_data/resized/
    ↓
generate synthetic dataset
    ↓
train Teacher/Student
```

```text
FULL SKELETON MODE tùy chọn:
source_data
    ↓
grayscale + resize all ảnh về 224x224
    ↓
binary + padding viền + Zhang-Suen skeleton
    ↓
endpoint / junction / component statistics
    ↓
generated_data/*/preprocess_stats.csv
```

Tức là **resize luôn chạy**, còn **skeleton có thể bật/tắt**.

## 3. Folder output sau preprocess

```text
generated_data/
├── resized/
│   ├── overlap_raw/           # ảnh thật đã grayscale + resize
│   └── single_chromosomes/    # ảnh đơn đã grayscale + resize
├── overlap_raw/
│   ├── preprocess_stats.csv
│   └── skeleton_debug/        # chỉ có khi bật --save-skeleton-debug
└── single_chromosomes/
    ├── preprocess_stats.csv
    └── skeleton_debug/        # chỉ có khi bật --save-skeleton-debug
```

`dataset/` không phải data gốc. Nó là data train/val/test tự sinh ra từ `source_data/single_chromosomes` hoặc `generated_data/resized/single_chromosomes`.

## 4. Flow chính

```text
source_data/single_chromosomes
    ↓
grayscale + resize
    ↓
generated_data/resized/single_chromosomes
    ↓
generate synthetic train/val/test
    ↓
train Teacher = CCI-Net style
    ↓
Teacher pseudo-label ảnh overlap_raw
    ↓
train Student = Swin Transformer + ResNet50 FPN v2
    ↓
evaluate confusion matrix trên synthetic test
    ↓
classify ảnh thật overlap_raw
```

Ảnh `overlap_raw` thật không có label nên không tạo confusion matrix trực tiếp được. Confusion matrix chỉ tính trên synthetic test set.

## 5. Chức năng từng file

### `1v1_create.py`
Tạo/check folder structure.

### `2v1_preprocess_zhang_suen.py`
Preprocess ảnh gốc:

- chuyển grayscale;
- resize/pad về kích thước vuông, mặc định `224x224`;
- lưu ảnh resized vào `generated_data/resized/`;
- nếu bật `--use-skeleton` thì chạy binary + padding + Zhang-Suen skeleton;
- nếu bật thêm `--save-skeleton-debug` thì lưu ảnh gray/binary/skeleton/overlay để debug.

### `3v1_build_dataset.py`
Generate synthetic dataset từ NST đơn:

- `touching`
- `overlapping`
- `touching_overlapping`

Mặc định script sẽ ưu tiên dùng ảnh đã resize trong `generated_data/resized/single_chromosomes`. Nếu chưa preprocess thì tự fallback về `source_data/single_chromosomes`.

### `4v1_models.py`
In thông tin model và parameter.

### `5v1_train_teacher.py`
Train Teacher model kiểu CCI-Net.

### `6v1_train_student_ssl.py`
Teacher pseudo-label `overlap_raw`, sau đó train Student bằng synthetic + pseudo-label.

### `7v1_evaluate_compare.py`
Xuất confusion matrix, classification report và prediction CSV cho Teacher/Student trên synthetic test set.

### `8v1_classify_overlap_raw.py`
Classify ảnh thật trong `overlap_raw`, ưu tiên dùng ảnh đã resize trong `generated_data/resized/overlap_raw`.

### `main.py`
Chạy toàn bộ pipeline.

## 6. Command nên dùng trên Colab

### Chạy nhanh, skeleton tắt mặc định

```bash
python main.py --epochs 200 --batch-size 64 --lr 0.0001 --image-size 224 --device cuda
```

### Test nhanh trước khi train lâu

```bash
python main.py --epochs 2 --batch-size 8 --synthetic-per-class 30 --image-size 224 --device cuda
```

### Chỉ resize + build dataset, chưa train

```bash
python main.py --skip-train --synthetic-per-class 100 --image-size 224
```

### Bật Zhang-Suen skeleton nhưng không lưu ảnh debug

```bash
python main.py --use-skeleton --epochs 200 --batch-size 64 --lr 0.0001 --image-size 224 --device cuda
```

### Bật skeleton và lưu ảnh debug

```bash
python main.py --use-skeleton --save-skeleton-debug --skip-train --synthetic-per-class 50 --image-size 224
```

### Chạy từng bước

```bash
python 1v1_create.py
python 2v1_preprocess_zhang_suen.py --image-size 224
python 3v1_build_dataset.py --synthetic-per-class 600 --image-size 224
python 5v1_train_teacher.py --epochs 200 --batch-size 64 --lr 0.0001 --device cuda --image-size 224
python 6v1_train_student_ssl.py --epochs 200 --batch-size 64 --lr 0.0001 --device cuda --image-size 224
python 7v1_evaluate_compare.py --batch-size 64 --device cuda --image-size 224
python 8v1_classify_overlap_raw.py --model student --batch-size 64 --device cuda --image-size 224
```

## 7. Khi nào nên bật skeleton?

Nên tắt skeleton khi:

- đang train classification;
- cần chạy nhanh trên Colab;
- chỉ cần model predict class của ảnh.

Nên bật skeleton khi:

- cần kiểm tra hình thái NST;
- cần endpoint/junction để giải thích trong report;
- muốn lọc ảnh NST đơn bằng `--strict-single`;
- chuẩn bị bước tách/cutting point sau classification.

## 8. Config mặc định

```text
Epoch: 200
Batch size: 64
Learning rate: 0.0001
Loss: CrossEntropyLoss
Dropout: 0.5
Early stopping: 10
Image size: 224
Skeleton: OFF by default
Augmentation: RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15°)
```

## 9. Lưu ý kỹ thuật

Project này là **image classification**, không phải pixel-level segmentation. Nó phân loại cụm NST thành class. Nếu cần tách chính xác NST A/B hoặc overlap mask C thì cần thêm module segmentation/cutting sau classification.
