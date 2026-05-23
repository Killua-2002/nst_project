# Chromosome Overlap Classification Project

Project này được dựng theo pipeline trong yêu cầu: ảnh gốc → grayscale → binary → padding viền bằng `skimage/numpy` → Zhang-Suen thinning/skeleton → kiểm tra hình thái skeleton → train Teacher/Student → xuất confusion matrix để so sánh.

## 1. Cấu trúc thư mục

```text
project/
├── source_data/
│   ├── overlap_raw/              # ảnh NST chồng/chạm cần classify
│   ├── single_chromosomes/        # ảnh NST đơn
│   └── labeled/                   # optional: labeled/<class_name>/*.png nếu có nhãn thật
├── generated_data/
│   ├── grayscale/                 # ảnh train đã grayscale + resize/pad
│   ├── binary/                    # ảnh binary
│   ├── skeleton/                  # ảnh skeleton Zhang-Suen
│   ├── overlay/                   # ảnh debug skeleton overlay
│   └── skeleton_stats.csv          # thống kê endpoints/branchpoints/components
├── dataset/
│   ├── train/<class_name>/
│   ├── val/<class_name>/
│   ├── test/<class_name>/
│   └── unlabeled/                 # dữ liệu overlap_raw để teacher pseudo-label
├── result/
│   ├── checkpoints/
│   ├── teacher_confusion_matrix.png
│   ├── student_confusion_matrix.png
│   ├── teacher_student_comparison.csv
│   └── classified_overlap_raw.csv
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
└── requirements.txt
```

## 2. Ý nghĩa các file code

### `1v1_create.py`
Tạo toàn bộ folder theo cấu trúc project: `source_data`, `generated_data`, `dataset/train/val/test`, `result/checkpoints`.

### `2v1_preprocess_zhang_suen.py`
Xử lý ảnh:
1. Đọc ảnh gốc.
2. Chuyển sang grayscale.
3. Resize + padding để ảnh vuông.
4. Binary hóa ảnh.
5. Padding viền trước khi skeleton để chống tạo chân giả ở mép ảnh.
6. Áp dụng Zhang-Suen thinning bằng `skimage.morphology.skeletonize(method="zhang")`.
7. Đếm số skeleton components, endpoints, branchpoints.
8. Lưu ảnh grayscale, binary, skeleton, overlay và file `skeleton_stats.csv`.

### `3v1_build_dataset.py`
Tạo dataset train/val/test. Nếu có nhãn thật trong `source_data/labeled/<class_name>`, code sẽ dùng nhãn thật. Nếu chưa có nhãn thật, code dùng nhãn rule-based từ skeleton để bootstrap.

Class mặc định trong `config.py`:

```python
CLASS_NAMES = [
    "single_path",
    "two_single_paths",
    "complex_overlap",
    "invalid_or_noise",
]
```

### `4v1_models.py`
Chứa model:

- Teacher: `CCINetTeacher` kiểu CCI-Net gồm backbone CNN, SE block, multi-scale feature fusion, recognition head.
- Student: `SwinResNet50FPNv2Student` kết hợp Swin Transformer + ResNet50 FPN v2.
- Fallback: `SmallCNNFallback` để pipeline vẫn chạy nếu máy thiếu torchvision/timm hoặc không đủ RAM/GPU.

### `5v1_train_teacher.py`
Train Teacher CCI-Net với:

- Epoch: 200
- Batch size: 64
- Learning rate: 0.0001
- Loss: CrossEntropyLoss
- Dropout: 0.5
- Early stopping: 10
- Data augmentation: RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation 15 độ

### `6v1_train_student_ssl.py`
Train Student bằng semi-supervised learning kiểu teacher-student:

1. Load Teacher CCI-Net đã train.
2. Teacher dự đoán pseudo-label cho `dataset/unlabeled`.
3. Chỉ lấy pseudo-label có confidence >= `PSEUDO_LABEL_THRESHOLD`.
4. Student train bằng supervised loss + pseudo-label loss.
5. Loss chính vẫn là `CrossEntropyLoss`.

### `7v1_evaluate_compare.py`
Đánh giá Teacher và Student trên test set, xuất:

- Confusion matrix của Teacher.
- Confusion matrix của Student.
- CSV so sánh accuracy, macro-F1.

### `8v1_classify_overlap_raw.py`
Dùng model Student để classify toàn bộ ảnh trong `source_data/overlap_raw`, xuất:

- `result/classified_overlap_raw.csv`
- ảnh debug grayscale/binary/skeleton/overlay trong `result/classified_overlap_raw/`

## 3. Cách chạy

Cài thư viện:

```bash
pip install -r requirements.txt
```

Cho dữ liệu vào:

```text
source_data/overlap_raw/
source_data/single_chromosomes/
```

Chạy toàn bộ pipeline:

```bash
python main.py
```

Chạy thử nhanh để check pipeline trước:

```bash
python main.py --epochs 2 --batch-size 4 --device cpu
```

Chỉ preprocess + build dataset, chưa train:

```bash
python main.py --skip-train
```

## 4. Lưu ý quan trọng

Classification model chỉ phân loại loại ảnh/cụm NST. Nếu muốn tách pixel-level thành NST A/NST B/overlap C thì cần segmentation mask và U-Net/Mask R-CNN/DaCSeg-style model. Project này vẫn có skeleton + rule-based analysis để hỗ trợ nhận diện ảnh có đúng 2 đường NST đơn không phân nhánh, nhưng output chính là nhãn classification và confusion matrix.

Nếu cần dùng đúng 3 lớp như báo cáo CCI-Net, hãy đổi `CLASS_NAMES` trong `config.py`, sau đó đặt ảnh có nhãn thật vào:

```text
source_data/labeled/touching/
source_data/labeled/overlapping/
source_data/labeled/touching_overlapping/
```

rồi sửa `CLASS_NAMES = ["touching", "overlapping", "touching_overlapping"]`.
