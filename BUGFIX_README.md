# Bugfix note

This version fixes the broken Colab run:

1. `utils_image.py`
   - Removed the invalid `remove_small_objects(mask, max_size=...)` call for older Colab/scikit-image builds.
   - Removed the recursive fallback that caused `RecursionError: maximum recursion depth exceeded`.
   - Added version-safe wrappers for `remove_small_objects` and `remove_small_holes`.

2. `utils_train.py`
   - Removed `torchvision.transforms.functional` dependency because some Colab/local environments have a mismatched `torchvision::nms` binary.
   - Implemented paired augmentation with PIL directly:
     - RandomResizedCrop style crop
     - RandomHorizontalFlip
     - RandomVerticalFlip
     - RandomRotation ±15°

3. Project logic
   - The task is A/B/C pixel-level segmentation, not image classification.
   - Output labels:
     - 0 = background
     - 1 = NST A only
     - 2 = NST B only
     - 3 = overlap region C
   - Final masks:
     - mask_A = A-only + C
     - mask_B = B-only + C
     - mask_C = C
   - Semi-supervised learning:
     - Teacher CCI-Net trains on synthetic masks generated from `single_chromosomes`.
     - Teacher pseudo-labels `overlap_raw`.
     - Student Swin-like + ResNet-FPN-v2 trains on synthetic + pseudo-label data.
   - Final output runs shape-aware repair + Zhang-Suen skeleton QC on A and B.
