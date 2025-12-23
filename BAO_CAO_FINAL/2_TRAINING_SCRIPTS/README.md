# SCRIPTS TRAINING - SMOKING DETECTION

## 📂 Files trong folder này

### 1. `train.py`
**Mô tả:** Main training script cơ bản

**Chức năng:**
- Training YOLOv8 với custom dataset
- Hỗ trợ tất cả YOLOv8 variants (n/s/m/l/x)
- Flexible configuration

**Sử dụng:**
```bash
python train.py \
    --data dataset/smoking_train_image_v6/data.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --batch 12
```

---

### 2. `train_v8_moderate.py` ⭐ RECOMMENDED
**Mô tả:** Training v8 với moderate augmentation strategy

**Đặc điểm:**
- ✅ Moderate augmentation (giữa v6 và v7)
- ✅ Optimized hyperparameters
- ✅ Target: mAP50 ≥79%, Recall ≥76%
- ✅ 50 epochs (2.5-3 giờ)

**Config highlights:**
```python
Augmentation:
  scale: 0.55
  copy_paste: 0.4
  mixup: 0.22
  translate: 0.15
  degrees: 12

Loss weights:
  box: 11.0
  cls: 2.2
  dfl: 2.3

Optimizer: AdamW
LR: 0.013 (cosine schedule)
```

**Sử dụng:**
```bash
python train_v8_moderate.py
# Tự động bắt đầu training
```

---

### 3. `smoking_detector.py`
**Mô tả:** Core smoking detection logic

**Chức năng:**
- Phát hiện smoking dựa trên distance giữa Cigarette và Person
- Tính toán distance từ cigarette đến head/upper body
- Support strict face-only mode

**Key functions:**
```python
is_smoking_detected(results, head_threshold=80, upper_threshold=150)
  → Returns: (is_smoking, smoking_persons, details)

get_smoking_label(is_smoking, details)
  → Returns: (label_text, color)
```

**Logic:**
```
IF Cigarette detected within upper_threshold of Person's upper body
THEN → SMOKING ✅
ELSE → NO SMOKING ❌
```

---

### 4. `cigarette_filter.py`
**Mô tả:** False positive filtering cho cigarette detections

**Chức năng:**
- Filter cigarettes based on size (min: 8px)
- Aspect ratio check (2.0-6.0 for elongated shape)
- Dynamic threshold adjustment

**Key functions:**
```python
filter_cigarette_detections(results, min_size_px=8, aspect_ratio_range=(2.0, 6.0))
  → Returns: Filtered results

get_recommended_thresholds(image_size)
  → Returns: Dynamic thresholds based on resolution
```

**Why filtering?**
- ❌ Remove tiny false positives (<8px)
- ❌ Remove non-elongated objects (aspect ratio check)
- ✅ Improve Precision (reduce FP)

---

## 🎯 TRAINING WORKFLOW

### Quy trình training chuẩn:

**1. Prepare Dataset**
```bash
dataset/smoking_train_image_v6/
├── data.yaml
├── train/ (8,324 images)
├── val/ (1,040 images)
└── test/ (1,041 images)
```

**2. Run Training**
```bash
python train_v8_moderate.py
```

**3. Monitor Progress**
```
Epoch 1/50: loss=2.5, mAP50=65%
Epoch 10/50: loss=1.8, mAP50=72%
Epoch 25/50: loss=1.2, mAP50=76%
Epoch 50/50: loss=0.9, mAP50=79% ✅
```

**4. Validate Results**
```bash
# Check results
code runs/train/smoking_detection_v8_moderate/results.csv

# Test model
python predict_image.py \
    --model runs/train/smoking_detection_v8_moderate/weights/best.pt \
    --image test.jpg
```

**5. Backup Results**
```bash
# Copy to ketquatrain
Copy-Item runs/train/smoking_detection_v8_moderate ketquatrain/v8_moderate/ -Recurse
```

---

## 📊 TRAINING VERSIONS COMPARISON

| Version | Augmentation | mAP50 | Recall | Status |
|---------|--------------|-------|--------|--------|
| v5_full | Light | 75.96% | 70.68% | ✅ Baseline |
| v6_optimized | Moderate | 77.42% | 73.58% | ⭐ Best |
| v7_improved | Aggressive | 75.65% | 72.12% | ❌ Failed |
| v8_moderate | Moderate+ | TBD | TBD | 🚀 Training |

**Key Insights:**
- ✅ Moderate augmentation (v6) là optimal
- ❌ Aggressive augmentation (v7) failed (overfitting)
- 🎯 v8 tăng nhẹ từ v6 để cải thiện Recall

---

## ⚙️ HYPERPARAMETERS GUIDE

### Augmentation Levels

**Light (v5):**
```python
scale: 0.7
copy_paste: 0.3
mixup: 0.15
```
→ Safe nhưng Recall thấp

**Moderate (v6, v8):** ⭐ OPTIMAL
```python
scale: 0.55-0.6
copy_paste: 0.35-0.4
mixup: 0.2-0.22
```
→ Balance giữa generalization và feature preservation

**Aggressive (v7):** ❌ NOT RECOMMENDED
```python
scale: 0.5
copy_paste: 0.5
mixup: 0.25
```
→ Overfitting, performance giảm

### Loss Weights

**For small objects (Cigarettes):**
```python
box: 10-11    # High → Focus localization
cls: 2.0-2.5  # Moderate → Balance
dfl: 2.0-2.5  # High → Small object focus
```

**For balanced detection:**
```python
# Increase Recall
cls: 2.0-2.2  # Lower cls → More detections

# Increase Precision  
cls: 2.5-3.0  # Higher cls → Fewer FPs
```

---

## 🚀 NEXT STEPS

**Sau khi training v8_moderate:**

**1. If SUCCESS (mAP50 ≥79%, Recall ≥76%):**
```bash
# Backup results
Copy-Item runs/train/smoking_detection_v8_moderate ketquatrain/v8_moderate/ -Recurse

# Test thoroughly
python predict_image.py --model runs/.../best.pt --image test.jpg
python predict_video.py --model runs/.../best.pt --video test.mp4

# Deploy to production ✅
```

**2. If FAILED (mAP50 <79%):**
```bash
# Try YOLOv8m (larger model)
python train.py --model yolov8m.pt --epochs 50 --batch 8

# Or improve dataset quality
# - Review labels
# - Collect more difficult samples
# - Balance classes
```

---

**Cập nhật:** December 23, 2025  
**Version:** 1.0
