# MODEL INFO - v7_improved (FAILED)

## Thông tin cơ bản

**Tên model:** smoking_detection_v7_improved  
**Trạng thái:** ❌ **FAILED** - Performance tệ hơn v6  
**Ngày train:** December 23, 2025  
**Thời gian train:** ~5.5 giờ (100 epochs)

---

## Kết quả Training

### Metrics cuối cùng

| Metric | Giá trị | So với v6 | Đánh giá |
|--------|---------|-----------|----------|
| **mAP50** | 75.65% | -1.77% ❌ | TỆ HƠN |
| **Precision** | 84.15% | -2.93% ❌ | TỆ HƠN |
| **Recall** | 72.12% | -1.46% ❌ | TỆ HƠN |
| **Best Epoch** | 100 | - | Không improve |

### So sánh với các version khác

```
v5_full:      mAP50 75.96%, P 85.09%, R 70.68%
v6_optimized: mAP50 77.42%, P 87.08%, R 73.58% ⭐ BEST
v7_improved:  mAP50 75.65%, P 84.15%, R 72.12% ❌ WORST
```

**Kết luận:** v7 thậm chí tệ hơn cả v5 baseline!

---

## Cấu hình Training

### Model Architecture
```yaml
Base model: YOLOv8s
Pretrained: COCO weights (yolov8s.pt)
Parameters: 11,136,374
GFLOPs: 28.6
Input size: 640x640
```

### Training Hyperparameters
```yaml
Device: CUDA (RTX 3050 Ti 4GB)
Epochs: 100
Batch size: 10 (giảm từ 14 vì aggressive aug)
Image size: 640
Patience: 30
Close mosaic: 10 (last 10 epochs)
Workers: 8
AMP: True
Seed: 0
```

### Optimizer & Learning Rate
```yaml
Optimizer: AdamW
Initial LR (lr0): 0.015 (cao hơn v6: 0.012)
Final LR (lrf): 0.0001 (thấp hơn v6: 0.001)
LR schedule: Cosine
Warmup epochs: 8 (nhiều hơn v6: 5)
Warmup momentum: 0.8
Momentum: 0.937
Weight decay: 0.0005
```

### Loss Weights
```yaml
Box loss: 12.0 (cao hơn v6: 10.0)
Class loss: 2.0 (thấp hơn v6: 2.5) ❌ Quá thấp!
DFL loss: 2.5 (cao hơn v6: 2.0)

Strategy: Giảm cls loss để tăng Recall
Result: FAILED - Cả Recall lẫn mAP đều giảm!
```

### Data Augmentation (AGGRESSIVE)
```yaml
# Geometric augmentation
scale: 0.5 ❌ Quá nhỏ! (v6: 0.6)
translate: 0.2 ❌ Quá mạnh! (v6: 0.1)
degrees: 15 ❌ Quá nhiều rotation! (v6: 10)
shear: 3 ❌ (v6: 2)
perspective: 0.0005
fliplr: 0.5
flipud: 0.0 (cigarettes không đảo)

# Advanced augmentation
mosaic: 1.0
mixup: 0.25 ❌ Quá cao! (v6: 0.2)
copy_paste: 0.5 ❌ Quá nhiều! (v6: 0.35)

# Color augmentation
hsv_h: 0.02 ❌ (v6: 0.015)
hsv_s: 0.8 ❌ (v6: 0.7)
hsv_v: 0.5 ❌ (v6: 0.4)
```

**❌ VẤN ĐỀ:** Aggressive augmentation quá mạnh!

---

## Dataset

```yaml
Data path: dataset/smoking_train_image_v6
Classes: 2 (Cigarette, Person)

Train: 8,324 images (80.0%)
Val: 1,040 images (10.0%)
Test: 1,041 images (10.0%)
Total: 10,405 images
```

Giống v6 - Dataset không đổi.

---

## Phân tích THẤT BẠI

### Nguyên nhân chính

**1. Overfitting vào Augmented Data**
```
Aggressive aug → Quá nhiều synthetic samples
→ Model học patterns của fake data
→ Performance trên real validation data GIẢM
```

**2. Augmentation phá hủy features**
```
scale=0.5: Cigarettes ~10-15px, quá nhỏ, mất chi tiết
copy_paste=0.5: Fake instances không realistic
mixup=0.25: Blend quá nhiều, mờ object boundaries
→ Model confused, không học được real features
```

**3. Loss weight imbalance**
```
cls=2.0 (giảm từ 2.5) quá thấp
→ Model không học classification tốt
→ False positives tăng, Precision giảm
```

**4. Training không convergence**
```
100 epochs nhưng:
  - Loss plateau từ epoch 30-40
  - Metrics không cải thiện
  - Overfitting sớm
```

### Evidence

**Metrics qua epochs:**
```
Epoch 20: mAP ~72%, Recall ~69%
Epoch 40: mAP ~74%, Recall ~71%
Epoch 60: mAP ~75%, Recall ~71.5%
Epoch 80: mAP ~75.5%, Recall ~72%
Epoch 100: mAP 75.65%, Recall 72.12%

→ Improvement rất chậm sau epoch 40
→ Không đạt v6's performance (77.42%)
```

**Loss curves:**
```
Box loss: Giảm chậm, plateau sớm
Cls loss: Dao động, không stable
DFL loss: Tốt (benefit từ aggressive aug)

→ Overall: Training không optimal
```

---

## So sánh với v6_optimized

### Augmentation Comparison

| Parameter | v6 (BEST) | v7 (FAILED) | Diff | Impact |
|-----------|-----------|-------------|------|--------|
| scale | 0.6 | 0.5 | -17% | ❌ Quá nhỏ |
| copy_paste | 0.35 | 0.5 | +43% | ❌ Quá nhiều |
| mixup | 0.2 | 0.25 | +25% | ❌ Quá mạnh |
| translate | 0.1 | 0.2 | +100% | ❌ Quá mạnh |
| degrees | 10 | 15 | +50% | ❌ Quá nhiều |

**Kết luận:** Tất cả augmentation đều TĂNG QUÁ MẠNH!

### Loss Weights Comparison

| Weight | v6 (BEST) | v7 (FAILED) | Strategy |
|--------|-----------|-------------|----------|
| box | 10.0 | 12.0 | Tăng localization |
| cls | 2.5 | 2.0 ❌ | Giảm → Recall? |
| dfl | 2.0 | 2.5 | Tăng small obj |

**Vấn đề:** cls=2.0 quá thấp, phá vỡ balance!

---

## Files trong folder

```
v7_improved/
├── weights/
│   └── best.pt (22.4 MB) ❌ Không nên dùng!
├── plots/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   └── ...
├── results.csv (100 epochs data)
├── args.yaml (training config)
└── MODEL_INFO.md (this file)
```

**⚠️ CẢNH BÁO:** 
- **KHÔNG SỬ DỤNG** best.pt của v7!
- Dùng v6_optimized/weights/best.pt thay thế
- v7 chỉ để tham khảo, học bài học

---

## Bài học rút ra

### ❌ Những gì KHÔNG nên làm

1. **Tăng augmentation quá mạnh** (>40% từ baseline)
   - scale giảm >15% → Phá hủy small objects
   - copy_paste >0.4 → Fake instances không realistic
   - mixup >0.22 → Blur object boundaries

2. **Giảm cls loss quá nhiều**
   - cls <2.2 → Classification kém
   - Không balance được detection/classification

3. **Không monitor training curves**
   - Loss plateau @ epoch 40 → Nên stop early
   - Không improve → Đang waste time

### ✅ Những gì NÊN làm

1. **Moderate augmentation**
   - Tăng 10-15% từ baseline (v6)
   - Sweet spot: giữa light và aggressive

2. **Balance loss weights**
   - v6's ratio (10:2.5:2) đã optimal
   - Không cần thay đổi nhiều

3. **Early stopping**
   - Monitor validation metrics
   - Stop nếu không improve sau 20 epochs

4. **Incremental testing**
   - Test từng thay đổi một
   - Không thay đổi nhiều params cùng lúc

---

## Khuyến nghị

### ❌ KHÔNG dùng model này

v7_improved performance TỆ HƠN v6:
```
mAP50: 75.65% < 77.42% (v6)
Precision: 84.15% < 87.08% (v6)
Recall: 72.12% < 73.58% (v6)
```

### ✅ Dùng v6_optimized thay thế

```bash
# Correct model path
model_path = "ketquatrain/v6_optimized/weights/best.pt"
```

### 🔄 Hướng cải thiện

**Thử v8_moderate thay vì v7:**
```python
# Moderate aug (giữa v6 và v7)
copy_paste = 0.4      # v6: 0.35, v7: 0.5
mixup = 0.22          # v6: 0.2, v7: 0.25
scale = 0.55          # v6: 0.6, v7: 0.5
translate = 0.15      # v6: 0.1, v7: 0.2
degrees = 12          # v6: 10, v7: 15

# Loss weights
cls = 2.2             # v6: 2.5, v7: 2.0 (moderate)
```

File: `train_v8_moderate.py` đã tạo.

---

## Kết luận

**v7_improved là một THẤT BẠI hoàn toàn:**
- Aggressive augmentation không hiệu quả
- Tất cả metrics đều giảm so với v6
- Waste 5.5 giờ training time

**Bài học:**
- ⚠️ Aggressive ≠ Better
- ⚠️ Có sweet spot cho augmentation
- ⚠️ v6's moderate approach đã optimal

**Action:**
- ❌ Không deploy v7
- ✅ Tiếp tục dùng v6
- 🎯 Thử v8_moderate để cải thiện thêm

---

*Model info completed: December 23, 2025*
*Status: ARCHIVED - Do not use in production*
