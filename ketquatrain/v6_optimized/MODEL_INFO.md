# 📊 MODEL INFO - v6_optimized

**Status:** ✅ CURRENT BEST MODEL  
**Training Date:** 17/12/2025  
**Training Duration:** ~3.9 hours

---

## 🎯 FINAL METRICS

```yaml
Performance (Best @ Epoch 80):
  mAP50: 77.27%
  mAP50-95: 58.31%
  Precision: 87.67%
  Recall: 70.64%
  
Loss Values (Final):
  Train:
    box_loss: 1.06
    cls_loss: 3.34
    dfl_loss: 1.66
  Validation:
    box_loss: 1.33
    cls_loss: 3.66
    dfl_loss: 1.69
```

---

## ⚙️ TRAINING CONFIGURATION

### Model & Dataset
```yaml
Base Model: yolov8s.pt (COCO pretrained)
Dataset: smoking_train_image_v6
Split: 80% train / 10% valid / 10% test
Classes: 2 (Cigarette, Person)
Image Size: 640x640
```

### Training Parameters
```yaml
Epochs: 80
Batch Size: 14
Device: CUDA (RTX 3050Ti 4GB)
Workers: 8
Optimizer: AdamW
Patience: 25
Close Mosaic: 10 (last 10 epochs)
```

### Learning Rate Schedule
```yaml
lr0: 0.012
lrf: 0.001
cos_lr: False (Linear decay)
warmup_epochs: 5
warmup_momentum: 0.8
momentum: 0.937
weight_decay: 0.0005
```

### Loss Weights
```yaml
box: 10.0    # Localization loss
cls: 2.5     # Classification loss  
dfl: 2.0     # Distribution Focal Loss
```

### Data Augmentation
```yaml
# Geometric
scale: 0.6
translate: 0.1
degrees: 10
shear: 2
perspective: 0.0005
flipud: 0.0    # No vertical flip (cigarettes)
fliplr: 0.5    # Horizontal flip

# Photometric
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
bgr: 0.0

# Advanced
mosaic: 1.0
mixup: 0.2
copy_paste: 0.35
cutmix: 0.0
```

---

## 📁 FILES IN THIS FOLDER

### Weights
- `weights/best.pt` - Best model (mAP50: 77.27%)
- `weights/last.pt` - Last epoch model

### Training Results
- `results.csv` - Metrics per epoch
- `args.yaml` - Full training configuration

### Plots & Visualizations
- `plots/results.png` - Training curves (loss, mAP, P, R)
- `plots/confusion_matrix.png` - Confusion matrix
- `plots/confusion_matrix_normalized.png` - Normalized confusion matrix
- `plots/BoxPR_curve.png` - Precision-Recall curve
- `plots/BoxF1_curve.png` - F1-Confidence curve
- `plots/BoxP_curve.png` - Precision-Confidence curve
- `plots/BoxR_curve.png` - Recall-Confidence curve

---

## 📊 PERFORMANCE ANALYSIS

### Strengths
✅ **High Precision (87.67%)** - Low false positive rate  
✅ **Stable Training** - No overfitting (train/val loss similar)  
✅ **Close Mosaic Effective** - Precision improved in last 10 epochs  
✅ **Good mAP50 (77.27%)** - Decent overall performance

### Weaknesses
🔴 **Low Recall (70.64%)** - Missing ~30% of cigarettes  
⚠️ **Plateau after Epoch 50** - Limited improvement in later stages  
⚠️ **Small Object Detection** - Struggles with small cigarettes  
⚠️ **Learning Rate** - Linear decay may not be optimal

---

## 🎯 RECOMMENDED IMPROVEMENTS

### Priority 1: Increase Recall
1. Increase box loss weight: 10.0 → 12.0
2. Increase dfl loss weight: 2.0 → 2.5
3. Stronger augmentation: scale=0.5, copy_paste=0.4

### Priority 2: Better LR Schedule
1. Switch to Cosine LR: cos_lr=True
2. Higher initial LR: lr0=0.015
3. Lower final LR: lrf=0.0001
4. Longer warmup: warmup_epochs=8

### Priority 3: Extended Training
1. More epochs: 80 → 100
2. Higher patience: 25 → 30

---

## 💡 USAGE

### Load Model
```python
from ultralytics import YOLO

# Load best model
model = YOLO('ketquatrain/v6_optimized/weights/best.pt')

# Inference
results = model.predict('image.jpg', conf=0.25)

# Validation
metrics = model.val()
```

### Fine-tune
```python
# Continue training from this checkpoint
model = YOLO('ketquatrain/v6_optimized/weights/best.pt')
model.train(
    data='path/to/data.yaml',
    epochs=40,
    lr0=0.005,  # Lower LR for fine-tuning
    # ... other params
)
```

---

## 📈 COMPARISON WITH OTHER VERSIONS

| Metric | v5_full | **v6_optimized** | Improvement |
|--------|---------|------------------|-------------|
| mAP50 | 75.96% | **77.27%** | +1.31% ✅ |
| Precision | 86.47% | **87.67%** | +1.20% ✅ |
| Recall | 70.68% | **70.64%** | -0.04% ≈ |

**Key Difference:** v6 has better loss weights (box=10.0, cls=2.5, dfl=2.0)

---

## 🔄 VERSION HISTORY

**v6_optimized** (17/12/2025)
- Initial training with optimized loss weights
- Mixup augmentation enabled (0.2)
- Close mosaic implemented (last 10 epochs)
- Status: Current best model ✅

---

*Model trained on: 17/12/2025*  
*Info last updated: 22/12/2025*
