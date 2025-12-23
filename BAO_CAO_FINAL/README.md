# BÁO CÁO DỰ ÁN: SMOKING DETECTION SYSTEM - YOLOv8

**Tên dự án:** Hệ thống phát hiện hành vi hút thuốc sử dụng YOLOv8  
**Thời gian thực hiện:** December 2025  
**Công nghệ:** YOLOv8, PyTorch, OpenCV, Python  
**Hardware:** NVIDIA RTX 3050 Ti 4GB, 16GB RAM

---

## 📋 MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Dataset](#3-dataset)
4. [Quy trình training](#4-quy-trình-training)
5. [Kết quả đạt được](#5-kết-quả-đạt-được)
6. [Hướng dẫn sử dụng](#6-hướng-dẫn-sử-dụng)
7. [Cấu trúc thư mục](#7-cấu-trúc-thư-mục)
8. [Kết luận](#8-kết-luận)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Mục tiêu

Xây dựng hệ thống AI tự động phát hiện hành vi hút thuốc trong ảnh, video và camera real-time với độ chính xác cao.

**Ứng dụng thực tế:**
- 🏥 Giám sát khu vực cấm hút thuốc (bệnh viện, trường học)
- 🏢 An ninh công cộng (văn phòng, trung tâm thương mại)
- 🚗 Giám sát hành vi lái xe (phát hiện lái xe hút thuốc)
- 📹 Phân tích video giám sát tự động

### 1.2. Đặc điểm kỹ thuật

**Model:** YOLOv8s (Small)
- Parameters: 11.1M
- Input size: 640x640
- Classes: 2 (Cigarette, Person)

**Logic phát hiện:**
```
IF Cigarette detected NEAR Person's head/upper body
THEN → SMOKING DETECTED ✅
```

**Thách thức:**
- ❌ Cigarettes nhỏ (1-3% ảnh, ~20-50px)
- ❌ Dễ bị che khuất (tay, môi, môi trường)
- ❌ Nhạy cảm với ánh sáng và góc chụp
- ❌ Class imbalance (Person: 70%, Cigarette: 30%)

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1. Workflow tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMOKING DETECTION SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    INPUT: Image / Video / Camera         │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  YOLOv8 Detection (Cigarette + Person)   │
        │  • Confidence threshold: 0.20            │
        │  • NMS threshold: 0.45                   │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    Cigarette Filter (Size & AR check)    │
        │  • Min size: 8px                         │
        │  • Aspect ratio: 2.0-6.0                 │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │   Smoking Detection Logic                │
        │  • Distance to head: ≤80px (visual)      │
        │  • Distance to upper body: ≤150px        │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  OUTPUT: Annotated Image/Video           │
        │  • Bounding boxes + Labels               │
        │  • SMOKING / NO SMOKING status           │
        │  • Confidence scores                     │
        └─────────────────────────────────────────┘
```

### 2.2. Core Components

**A. Training Module** (`train_*.py`)
- Data augmentation
- Model training
- Hyperparameter optimization
- Validation & evaluation

**B. Detection Module** (`smoking_detector.py`)
- YOLOv8 inference
- Smoking logic detection
- Distance calculation

**C. Filter Module** (`cigarette_filter.py`)
- False positive reduction
- Size & aspect ratio filtering
- Dynamic threshold adjustment

**D. Prediction Scripts**
- `predict_image.py`: Single image prediction
- `predict_video.py`: Video processing with frame extraction
- `predict_camera.py`: Real-time camera detection

---

## 3. DATASET

### 3.1. Thống kê

**Tổng số:** 10,405 images
- ✅ Train: 8,324 images (80%)
- ✅ Validation: 1,040 images (10%)
- ✅ Test: 1,041 images (10%)

**Classes:**
- 🚬 Cigarette: ~10,400 instances (30%)
- 👤 Person: ~24,200 instances (70%)

### 3.2. Đặc điểm dataset

**Cigarette characteristics:**
- Size: 20-50px (1-3% image area)
- Aspect ratio: 2.5-5.0 (elongated)
- Color: White/yellow (blend with background)
- Occlusion: Hand, mouth, smoke

**Person characteristics:**
- Size: 200-500px (varies with distance)
- Full body or upper body visible
- Various poses and angles

**Challenges:**
- Small object detection (<32px)
- Class imbalance (70:30 ratio)
- Lighting variations
- Motion blur in videos
- Occlusion and overlap

---

## 4. QUY TRÌNH TRAINING

### 4.1. Training Evolution (Các phiên bản)

#### **v5_full - Baseline Model** ✅
**Mục tiêu:** Thiết lập baseline performance

**Config:**
```yaml
Model: YOLOv8s (11.1M params)
Epochs: 80
Batch: 16
Optimizer: SGD
LR: 0.01 → 0.001
Augmentation: Light
  - scale: 0.7
  - copy_paste: 0.3
  - mixup: 0.15
Loss weights:
  - box: 7.5
  - cls: 0.5
  - dfl: 1.5
```

**Kết quả:**
- ✅ mAP50: 75.96%
- ✅ Precision: 85.09%
- ❌ Recall: 70.68% (low)

**Nhận xét:**
- Baseline ổn định
- Precision tốt nhưng Recall thấp
- Missing ~29% cigarettes

---

#### **v6_optimized - Best Model** ⭐ CURRENT BEST

**Mục tiêu:** Cải thiện Recall và tổng thể mAP50

**Cải tiến so với v5:**
```diff
+ Optimizer: SGD → AdamW
+ LR schedule: Step → Cosine
+ Batch: 16 → 14

Augmentation (MODERATE):
+ scale: 0.7 → 0.6
+ copy_paste: 0.3 → 0.35
+ mixup: 0.15 → 0.2
+ hsv_h: 0 → 0.015
+ hsv_s: 0 → 0.7
+ hsv_v: 0 → 0.4

Loss weights (OPTIMIZED):
+ box: 7.5 → 10.0
+ cls: 0.5 → 2.5
+ dfl: 1.5 → 2.0
```

**Kết quả:**
- ⭐ mAP50: **77.42%** (+1.46% vs v5)
- ⭐ Precision: **87.08%** (+1.99% vs v5)
- ⭐ Recall: **73.58%** (+2.90% vs v5)

**Nhận xét:**
- ✅ Tất cả metrics đều cải thiện
- ✅ Training stable, convergence tốt
- ✅ Moderate augmentation là sweet spot
- ❌ Recall vẫn <75% (thiếu sót 26.4% cigarettes)

---

#### **v7_improved - Aggressive Aug** ❌ FAILED

**Mục tiêu:** Tăng Recall lên 75-77% bằng aggressive augmentation

**Strategy:**
```diff
Epochs: 80 → 100
Batch: 14 → 10
LR: 0.012 → 0.015

Augmentation (AGGRESSIVE):
- scale: 0.6 → 0.5 ❌
- copy_paste: 0.35 → 0.5 ❌
- mixup: 0.2 → 0.25 ❌
- translate: 0.1 → 0.2 ❌
- degrees: 10 → 15 ❌

Loss weights:
- box: 10.0 → 12.0
- cls: 2.5 → 2.0 ❌
- dfl: 2.0 → 2.5
```

**Kết quả:**
- ❌ mAP50: **75.65%** (-1.77% vs v6)
- ❌ Precision: **84.15%** (-2.93% vs v6)
- ❌ Recall: **72.12%** (-1.46% vs v6)

**Phân tích thất bại:**
1. **Overfitting trên augmented data:** Model học patterns của fake data
2. **Augmentation phá hủy features:** Cigarettes quá nhỏ (~10-15px)
3. **Loss imbalance:** cls=2.0 quá thấp → Classification kém

**Bài học:**
⚠️ Aggressive augmentation ≠ Better performance  
⚠️ v6's moderate augmentation là optimal  
⚠️ Cần balance giữa augmentation và feature preservation

---

#### **v8_moderate - Current Training** 🚀 IN PROGRESS

**Mục tiêu:** mAP50 79-80%, Recall 76-78%

**Strategy:** Moderate augmentation (giữa v6 và v7)

```yaml
Epochs: 50 (reduced for faster iteration)
Batch: 12
Optimizer: AdamW
LR: 0.013 (cosine schedule)

Augmentation (MODERATE):
  scale: 0.55        # v6: 0.6, v7: 0.5
  copy_paste: 0.4    # v6: 0.35, v7: 0.5
  mixup: 0.22        # v6: 0.2, v7: 0.25
  translate: 0.15    # v6: 0.1, v7: 0.2
  degrees: 12        # v6: 10, v7: 15

Loss weights:
  box: 11.0          # v6: 10.0
  cls: 2.2           # v6: 2.5 (giảm nhẹ → tăng Recall)
  dfl: 2.3           # v6: 2.0
```

**Expected Results:**
- Target: mAP50 ≥79%, Recall ≥76-78%
- Success probability: 70-80%
- Training time: 2.5-3 hours

**Status:** 🔥 ĐANG TRAINING...

---

### 4.2. So sánh tổng quan

| Version | mAP50 | Precision | Recall | Augmentation | Status |
|---------|-------|-----------|--------|--------------|--------|
| v5_full | 75.96% | 85.09% | 70.68% | Light | ✅ Baseline |
| **v6_optimized** | **77.42%** | **87.08%** | **73.58%** | **Moderate** | ⭐ **BEST** |
| v7_improved | 75.65% | 84.15% | 72.12% | Aggressive | ❌ FAILED |
| v8_moderate | TBD | TBD | TBD | Moderate+ | 🚀 Training |

---

## 5. KẾT QUẢ ĐẠT ĐƯỢC

### 5.1. Model Performance (v6_optimized - Current Best)

**Overall Metrics:**
```
mAP50:       77.42% ⭐
mAP50-95:    48.23%
Precision:   87.08% ⭐
Recall:      73.58% ⚠️ (Low)
```

**Per-class Performance:**
```
Cigarette:
  Precision: 85.9%
  Recall:    68.2% ⚠️
  mAP50:     73.8%
  
Person:
  Precision: 88.3%
  Recall:    78.9%
  mAP50:     81.0%
```

**Inference Speed:**
- Preprocess: 0.4ms
- Inference: 5.8ms/image (RTX 3050 Ti)
- Postprocess: 1.2ms
- **Total: ~7.4ms/image (~135 FPS)**

### 5.2. Prediction Capabilities

**A. Image Prediction** (`predict_image.py`)
- ✅ Single image detection
- ✅ Batch processing support
- ✅ Auto-save với format: `{timestamp}_{smoking/non_smoking}_{filename}.jpg`
- ✅ Confidence threshold: 0.20 (optimal)

**B. Video Prediction** (`predict_video.py`)
- ✅ Video processing với frame extraction
- ✅ Tự động tạo folder lưu frames có smoking
- ✅ Chạy ngầm (no preview) mặc định
- ✅ Output video: `{timestamp}_{smoking/non_smoking}_{videoname}.mp4`
- ✅ Frame output: `{timestamp}_smoking_frame_{framenum}.jpg`
- ✅ Tốc độ: ~54 FPS (không preview)
- ✅ Smoking threshold: ≥5% frames → classified as "smoking"

**C. Camera Prediction** (`predict_camera.py`)
- ✅ Real-time detection
- ✅ Auto-save khi phát hiện smoking
- ✅ Manual save với 's' key
- ✅ Format: `{timestamp}_smoking_camera.jpg`

### 5.3. Ví dụ Output

**Image:**
```
Input:  test_image.jpg
Output: 20251223_112530_smoking_test_image.jpg
Status: SMOKING ✅ (Cigarette near head, distance: 45px)
```

**Video:**
```
Input:  test_video.mp4 (441 frames)
Output: 
  - Video: 20251223_112939_smoking_test_video.mp4
  - Frames folder: test_video_frames/
    - 20251223_112940_123_smoking_frame_0015.jpg
    - 20251223_112940_456_smoking_frame_0032.jpg
    - ... (66 frames total)
Result: 66/441 frames (15%) có smoking → Status: SMOKING
```

---

## 6. HƯỚNG DẪN SỬ DỤNG

### 6.1. Cài đặt môi trường

```bash
# Clone repository
git clone [repository_url]
cd "smoking_with_yolov8 + aug"

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Requirements:**
```
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas
matplotlib
pyyaml
```

### 6.2. Training Model

**Train từ đầu:**
```bash
# v8_moderate (recommended)
python train_v8_moderate.py

# Custom training
python train.py --epochs 50 --batch 12 --data dataset/smoking_train_image_v6/data.yaml
```

**Continue training:**
```bash
python train.py --resume runs/train/smoking_detection_v8_moderate/weights/last.pt
```

### 6.3. Prediction

**A. Ảnh đơn:**
```bash
python predict_image.py --image input_data/images/test.jpg
# Output: results/image/{timestamp}_smoking_test.jpg
```

**B. Video (với frame extraction):**
```bash
# Chạy ngầm, lưu video + frames
python predict_video.py --video input_data/videos/test.mp4

# Với preview
python predict_video.py --video test.mp4 --show

# Chỉ lưu frames, không lưu video
python predict_video.py --video test.mp4 --no-save
```

**C. Camera real-time:**
```bash
python predict_camera.py
# Press 's' to save frame
# Press 'q' to quit
```

**D. Custom model:**
```bash
python predict_image.py --model runs/train/custom_model/weights/best.pt --image test.jpg
```

### 6.4. Tùy chỉnh parameters

```bash
# Confidence threshold
python predict_image.py --image test.jpg --conf 0.25

# Distance thresholds
python predict_camera.py --head-dist 100 --upper-dist 200

# Strict face detection only
python predict_image.py --image test.jpg --strict-face
```

---

## 7. CẤU TRÚC THƯ MỤC

### 7.1. Project Structure

```
smoking_with_yolov8 + aug/
│
├── BAO_CAO_FINAL/                      # 📁 BÁO CÁO TỔNG HỢP
│   ├── README.md                       # File này
│   ├── 1_TONG_QUAN/                    # Tổng quan dự án
│   ├── 2_TRAINING_SCRIPTS/             # Scripts training
│   ├── 3_PREDICTION_SCRIPTS/           # Scripts prediction
│   ├── 4_TRAINING_RESULTS/             # Kết quả training
│   └── 5_HUONG_DAN/                    # Hướng dẫn chi tiết
│
├── dataset/                             # 📂 DATASET
│   └── smoking_train_image_v6/
│       ├── data.yaml
│       ├── train/ (8,324 images)
│       ├── val/ (1,040 images)
│       └── test/ (1,041 images)
│
├── runs/                                # 🎯 TRAINING OUTPUTS
│   └── train/
│       ├── smoking_detection_v5_full/
│       ├── smoking_detection_v6_optimized/   ⭐ BEST
│       ├── smoking_detection_v7_improved/    ❌ FAILED
│       └── smoking_detection_v8_moderate/    🚀 TRAINING
│
├── ketquatrain/                         # 📊 ARCHIVED RESULTS
│   ├── BAO_CAO_TONG_KET_TRAINING.md
│   ├── v5_full/
│   ├── v6_optimized/
│   └── v7_improved/
│
├── input_data/                          # 📥 TEST DATA
│   ├── images/
│   └── videos/
│
├── results/                             # 📤 PREDICTION OUTPUTS
│   ├── image/
│   ├── video/
│   │   └── {videoname}_frames/         # Frames có smoking
│   └── camera/
│
├── Training Scripts:                    # 🔧 TRAINING
│   ├── train.py                        # Main training script
│   ├── train_v6.py                     # v6 training
│   ├── train_v7_improved.py            # v7 training (failed)
│   └── train_v8_moderate.py            # v8 training (current)
│
├── Prediction Scripts:                  # 🔮 PREDICTION
│   ├── predict_image.py                # Image prediction
│   ├── predict_video.py                # Video + frame extraction
│   └── predict_camera.py               # Real-time camera
│
├── Core Modules:                        # ⚙️ CORE LOGIC
│   ├── smoking_detector.py             # Smoking detection logic
│   └── cigarette_filter.py             # False positive filter
│
├── Analysis Scripts:                    # 📈 ANALYSIS
│   ├── check_v6_results.py
│   ├── check_v7_results.py
│   └── analyze_issues.py
│
└── Documentation:                       # 📚 DOCS
    ├── README.md
    ├── MODEL_GUIDE.md
    ├── TRAINING_OPTIMIZATION_SUMMARY.md
    └── DATA_SPLITS_IMPACT_GUIDE.md
```

### 7.2. Output File Naming Convention

**Image:**
```
{YYYYMMDD_HHMMSS}_{smoking/non_smoking}_{original_name}.jpg
Example: 20251223_112530_smoking_test_image.jpg
```

**Video:**
```
Video: {YYYYMMDD_HHMMSS}_{smoking/non_smoking}_{original_name}.mp4
Frames: {YYYYMMDD_HHMMSS_mmm}_smoking_frame_{framenum:04d}.jpg

Example: 
  20251223_112939_smoking_video.mp4
  20251223_112940_123_smoking_frame_0015.jpg
```

**Camera:**
```
{YYYYMMDD_HHMMSS}_{smoking/non_smoking}_camera.jpg
Example: 20251223_112622_smoking_camera.jpg
```

---

## 8. KẾT LUẬN

### 8.1. Thành tựu đạt được

✅ **Model Performance:**
- Phát triển thành công model YOLOv8s với mAP50 77.42%
- Precision cao (87.08%) - Ít false positives
- Inference speed: ~135 FPS (real-time capable)

✅ **System Features:**
- Hỗ trợ đầy đủ 3 modes: Image, Video, Camera
- Tự động lưu frames có smoking từ video
- Chạy ngầm hiệu quả (54 FPS video processing)
- Output có tên file rõ ràng với status

✅ **Training Pipeline:**
- Tối ưu hóa qua 3 versions (v5 → v6 → v7)
- Xác định được optimal augmentation strategy
- Documented đầy đủ failures và lessons learned

### 8.2. Hạn chế hiện tại

❌ **Recall thấp (73.58%):**
- Missing ~26.4% cigarettes
- Yếu với small objects (<32px)
- Nhạy cảm với occlusion và lighting

❌ **Class imbalance:**
- Person: 70% samples
- Cigarette: 30% samples
- Model thiên về Person detection

❌ **Hardware limitations:**
- 4GB VRAM giới hạn batch size
- Không thể train models lớn hơn (YOLOv8m, YOLOv8l)

### 8.3. Hướng phát triển tiếp theo

**BƯỚC 1: v8_moderate (ĐANG THỰC HIỆN)** 🚀
- Mục tiêu: mAP50 ≥79%, Recall ≥76-78%
- Thời gian: 2.5-3 giờ
- Xác suất thành công: 70-80%

**BƯỚC 2: YOLOv8m (Nếu v8 thất bại)**
- Model lớn hơn: 25.9M params (2.3x YOLOv8s)
- Expected: mAP50 80-82%, Recall 78-80%
- Trade-off: +2-3% accuracy, -30% speed

**BƯỚC 3: Data Quality Improvement (Dài hạn)**
- Review và fix labels
- Collect targeted data (small cigarettes, difficult angles)
- Balance class distribution
- Retrain với cleaner dataset

**BƯỚC 4: Advanced Techniques**
- Two-stage detection (YOLO + specialized cigarette detector)
- Ensemble models
- Focal loss for class imbalance
- Knowledge distillation

### 8.4. Production Deployment

**Metrics yêu cầu:**
```
MVP (Minimum Viable Product):
  mAP50:     ≥78%
  Precision: ≥85%
  Recall:    ≥75%
  
Production Ready:
  mAP50:     ≥80%
  Precision: ≥86%
  Recall:    ≥77%
  FPS:       ≥30 (real-time)
```

**Current Status:**
- mAP50: 77.42% ✅ (Close to MVP)
- Precision: 87.08% ✅ (Excellent)
- Recall: 73.58% ⚠️ (Below MVP)
- FPS: 135 ✅ (Real-time capable)

**→ Cần cải thiện Recall lên ≥75% để đạt MVP**

---

## 📞 LIÊN HỆ & HỖ TRỢ

**Tài liệu tham khảo:**
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- PyTorch: https://pytorch.org/docs/
- OpenCV: https://docs.opencv.org/

**Files liên quan trong báo cáo:**
- `1_TONG_QUAN/`: Overview và architecture
- `2_TRAINING_SCRIPTS/`: Training code và configs
- `3_PREDICTION_SCRIPTS/`: Prediction scripts
- `4_TRAINING_RESULTS/`: Kết quả chi tiết các versions
- `5_HUONG_DAN/`: Hướng dẫn sử dụng chi tiết

---

**Cập nhật:** December 23, 2025  
**Version:** 1.0  
**Status:** 🚀 v8_moderate đang training...

---

*Dự án được thực hiện với mục đích nghiên cứu và ứng dụng AI trong phát hiện hành vi.*
