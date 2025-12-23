# HƯỚNG DẪN SỬ DỤNG HỆ THỐNG SMOKING DETECTION

## 📚 MỤC LỤC
1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Training Model](#2-training-model)
3. [Prediction](#3-prediction)
4. [Tùy chỉnh Parameters](#4-tùy-chỉnh-parameters)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. CÀI ĐẶT MÔI TRƯỜNG

### 1.1. Yêu cầu hệ thống

**Minimum:**
- OS: Windows 10/11, Linux, macOS
- Python: 3.8+
- RAM: 8GB
- Storage: 20GB free

**Recommended:**
- Python: 3.9-3.11
- RAM: 16GB
- GPU: NVIDIA với CUDA support
- VRAM: 4GB+ (RTX 3050 Ti hoặc tương đương)
- Storage: 50GB free

### 1.2. Cài đặt Dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA (nếu có GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

**requirements.txt:**
```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
pillow>=10.0.0
```

### 1.3. Kiểm tra cài đặt

```bash
# Test imports
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import torch; print('PyTorch OK')"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

---

## 2. TRAINING MODEL

### 2.1. Chuẩn bị Dataset

**Cấu trúc thư mục:**
```
dataset/smoking_train_image_v6/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**data.yaml:**
```yaml
path: ../dataset/smoking_train_image_v6
train: train/images
val: val/images
test: test/images

nc: 2
names: ['Cigarette', 'Person']
```

### 2.2. Training Commands

**A. Quick Start (Recommended - v8_moderate):**
```bash
cd "smoking_with_yolov8 + aug"
python train_v8_moderate.py
```

**Kết quả:**
- Training time: ~2.5-3 giờ (50 epochs, RTX 3050 Ti)
- Output: `runs/train/smoking_detection_v8_moderate/`
- Expected: mAP50 ≥79%, Recall ≥76%

**B. Custom Training:**
```bash
python train.py \
    --data dataset/smoking_train_image_v6/data.yaml \
    --epochs 50 \
    --batch 12 \
    --imgsz 640 \
    --name my_custom_training
```

**C. Continue từ checkpoint:**
```bash
python train.py --resume runs/train/smoking_detection_v8_moderate/weights/last.pt
```

**D. Transfer learning từ model khác:**
```bash
python train.py \
    --model runs/train/smoking_detection_v6_optimized/weights/best.pt \
    --data dataset/smoking_train_image_v6/data.yaml \
    --epochs 20 \
    --name fine_tuning
```

### 2.3. Monitor Training

**Real-time monitoring:**
```bash
# TensorBoard (nếu cài đặt)
tensorboard --logdir runs/train

# Hoặc xem trực tiếp trong terminal
# Training sẽ hiển thị:
# - Loss (box, cls, dfl)
# - Metrics (Precision, Recall, mAP50, mAP50-95)
# - Learning rate
# - ETA
```

**Xem kết quả:**
```bash
# Results CSV
code runs/train/smoking_detection_v8_moderate/results.csv

# Training args
code runs/train/smoking_detection_v8_moderate/args.yaml

# Confusion matrix, F1 curve, PR curve
explorer runs/train/smoking_detection_v8_moderate/
```

### 2.4. Training Parameters chi tiết

**Basic Settings:**
```python
epochs: 50              # Số epochs (v6: 80, v8: 50)
batch: 12               # Batch size (tùy VRAM)
imgsz: 640              # Image size
patience: 25            # Early stopping
close_mosaic: 10        # Tắt mosaic cuối training
```

**Optimizer:**
```python
optimizer: 'AdamW'      # AdamW hoặc SGD
lr0: 0.013              # Initial learning rate
lrf: 0.0005             # Final LR (fraction of lr0)
cos_lr: True            # Cosine LR scheduler
warmup_epochs: 6        # Warmup epochs
momentum: 0.937
weight_decay: 0.0005
```

**Loss Weights:**
```python
box: 11.0               # Box loss weight
cls: 2.2                # Classification loss
dfl: 2.3                # DFL loss (small objects)
```

**Augmentation:**
```python
scale: 0.55             # Image scaling
copy_paste: 0.4         # Copy-paste augmentation
mixup: 0.22             # Mixup augmentation
translate: 0.15         # Translation
degrees: 12             # Rotation
shear: 2.5              # Shearing
fliplr: 0.5             # Horizontal flip
hsv_h: 0.018            # Hue augmentation
hsv_s: 0.75             # Saturation
hsv_v: 0.45             # Value
```

---

## 3. PREDICTION

### 3.1. Image Prediction

**A. Basic Usage:**
```bash
python predict_image.py --image input_data/images/test.jpg
```

**Output:**
- File: `results/image/{timestamp}_smoking_test.jpg`
- Console: Detection results với confidence scores

**B. Batch Processing:**
```bash
# Process all images in folder
python predict_image.py --image input_data/images/

# Output: results/image/{timestamp}_smoking_{filename}.jpg
```

**C. Custom Model:**
```bash
python predict_image.py \
    --model runs/train/my_model/weights/best.pt \
    --image test.jpg \
    --conf 0.25
```

**D. Ví dụ Output:**
```
Input:  test_smoking.jpg
Output: 20251223_112530_smoking_test_smoking.jpg

Detection Results:
  SMOKING DETECTED ✅
  - Cigarette (conf: 0.85) near Person head (distance: 45px)
  - Person (conf: 0.92)
```

### 3.2. Video Prediction

**A. Basic Usage (Chạy ngầm + Frame extraction):**
```bash
python predict_video.py --video input_data/videos/test.mp4
```

**Output:**
```
results/video/
├── 20251223_112939_smoking_test.mp4        # Annotated video
└── test_frames/                            # Frames có smoking
    ├── 20251223_112940_123_smoking_frame_0015.jpg
    ├── 20251223_112940_456_smoking_frame_0032.jpg
    └── ... (66 frames total)
```

**B. Với Preview:**
```bash
python predict_video.py --video test.mp4 --show
# Press 'q' để dừng
```

**C. Chỉ lưu frames, không lưu video:**
```bash
python predict_video.py --video test.mp4 --no-save
```

**D. Không lưu frames:**
```bash
python predict_video.py --video test.mp4 --no-frames
```

**E. Full Options:**
```bash
python predict_video.py \
    --video test.mp4 \
    --model runs/train/custom/weights/best.pt \
    --conf 0.25 \
    --head-dist 100 \
    --upper-dist 200 \
    --show \
    --debug
```

**F. Ví dụ Output:**
```
🎬 Processing video: test.mp4
📊 Video info: 1280x720 @ 30fps, 900 frames
📁 Frames folder: results/video/test_frames/

============================================================
🎯 KẾT QUẢ XỬ LÝ VIDEO
============================================================
  Tổng frames: 900
  Frames có smoking: 135 (15.0%)
  Thời gian xử lý: 16.7s
  FPS trung bình: 54.0
  💾 Video đã lưu: results/video/20251223_112939_smoking_test.mp4
  📁 Frames đã lưu: 135 ảnh trong results/video/test_frames/
============================================================
```

### 3.3. Camera Real-time

**A. Basic Usage:**
```bash
python predict_camera.py
```

**Controls:**
- `s`: Save current frame (nếu có smoking)
- `q`: Quit

**B. Custom Settings:**
```bash
python predict_camera.py \
    --model runs/train/custom/weights/best.pt \
    --conf 0.25 \
    --camera 0 \
    --head-dist 100
```

**C. Auto-save khi phát hiện smoking:**
```bash
# Mặc định: Tự động lưu khi phát hiện smoking
# Output: results/camera/{timestamp}_smoking_camera.jpg
```

**D. Ví dụ Output:**
```
🎥 Camera: 0
📸 Auto-save: ON (saves when smoking detected)

Frame 150:
  SMOKING DETECTED ✅
  - Cigarette (0.87) near Person head (42px)
  💾 Saved: results/camera/20251223_112530_smoking_camera.jpg

Press 's' to save, 'q' to quit
```

---

## 4. TÙY CHỈNH PARAMETERS

### 4.1. Confidence Threshold

**Default: 0.20** (optimal for best mAP50)

```bash
# Tăng conf → Ít false positives, nhiều false negatives
python predict_image.py --image test.jpg --conf 0.30

# Giảm conf → Nhiều detections, nhiều false positives
python predict_image.py --image test.jpg --conf 0.15
```

**Khuyến nghị:**
- **0.20**: Optimal (best mAP50=66.07%)
- **0.25**: Precision cao hơn, ít FP
- **0.15**: Recall cao hơn, nhiều FP

### 4.2. Distance Thresholds

**Head distance (--head-dist):**
- Default: 80px (để vẽ line từ cigarette đến head)
- Chỉ ảnh hưởng visualization, không ảnh hưởng detection

**Upper body distance (--upper-dist):**
- Default: 150px (để DETECT smoking)
- Cigarette trong 150px từ upper body → SMOKING ✅

```bash
# Strict detection (chỉ gần đầu)
python predict_image.py --image test.jpg --head-dist 60 --upper-dist 100

# Loose detection (xa hơn)
python predict_image.py --image test.jpg --head-dist 100 --upper-dist 200
```

### 4.3. Strict Face-only Mode

```bash
# Chỉ phát hiện cigarette GẦN MẶT (bỏ qua nửa trên thân)
python predict_image.py --image test.jpg --strict-face
```

**Use case:**
- Môi trường đông người
- Giảm false positives
- Chỉ quan tâm cigarette gần miệng

### 4.4. Debug Mode

```bash
python predict_image.py --image test.jpg --debug

# Output:
# - Detailed detection info
# - Distance calculations
# - Bbox coordinates
# - Confidence scores
```

---

## 5. TROUBLESHOOTING

### 5.1. Lỗi thường gặp

**A. CUDA Out of Memory:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```bash
# Giảm batch size
python train.py --batch 8  # hoặc 6, 4

# Hoặc giảm image size
python train.py --imgsz 512 --batch 12
```

**B. Model không tồn tại:**
```
❌ Model không tồn tại: runs/train/.../best.pt
```

**Giải pháp:**
```bash
# Check model path
ls runs/train/smoking_detection_v6_optimized/weights/

# Sử dụng đường dẫn đầy đủ
python predict_image.py --model "E:/path/to/best.pt" --image test.jpg
```

**C. Video không mở được:**
```
❌ Không thể mở video: test.mp4
```

**Giải pháp:**
```bash
# Check codec
ffmpeg -i test.mp4

# Convert nếu cần
ffmpeg -i test.mp4 -c:v libx264 test_converted.mp4
```

**D. Low FPS trong real-time:**
```
FPS: 5-10 (quá chậm)
```

**Giải pháp:**
```bash
# Giảm image size
python predict_camera.py --imgsz 416

# Tăng conf threshold
python predict_camera.py --conf 0.30

# Sử dụng GPU
python -c "import torch; print(torch.cuda.is_available())"  # Phải True
```

### 5.2. Performance Optimization

**A. Training faster:**
```bash
# Sử dụng mixed precision (tự động)
# Giảm epochs cho testing
python train.py --epochs 20

# Tăng workers (CPU cores)
python train.py --workers 12

# Sử dụng cache
python train.py --cache ram  # Hoặc --cache disk
```

**B. Inference faster:**
```python
# Export sang TensorRT (GPU only)
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='engine')  # TensorRT

# Sử dụng
model = YOLO('best.engine')
results = model.predict('test.jpg')
```

**C. Batch inference:**
```python
from ultralytics import YOLO
import glob

model = YOLO('best.pt')
images = glob.glob('input_data/images/*.jpg')

# Batch processing
results = model.predict(images, batch=16)  # Nhanh hơn loop
```

### 5.3. Quality Issues

**A. Nhiều False Positives:**
```bash
# Tăng confidence threshold
python predict_image.py --conf 0.30

# Sử dụng strict face-only mode
python predict_image.py --strict-face

# Retrain với better data
```

**B. Nhiều False Negatives (missing cigarettes):**
```bash
# Giảm confidence threshold
python predict_image.py --conf 0.15

# Tăng distance threshold
python predict_image.py --upper-dist 200

# Retrain với better augmentation
```

**C. Kém với small cigarettes:**
```bash
# Sử dụng model lớn hơn
python train.py --model yolov8m.pt  # hoặc yolov8l.pt

# Tăng image size
python train.py --imgsz 800

# Adjust loss weights
python train.py --dfl 2.5  # Focus small objects
```

---

## 6. TIPS & BEST PRACTICES

### 6.1. Training Tips

✅ **DO:**
- Sử dụng moderate augmentation (như v6, v8)
- Monitor validation metrics (not just training loss)
- Save checkpoints thường xuyên (`--save-period 10`)
- Validate trên test set sau training
- Document training config trong args.yaml

❌ **DON'T:**
- Aggressive augmentation (như v7 - failed)
- Train quá lâu (risk overfitting)
- Ignore validation loss tăng
- Forget to backup best model

### 6.2. Prediction Tips

✅ **DO:**
- Test với nhiều conf thresholds
- Verify outputs trước khi deploy
- Use debug mode khi gặp issue
- Batch process khi có nhiều images

❌ **DON'T:**
- Use default conf cho mọi use case
- Deploy without testing
- Ignore false positives/negatives
- Process videos without frame limit check

### 6.3. Dataset Tips

✅ **DO:**
- Balanced classes (50:50 ideal)
- High quality labels
- Diverse scenarios (lighting, angles, distances)
- Regular data cleaning

❌ **DON'T:**
- Ignore class imbalance
- Accept poor quality labels
- Collect only easy samples
- Never review/update dataset

---

## 7. ADVANCED USAGE

### 7.1. Python API

```python
from ultralytics import YOLO
import cv2
from smoking_detector import is_smoking_detected
from cigarette_filter import filter_cigarette_detections

# Load model
model = YOLO('runs/train/smoking_detection_v6_optimized/weights/best.pt')

# Predict
image = cv2.imread('test.jpg')
results = model.predict(image, conf=0.20)

# Filter cigarettes
results = filter_cigarette_detections(results)

# Smoking detection
is_smoking, persons, details = is_smoking_detected(results)

print(f"Smoking: {is_smoking}")
print(f"Persons: {persons}")
print(f"Details: {details}")
```

### 7.2. Custom Callback

```python
from ultralytics import YOLO

def on_train_epoch_end(trainer):
    # Custom logic after each epoch
    print(f"Epoch {trainer.epoch}: mAP50 = {trainer.metrics['metrics/mAP50(B)']:.4f}")

model = YOLO('yolov8s.pt')
model.add_callback('on_train_epoch_end', on_train_epoch_end)
model.train(data='data.yaml', epochs=50)
```

### 7.3. Export Models

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# TensorRT (GPU, fastest)
model.export(format='engine')

# ONNX (cross-platform)
model.export(format='onnx')

# CoreML (iOS/macOS)
model.export(format='coreml')

# TFLite (mobile)
model.export(format='tflite')
```

---

**Cập nhật:** December 23, 2025  
**Version:** 1.0  
**Contact:** Support via documentation repository
