# Smoking Detection System - YOLOv8s

Hệ thống phát hiện hành vi hút thuốc (Smoking vs Non-Smoking) sử dụng YOLOv8s với augmentation mạnh.

## 📋 Dataset

- **Nguồn**: [Roboflow - smoking-tasfx v4](https://universe.roboflow.com/richie-lab/smoking-tasfx)
- **Classes (2)**: Person, Cigarette (đã loại bỏ Smoke và smoking do performance kém)
- **Format**: YOLO (txt annotations)
- **License**: CC BY 4.0
- **Training**: 11,910 images
- **Validation**: 312 images
- **Test**: 312 images

## 🏆 Model Performance (50 epochs - YOLOv8s)

**Training Configuration:**
- Model: YOLOv8s (11.2M parameters)
- Epochs: 50 (completed in 3.29 hours on GPU)
- Batch size: 12
- Optimizer: Adam (lr0=0.01)
- Heavy augmentation: mosaic=1.0, mixup=0.15, copy_paste=0.1, rotation=15°, scale=0.6

**Final Results (Epoch 50):**
- **mAP50**: 66.36%
- **mAP50-95**: 34.01%
- **Precision**: 66.31%
- **Recall**: 65.99%

**Model Location:**
```
wsf1/runs/train/smoking_detection_2classes/weights/best.pt
```

## 🎯 Logic Phân Loại - Proximity-Based Detection

**SMOKING**: Phát hiện khi `Person` (class 1) và `Cigarette` (class 0) **gần nhau**:
- Khoảng cách từ cigarette đến face/head < 80px (vẽ đường kết nối)
- Khoảng cách từ cigarette đến upper body < 150px (phát hiện SMOKING)
- Có thể tắt upper body detection với `--strict-face`

**NON-SMOKING**: Không thỏa điều kiện trên

> **💡 Lưu ý**: Logic dựa trên **khoảng cách** giữa người và thuốc lá, chính xác hơn detection riêng lẻ.

## 🚀 Cài Đặt

```bash
# Cài đặt dependencies
pip install ultralytics opencv-python torch torchvision

# Kiểm tra GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 📁 Cấu Trúc Project

```
smoking_with_yolov8 + aug/         # Main project directory
├── train.py                       # Script huấn luyện
├── predict_image.py               # Dự đoán trên ảnh
├── predict_video.py               # Dự đoán trên video
├── predict_camera.py              # Dự đoán realtime camera
├── evaluate_on_testset.py         # Đánh giá model trên test set
├── test_confidence_thresholds.py  # Test optimal confidence
├── visualize_training.py          # Visualization training progress
├── analyze_detection_errors.py    # Phân tích detection errors
│
├── input_data/                    # Input test data
│   ├── images/                    # Test images
│   └── videos/                    # Test videos
│
├── results/                       # Output predictions
│   ├── image/                     # Image results
│   ├── video/                     # Video results
│   └── camera/                    # Camera results
│
└── runs/                          # Training & evaluation results
    ├── train/
    │   └── smoking_detection_2classes/  # Model weights (21.48 MB)
    │       ├── weights/
    │       │   ├── best.pt        # Best model (mAP50=66.36%)
    │       │   └── last.pt        # Last epoch
    │       ├── results.csv        # Training metrics (50 epochs)
    │       └── args.yaml          # Training configuration
    └── test/
        ├── confidence_threshold_comparison.csv
        └── smoking_detection_evaluation/

Dataset location: ../dataset/smoking_train_image/
    ├── data.yaml                  # YOLO configuration
    ├── train/                     # 11,910 images
    ├── valid/                     # 312 images
    └── test/                      # 312 images
```
            └── args.yaml          # Training config
```

## 🏋️ Huấn Luyện Model

```bash
python train.py
```

**Cấu hình:**
- Model: YOLOv8n (nano - nhanh, phù hợp RTX 3050Ti)
- Epochs: 50 (patience=20)
- Batch size: 16
- Image size: 640x640
- Optimizer: Adam
- Device: GPU (CUDA)
- Classes: 3 (Cigarette, Person, smoking)

**Kết quả lưu tại:**
- `runs/train/smoking_detection_v3/weights/best.pt` - Model tốt nhất
- `runs/train/smoking_detection_v3/weights/last.pt` - Model cuối cùng

## 🔍 Dự Đoán

### 1. Dự đoán trên ảnh

```bash
python predict_image.py
# Nhập đường dẫn ảnh khi được hỏi
```

**Output:**
- Hiển thị ảnh với bounding boxes (màu đỏ: Cigarette, cam: smoking, xanh: Person)
- Trạng thái: SMOKING / NON-SMOKING (dựa trên proximity detection)
- Lưu tại: `runs/predict/images/`

### 2. Dự đoán trên video

```bash
python predict_video.py
# Nhập đường dẫn video khi được hỏi
```

**Output:**
- Video với bounding boxes (proximity-based detection)
- Thống kê: số frames có smoking, tỷ lệ %
- Lưu tại: `runs/predict/videos/`

**Controls:**
- `q`: Dừng xử lý

### 3. Dự đoán realtime camera

```bash
python predict_camera.py
```

**Features:**
- Hiển thị realtime với bounding boxes
- **Proximity-based detection**: Phát hiện chính xác hơn dựa trên mối quan hệ Person-Cigarette
- Tự động lưu ảnh vi phạm mỗi 1 giây (khi detect smoking)
- Thống kê: số frames, tỷ lệ smoking, violation count

**Controls:**
- `q`: Thoát
- `s`: Chụp ảnh thủ công

**Output:**
- Ảnh vi phạm lưu tại: `violations/`

## ⚙️ Tùy Chỉnh

### Điều chỉnh proximity threshold
Trong các file predict, tham số `proximity_threshold`:
```python
# Khoảng cách tối đa giữa Person và Cigarette (pixels)
is_smoking(results, proximity_threshold=150)  # Mặc định: 150
# Tăng → dễ phát hiện smoking (ít false negative)
# Giảm → strict hơn (ít false positive)
```

### Thay đổi confidence threshold
Trong các file predict, tham số `conf`:
```python
results = model.predict(source=..., conf=0.25)  # 0.1 - 1.0
# Giảm xuống 0.15 để tăng recall (phát hiện nhiều hơn)
```

### Thay đổi camera ID
Trong `predict_camera.py`, dòng 152:
```python
CAMERA_ID = 0  # 0: webcam mặc định, 1: USB camera
```

## 📊 Đánh Giá Model

Xem metrics trong thư mục training:
- `runs/train/smoking_detection_v3/results.png` - Đồ thị loss, mAP
- `runs/train/smoking_detection_v3/confusion_matrix.png` - Ma trận nhầm lẫn
- `runs/train/smoking_detection_v3/val_batch*.jpg` - Predictions trên validation set

**Cải tiến trong v3:**
- ✅ Loại bỏ Smoke class (performance kém 9.66% mAP50)
- ✅ Proximity-based detection (chính xác hơn)
- ✅ 3 classes focus: Cigarette, Person, smoking

## 🖥️ Yêu Cầu Hệ Thống

- **GPU**: RTX 3050Ti (hoặc tương đương, 4GB+ VRAM)
- **CPU**: i7-12700H (hoặc tương đương)
- **RAM**: 8GB+
- **Python**: 3.8+
- **CUDA**: 11.8+ (cho PyTorch GPU)

## 📝 Ghi Chú

- **Version:** v3 (3 classes với proximity-based detection)
- Thời gian training: ~40-50 phút (50 epochs trên RTX 3050Ti)
- FPS realtime camera: ~20-30 FPS (RTX 3050Ti)
- Model size: ~6MB (YOLOv8n)
- **Proximity threshold:** 150 pixels (có thể điều chỉnh)

## 🐛 Troubleshooting

### Lỗi GPU Out of Memory
```python
# Giảm batch size trong train.py
batch=8  # thay vì 16
```

### Camera không mở được
```python
# Thử camera ID khác
CAMERA_ID = 1  # hoặc 2, 3
```

### Import Error
```bash
pip install --upgrade ultralytics opencv-python
```

## 📄 License

MIT License
