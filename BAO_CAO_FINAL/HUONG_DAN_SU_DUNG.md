# HƯỚNG DẪN SỬ DỤNG SAU KHI TÁI CẤU TRÚC

## 📋 Tổng quan

Sau khi tái cấu trúc, dự án đã được tổ chức lại như sau:

```
wsf1/
├── runs/train/                              # Models đã train
│   ├── smoking_detection_v7_improved/       # ✅ Model tốt nhất
│   ├── smoking_detection_v3_improved/
│   └── smoking_detection_2classes/
├── dataset/
│   └── smoking_train_image_v6/              # Dataset chính
├── smoking_with_yolov8 + aug/
│   ├── input_data/                          # Input files
│   │   ├── images/                          # ✅ Ảnh để test
│   │   └── videos/                          # Video để test
│   └── BAO_CAO_FINAL/
│       ├── 2_TRAINING_SCRIPTS/              # Scripts training
│       │   ├── train.py
│       │   └── train_v8_moderate.py
│       └── 3_PREDICTION_SCRIPTS/            # ✅ Scripts prediction
│           ├── predict_image.py             # Dự đoán ảnh
│           ├── predict_video.py             # Dự đoán video
│           ├── predict_camera.py            # Dự đoán realtime
│           ├── smoking_detector.py          # Logic phát hiện
│           ├── cigarette_filter.py          # Filter false positives
│           └── results/                     # Kết quả output
│               ├── image/
│               ├── video/
│               └── camera/
```

## ✅ Các đường dẫn đã được sửa

### 1. **Prediction Scripts** (3_PREDICTION_SCRIPTS/)

Tất cả các scripts đã được cập nhật để tự động tìm đường dẫn:

- **Model path**: Tự động trỏ đến `wsf1/runs/train/smoking_detection_v7_improved/weights/best.pt`
- **Input data**: Tự động trỏ đến `wsf1/smoking_with_yolov8 + aug/input_data/images`
- **Output**: Lưu tại `BAO_CAO_FINAL/3_PREDICTION_SCRIPTS/results/`

### 2. **Training Scripts** (2_TRAINING_SCRIPTS/)

- **Dataset path**: Tự động trỏ đến `wsf1/dataset/smoking_train_image_v6/data.yaml`
- **Output**: Lưu tại `wsf1/runs/train/`

## 🚀 Cách sử dụng

### A. Dự đoán trên ảnh

```bash
cd "e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\smoking_with_yolov8 + aug\BAO_CAO_FINAL\3_PREDICTION_SCRIPTS"

# Dự đoán 1 ảnh cụ thể
python predict_image.py --image "path/to/image.jpg"

# Dự đoán tất cả ảnh trong input_data/images (mặc định)
python predict_image.py

# Với debug mode
python predict_image.py --debug

# Mở folder kết quả sau khi xử lý
python predict_image.py --show
```

**Tham số:**
- `--model`: Đường dẫn model (mặc định: auto-detect best.pt)
- `--image`: Ảnh cụ thể để dự đoán
- `--input-dir`: Thư mục chứa nhiều ảnh (mặc định: input_data/images)
- `--output`: Thư mục lưu kết quả (mặc định: results/image)
- `--conf`: Ngưỡng confidence (mặc định: 0.20)
- `--debug`: Hiển thị chi tiết

### B. Dự đoán trên video

```bash
cd "e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\smoking_with_yolov8 + aug\BAO_CAO_FINAL\3_PREDICTION_SCRIPTS"

python predict_video.py --video "path/to/video.mp4"
```

### C. Dự đoán realtime từ camera

```bash
cd "e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\smoking_with_yolov8 + aug\BAO_CAO_FINAL\3_PREDICTION_SCRIPTS"

python predict_camera.py --camera 0
```

### D. Training (nếu cần train lại)

```bash
cd "e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\smoking_with_yolov8 + aug\BAO_CAO_FINAL\2_TRAINING_SCRIPTS"

# Train với moderate augmentation
python train_v8_moderate.py

# Train basic
python train.py
```

## 📊 Cấu trúc dữ liệu

### Input (để test)

Đặt ảnh/video cần test vào:
- Ảnh: `wsf1/smoking_with_yolov8 + aug/input_data/images/`
- Video: `wsf1/smoking_with_yolov8 + aug/input_data/videos/`

### Output (kết quả)

Kết quả được lưu tại:
- `BAO_CAO_FINAL/3_PREDICTION_SCRIPTS/results/image/` - Ảnh kết quả
- `BAO_CAO_FINAL/3_PREDICTION_SCRIPTS/results/video/` - Video kết quả
- `BAO_CAO_FINAL/3_PREDICTION_SCRIPTS/results/camera/` - Camera screenshots

### Models

Models đã train nằm tại:
- `wsf1/runs/train/smoking_detection_v7_improved/weights/best.pt` ⭐ **Model tốt nhất**
- `wsf1/runs/train/smoking_detection_v3_improved/weights/best.pt`
- `wsf1/runs/train/smoking_detection_2classes/weights/best.pt`

### Dataset

Dataset chính:
- `wsf1/dataset/smoking_train_image_v6/`
  - `train/` - Training set
  - `val/` - Validation set
  - `test/` - Test set
  - `data.yaml` - Config file

## ⚙️ Cấu hình tối ưu

Model hiện tại (**v7_improved**) sử dụng:
- **Confidence threshold**: 0.20 (tối ưu cho mAP50)
- **Head distance**: 80px (khoảng cách tối đa cigarette → đầu)
- **Upper body distance**: 150px (khoảng cách tối đa cigarette → nửa trên cơ thể)

## 🔧 Troubleshooting

### Lỗi: Không tìm thấy model

```bash
# Kiểm tra model có tồn tại không
Test-Path "e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\runs\train\smoking_detection_v7_improved\weights\best.pt"

# Nếu không có, chỉ định model khác
python predict_image.py --model "path/to/your/model.pt"
```

### Lỗi: Không tìm thấy ảnh

```bash
# Copy ảnh test vào input_data/images
# Script sẽ tự động tìm và xử lý
```

### Lỗi: Import module

```bash
# Đảm bảo đang chạy từ đúng thư mục
cd "e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\smoking_with_yolov8 + aug\BAO_CAO_FINAL\3_PREDICTION_SCRIPTS"
```

## 📝 Lưu ý

1. **Luôn chạy từ thư mục 3_PREDICTION_SCRIPTS** để imports hoạt động đúng
2. **Model v7_improved** là model tốt nhất hiện tại
3. **Dataset v6** đã được optimize và balanced
4. Kết quả được lưu tự động với timestamp và label (smoking/non_smoking)

## 🎯 Performance

Model hiện tại (v7_improved):
- **mAP50**: ~66%
- **Cigarette detection**: ~54%
- **Person detection**: ~78%
- Tốt cho real-world scenarios

---

**Cập nhật**: 23/12/2025
**Version**: 1.0 (Sau tái cấu trúc)
