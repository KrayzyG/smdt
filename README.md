# 🚬 Smoking Detection System (SMDT)

Hệ thống phát hiện hành vi hút thuốc sử dụng YOLOv8 với proximity-based logic.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

## 📖 Mô Tả

Hệ thống AI phát hiện hành vi hút thuốc trong ảnh/video/camera thời gian thực bằng cách:
- Phát hiện **Person** và **Cigarette** sử dụng YOLOv8
- Tính toán khoảng cách giữa người và thuốc lá
- Phân loại **SMOKING** hoặc **NON-SMOKING** dựa trên proximity logic

## ✨ Tính Năng

- ✅ Phát hiện thời gian thực qua webcam
- ✅ Xử lý ảnh và video
- ✅ Proximity-based logic (khoảng cách person-cigarette)
- ✅ Bộ lọc thuốc lá thông minh (cigarette_filter.py)
- ✅ Model đã được train với heavy augmentation
- ✅ Hỗ trợ nhiều phiên bản training (v6, v7, v8)

## 🎯 Model Performance

**Best Model: YOLOv8s (v5_full)**
- **mAP50**: 66.36%
- **mAP50-95**: 34.01%
- **Classes**: Person, Cigarette
- **Training**: 80 epochs với augmentation mạnh

## 🚀 Cài Đặt Nhanh

### 1. Clone Repository

```bash
git clone https://github.com/KrayzyG/smdt.git
cd smdt
```

### 2. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Requirements chính:**
- Python 3.8+
- ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- torch >= 2.0.0
- torchvision >= 0.15.0

### 3. Download Model Weights

Model weights đã được train sẵn:

```bash
# Download từ releases hoặc sử dụng model có sẵn
# Đặt file .pt vào thư mục gốc hoặc ketquatrain/v*/weights/
```

**Models có sẵn:**
- `yolo11n.pt` - YOLO11 nano (lightweight)
- `yolov8s.pt` - YOLOv8 small (khuyến nghị)
- Custom trained models trong `ketquatrain/`

## 📊 Cấu Trúc Project

```
smdt/
├── README.md                       # File này
├── requirements.txt                # Dependencies
├── .gitignore                      # Git ignore config
│
├── train.py                        # Training script chính
├── train_v6.py                     # Training tối ưu v6
├── train_v7_improved.py           # Training cải tiến v7
├── train_v8_moderate.py           # Training v8 (moderate aug)
│
├── predict_camera.py               # Dự đoán từ camera
├── predict_image.py                # Dự đoán từ ảnh
├── predict_video.py                # Dự đoán từ video
├── smoking_detector.py             # Class detector chính
├── cigarette_filter.py             # Bộ lọc thuốc lá
│
├── yolo11n.pt                      # YOLO11 nano weights
├── yolov8s.pt                      # YOLOv8 small weights
│
├── BAO_CAO_FINAL/                  # Báo cáo và documentation
│   ├── HUONG_DAN_SU_DUNG.md       # Hướng dẫn sử dụng
│   ├── 1_TONG_QUAN/               # Phân tích models
│   ├── 2_TRAINING_SCRIPTS/        # Training scripts
│   ├── 3_PREDICTION_SCRIPTS/      # Prediction scripts
│   └── 5_HUONG_DAN/               # Guides
│
├── input_data/                     # Input cho prediction
│   ├── images/                     # Ảnh input
│   └── videos/                     # Video input
│
├── results/                        # Kết quả prediction
│   ├── camera/                     # Kết quả camera
│   ├── image/                      # Kết quả ảnh
│   └── video/                      # Kết quả video
│
└── runs/                           # Training results
    └── train/                      # Training outputs
```

## 🎮 Sử Dụng

### Dự Đoán Từ Camera (Realtime)

```bash
python predict_camera.py --model ketquatrain/v5_full/weights/best.pt
```

**Options:**
- `--model`: Đường dẫn đến model weights
- `--conf`: Confidence threshold (default: 0.25)
- `--strict-face`: Chỉ phát hiện SMOKING khi cigarette gần face

### Dự Đoán Từ Ảnh

```bash
python predict_image.py --source input_data/images/ --model ketquatrain/v5_full/weights/best.pt
```

**Options:**
- `--source`: Đường dẫn ảnh hoặc folder
- `--model`: Model weights
- `--save`: Lưu kết quả

### Dự Đoán Từ Video

```bash
python predict_video.py --source input_data/videos/video.mp4 --model ketquatrain/v5_full/weights/best.pt
```

### Sử Dụng Cigarette Filter

```bash
python cigarette_filter.py --source input_data/images/ --model yolov8s.pt --conf 0.3
```

Chỉ hiển thị các bounding box có class **Cigarette** với confidence >= threshold.

## 🏋️ Training

### Training Cơ Bản

```bash
python train.py
```

### Training Tối Ưu (Khuyến Nghị)

```bash
# V8 - Moderate augmentation
python train_v8_moderate.py

# V7 - Improved version
python train_v7_improved.py
```

**Tham số training quan trọng:**
- `epochs`: Số epoch (50-100 khuyến nghị)
- `batch`: Batch size (8-16 tùy GPU)
- `imgsz`: Image size (640 mặc định)
- `data`: Đường dẫn đến data.yaml

## 📚 Documentation

Xem thêm tài liệu chi tiết trong `BAO_CAO_FINAL/`:

- **[HUONG_DAN_SU_DUNG.md](BAO_CAO_FINAL/HUONG_DAN_SU_DUNG.md)** - Hướng dẫn sử dụng đầy đủ
- **[1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md](BAO_CAO_FINAL/1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md)** - Phân tích models
- **[THUAT_TOAN_SU_DUNG.md](BAO_CAO_FINAL/THUAT_TOAN_SU_DUNG.md)** - Giải thích thuật toán
- **[TRAINING_OPTIMIZATION_SUMMARY.md](TRAINING_OPTIMIZATION_SUMMARY.md)** - Tối ưu training
- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** - Hướng dẫn chọn model

## 🎯 Logic Phát Hiện

### Proximity-Based Detection

**SMOKING** được phát hiện khi:
1. Phát hiện cả **Person** và **Cigarette**
2. Khoảng cách cigarette → face < 80px (ưu tiên)
3. Khoảng cách cigarette → upper body < 150px (fallback)

**NON-SMOKING**: Không thỏa điều kiện trên

### Vùng Detection

- **Face region**: 30% đầu của bounding box person
- **Upper body**: 50% phía trên của bounding box person

## 🤝 Đóng Góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Project này được phân phối dưới MIT License. Xem [LICENSE](LICENSE) để biết thêm thông tin.

## 📧 Liên Hệ

- GitHub: [@KrayzyG](https://github.com/KrayzyG)
- Repository: [https://github.com/KrayzyG/smdt.git](https://github.com/KrayzyG/smdt.git)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Dataset](https://universe.roboflow.com/richie-lab/smoking-tasfx)
- Cộng đồng YOLO Vietnam

---

⭐ Nếu project này hữu ích, hãy cho một star!
