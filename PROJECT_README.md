# 🚬 YOLOv8 Smoking Detection System

**Hệ thống phát hiện hành vi hút thuốc lá sử dụng YOLOv8**

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 MỤC LỤC

1. [Tổng quan](#-tổng-quan)
2. [Tính năng](#-tính-năng)
3. [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
4. [Cài đặt](#-cài-đặt)
5. [Cấu trúc dự án](#-cấu-trúc-dự-án)
6. [Sử dụng](#-sử-dụng)
7. [Model Performance](#-model-performance)
8. [Tài liệu chi tiết](#-tài-liệu-chi-tiết)
9. [Troubleshooting](#-troubleshooting)
10. [Đóng góp](#-đóng-góp)
11. [License](#-license)

---

## 🎯 TỔNG QUAN

Hệ thống **YOLOv8 Smoking Detection** là giải pháp AI tiên tiến để phát hiện tự động hành vi hút thuốc lá trong ảnh, video và camera real-time. Dự án sử dụng kiến trúc YOLOv8s với 2 classes:

- 🚬 **Cigarette**: Điếu thuốc lá
- 👤 **Person**: Người

### Ứng dụng thực tế
- ✅ Giám sát khu vực cấm hút thuốc
- ✅ An toàn phòng cháy chữa cháy
- ✅ Kiểm soát môi trường không khói thuốc
- ✅ Hệ thống cảnh báo tự động

### Điểm nổi bật
- ⚡ **Tốc độ cao**: 135 FPS (image), 54 FPS (video background)
- 🎯 **Chính xác**: mAP50 = 77.42%, Precision = 87.62%
- 🔍 **Recall tốt**: 73.93% (phát hiện 74/100 cases)
- 🖥️ **GPU-friendly**: Chạy tốt trên RTX 3050 Ti 4GB VRAM
- 📱 **Đa nền tảng**: Windows, Linux, macOS

---

## ✨ TÍNH NĂNG

### 1. Phát hiện ảnh (Image Detection)
```python
python predict_image.py --image input_data/images/test.jpg --conf 0.5
```
- ✅ Xử lý batch nhiều ảnh cùng lúc
- ✅ Tự động phân loại smoking/non-smoking
- ✅ Lưu kết quả với bounding boxes
- ✅ Tốc độ: ~135 FPS

### 2. Phát hiện video (Video Detection)
```python
python predict_video.py --video input_data/videos/test.mp4
```
- ✅ Xử lý video background (không hiển thị)
- ✅ Tự động lưu frames có smoking
- ✅ Phân loại smoking/non-smoking cho toàn video
- ✅ Tốc độ: ~54 FPS (background mode)

### 3. Camera real-time (Live Detection)
```python
python predict_camera.py --camera 0 --conf 0.5
```
- ✅ Phát hiện real-time từ webcam
- ✅ Tự động lưu ảnh khi phát hiện smoking
- ✅ Hiển thị FPS và confidence score
- ✅ Hỗ trợ nhiều camera (USB, IP camera)

### 4. Training Model
```python
python train.py  # Standard training
python train_v6.py  # Best model (recommended)
python train_v8_moderate.py  # Latest experiment
```
- ✅ Custom hyperparameters
- ✅ Tự động early stopping
- ✅ TensorBoard logging
- ✅ Checkpoint saving

---

## 💻 YÊU CẦU HỆ THỐNG

### Phần cứng khuyến nghị
- **GPU**: NVIDIA GPU với ≥4GB VRAM (RTX 3050 Ti hoặc tốt hơn)
- **RAM**: ≥16GB
- **Storage**: ≥5GB trống (cho model + dataset)

### Phần cứng tối thiểu
- **CPU**: Intel i5 hoặc tương đương
- **RAM**: 8GB
- **GPU**: Không bắt buộc (có thể chạy trên CPU)

### Phần mềm
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.8 - 3.13
- **CUDA**: 11.8+ (nếu dùng GPU)
- **cuDNN**: 8.6+ (nếu dùng GPU)

---

## 🚀 CÀI ĐẶT

### Bước 1: Clone repository
```bash
git clone <repository-url>
cd "smoking_with_yolov8 + aug"
```

### Bước 2: Tạo môi trường ảo
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Tải pre-trained weights (Optional)
Weights đã được bao gồm trong repository:
- `yolov8s.pt` - YOLOv8s pretrained (COCO)
- `ketquatrain/v6_optimized/weights/best.pt` - Best model (recommended)

### Bước 5: Kiểm tra cài đặt
```bash
python predict_image.py --image input_data/images/WIN_20251113_04_32_25_Pro.jpg
```

Nếu thành công, bạn sẽ thấy:
```
✅ Loading model...
✅ Processing image...
✅ Result saved to: results/image/...
```

---

## 📁 CẤU TRÚC DỰ ÁN

```
smoking_with_yolov8 + aug/
│
├── 📄 Core Scripts (Production-ready)
│   ├── predict_image.py          # Phát hiện ảnh
│   ├── predict_video.py          # Phát hiện video
│   ├── predict_camera.py         # Camera real-time
│   ├── smoking_detector.py       # Core detector class
│   └── cigarette_filter.py       # Logic filter smoking
│
├── 🎓 Training Scripts
│   ├── train.py                  # Standard training
│   ├── train_v6.py               # Best model (v6_optimized)
│   ├── train_v7_improved.py      # Failed experiment (reference)
│   └── train_v8_moderate.py      # Latest experiment
│
├── 📊 Dataset Tools
│   └── optimize_dataset_v6.py    # Dataset optimization
│
├── 📦 Input/Output
│   ├── input_data/               # Input files
│   │   ├── images/               # Test images
│   │   └── videos/               # Test videos
│   ├── results/                  # Detection results
│   │   ├── image/                # Image results
│   │   ├── video/                # Video results
│   │   └── camera/               # Camera snapshots
│   └── runs/                     # Training runs (auto-generated)
│
├── 🏆 Models & Weights
│   ├── ketquatrain/              # Training results
│   │   ├── v5_full/              # Baseline model
│   │   ├── v6_optimized/         # ⭐ BEST MODEL
│   │   └── v7_improved/          # Failed experiment
│   ├── yolov8s.pt                # YOLOv8s pretrained
│   └── yolo11n.pt                # YOLO11n pretrained
│
├── 📚 Documentation
│   ├── BAO_CAO_FINAL/            # ⭐ Complete project report
│   │   ├── README.md             # Main comprehensive report
│   │   ├── INDEX.md              # Navigation guide
│   │   ├── CHECKLIST.md          # Submission checklist
│   │   ├── 1_TONG_QUAN/          # Overview docs
│   │   │   ├── BAO_CAO_TONG_KET_TRAINING.md
│   │   │   ├── PHAN_TICH_CHI_TIET_CAC_MODEL.md
│   │   │   ├── MODEL_GUIDE.md
│   │   │   └── TRAINING_OPTIMIZATION_SUMMARY.md
│   │   ├── 2_TRAINING_SCRIPTS/   # Training scripts + docs
│   │   ├── 3_PREDICTION_SCRIPTS/ # Prediction scripts + docs
│   │   ├── 4_TRAINING_RESULTS/   # Training results (manual copy)
│   │   └── 5_HUONG_DAN/          # Usage guides
│   │       └── HUONG_DAN_SU_DUNG.md
│   ├── README.md                 # Quick start guide
│   ├── MODEL_GUIDE.md            # Model comparison
│   ├── PATH_STRUCTURE.md         # Directory structure
│   ├── DATA_SPLITS_IMPACT_GUIDE.md
│   └── GOOGLE_COLAB_TRAINING_GUIDE.md
│
└── 📋 Configuration
    ├── requirements.txt          # Python dependencies
    ├── .gitignore                # Git ignore rules
    └── PROJECT_README.md         # This file
```

### 📌 Folder quan trọng

**1. Prediction Scripts** (Core functionality)
```
predict_image.py, predict_video.py, predict_camera.py
smoking_detector.py, cigarette_filter.py
```

**2. Best Model** (Production use)
```
ketquatrain/v6_optimized/weights/best.pt
mAP50: 77.42%, Precision: 87.62%, Recall: 73.93%
```

**3. Documentation** (Comprehensive reports)
```
BAO_CAO_FINAL/  - Complete project documentation
├── README.md   - Main report (8 sections, ~13K tokens)
├── 1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md  - Model analysis
└── 5_HUONG_DAN/HUONG_DAN_SU_DUNG.md  - Usage guide
```

**4. Training Results**
```
runs/train/  - YOLOv8 training runs
ketquatrain/ - Organized training results
```

---

## 🎮 SỬ DỤNG

### Quick Start

#### 1. Phát hiện ảnh đơn
```bash
python predict_image.py --image input_data/images/test.jpg --conf 0.5
```

#### 2. Phát hiện batch nhiều ảnh
```bash
python predict_image.py --image input_data/images/ --conf 0.5
```

#### 3. Phát hiện video (background mode)
```bash
python predict_video.py --video input_data/videos/test.mp4
```

#### 4. Phát hiện video (với preview)
```bash
python predict_video.py --video input_data/videos/test.mp4 --show
```

#### 5. Camera real-time
```bash
python predict_camera.py --camera 0 --conf 0.5
```

### Advanced Usage

#### Custom confidence threshold
```bash
python predict_image.py --image test.jpg --conf 0.7  # Higher precision
python predict_image.py --image test.jpg --conf 0.3  # Higher recall
```

#### Custom distance threshold (smoking detection)
```bash
# smoking_detector.py line 15
self.distance_threshold = 150  # Default: 150 pixels
# Reduce for stricter detection, increase for looser
```

#### Video với frame extraction
```bash
# Default: Saves smoking frames to {videoname}_frames/
python predict_video.py --video test.mp4

# Disable frame saving
python predict_video.py --video test.mp4 --no-frames
```

#### Training với custom parameters
```python
# Modify train_v6.py or create new training script
model.train(
    data='dataset/smoking_train_image_v6/data.yaml',
    epochs=80,
    batch=14,
    imgsz=640,
    lr0=0.012,
    patience=25,
    # ... other parameters
)
```

---

## 🏆 MODEL PERFORMANCE

### Best Model: v6_optimized

| Metric | Value | Rank |
|--------|-------|------|
| **mAP50** | **77.42%** | 🥇 Best |
| **mAP50-95** | **59.05%** | 🥇 Best |
| **Precision** | **87.62%** | 🥇 Best |
| **Recall** | **73.93%** | 🥇 Best |
| **F1-Score** | **80.2%** | 🥇 Best |
| **Training Time** | 3.9 hours | - |
| **Inference Speed** | 135 FPS | - |

### Model Comparison

| Model | mAP50 | Precision | Recall | Status |
|-------|-------|-----------|--------|--------|
| v5_full | 75.96% | 87.67% | 70.64% | ✅ Baseline |
| **v6_optimized** | **77.42%** | **87.62%** | **73.93%** | ⭐ **BEST** |
| v7_improved | 75.65% | 85.88% | 70.46% | ❌ Failed |
| v8_moderate | 72.95% | 82.90% | 68.97% | ⏸️ Incomplete (40/50 epochs) |

### Performance Details

**v6_optimized advantages:**
- ✅ **Best overall mAP50**: 77.42% (highest among all models)
- ✅ **Best Recall**: 73.93% (detects 74/100 smoking cases)
- ✅ **High Precision**: 87.62% (only 12% false positives)
- ✅ **Production ready**: Stable, tested, documented
- ✅ **Fast inference**: 135 FPS on RTX 3050 Ti

**Why v6 is the best:**
1. **Optimized loss weights**: Box=10.0, Cls=2.5, DFL=2.0
2. **Better augmentation**: Scale=0.6 optimized for small cigarette detection
3. **Dataset v6**: Cleaner, more balanced data
4. **Learning rate**: lr0=0.012 provides good convergence
5. **No overfitting**: Validation metrics match training performance

### Inference Speed Benchmarks

| Mode | Hardware | FPS | Latency |
|------|----------|-----|---------|
| Image (single) | RTX 3050 Ti | ~135 FPS | 7.4ms |
| Image (batch 16) | RTX 3050 Ti | ~200 FPS | 5ms |
| Video (preview) | RTX 3050 Ti | ~31 FPS | 32ms |
| Video (background) | RTX 3050 Ti | ~54 FPS | 18.5ms |
| Camera (real-time) | RTX 3050 Ti | ~60 FPS | 16.7ms |

---

## 📖 TÀI LIỆU CHI TIẾT

### Tài liệu chính

1. **[BAO_CAO_FINAL/README.md](BAO_CAO_FINAL/README.md)**
   - Báo cáo tổng quan toàn bộ dự án
   - 8 sections: Overview, Architecture, Dataset, Training, Results, Usage, Structure, Conclusions
   - ~13,000 tokens, comprehensive documentation

2. **[BAO_CAO_FINAL/1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md](BAO_CAO_FINAL/1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md)**
   - Phân tích chi tiết 4 models (v5, v6, v7, v8)
   - Pre-training vs Post-training comparison
   - Detailed hyperparameter analysis
   - ~35,000 tokens

3. **[BAO_CAO_FINAL/5_HUONG_DAN/HUONG_DAN_SU_DUNG.md](BAO_CAO_FINAL/5_HUONG_DAN/HUONG_DAN_SU_DUNG.md)**
   - Hướng dẫn sử dụng chi tiết
   - Installation, Training, Prediction
   - Troubleshooting, Tips & Best Practices
   - Advanced usage with Python API

### Tài liệu kỹ thuật

- **[MODEL_GUIDE.md](MODEL_GUIDE.md)** - So sánh các model versions
- **[PATH_STRUCTURE.md](PATH_STRUCTURE.md)** - Cấu trúc thư mục
- **[DATA_SPLITS_IMPACT_GUIDE.md](DATA_SPLITS_IMPACT_GUIDE.md)** - Dataset splitting
- **[GOOGLE_COLAB_TRAINING_GUIDE.md](GOOGLE_COLAB_TRAINING_GUIDE.md)** - Training on Colab

### Navigation

- **[BAO_CAO_FINAL/INDEX.md](BAO_CAO_FINAL/INDEX.md)** - Quick navigation guide
- **[BAO_CAO_FINAL/CHECKLIST.md](BAO_CAO_FINAL/CHECKLIST.md)** - Submission checklist

---

## 🐛 TROUBLESHOOTING

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size in training script
batch = 10  # Instead of 14

# Or use CPU
device = 'cpu'
```

#### 2. Model not found
```
FileNotFoundError: Cannot find model weights
```

**Solution:**
```bash
# Use absolute path
python predict_image.py --model "E:/path/to/best.pt" --image test.jpg

# Or use default best model
python predict_image.py --image test.jpg  # Uses ketquatrain/v6_optimized/weights/best.pt
```

#### 3. Low FPS in video processing
```
FPS: 5-10 (too slow)
```

**Solution:**
```bash
# Use background mode (no preview)
python predict_video.py --video test.mp4  # Default: no preview, ~54 FPS

# Reduce resolution
python predict_video.py --video test.mp4 --imgsz 320  # Instead of 640
```

#### 4. Too many false positives
```
Precision low, many false detections
```

**Solution:**
```bash
# Increase confidence threshold
python predict_image.py --image test.jpg --conf 0.7  # Instead of 0.5

# Adjust distance threshold in smoking_detector.py
self.distance_threshold = 100  # Stricter (default: 150)
```

#### 5. Missing smoking detections
```
Recall low, missed smoking cases
```

**Solution:**
```bash
# Decrease confidence threshold
python predict_image.py --image test.jpg --conf 0.3  # Instead of 0.5

# Adjust distance threshold
self.distance_threshold = 200  # Looser (default: 150)
```

### Performance Optimization

**For faster inference:**
```python
# Use FP16 (half precision)
model = YOLO('best.pt')
results = model.predict('image.jpg', half=True)

# Batch processing
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = model.predict(images, batch=8)
```

**For better accuracy:**
```python
# Test-time augmentation (slower but more accurate)
results = model.predict('image.jpg', augment=True)

# Higher resolution
results = model.predict('image.jpg', imgsz=1280)  # Instead of 640
```

---

## 🤝 ĐÓNG GÓP

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings for functions
- Update documentation
- Test thoroughly before PR

---

## 📧 LIÊN HỆ

**Project Maintainer:** [Your Name]  
**Email:** [Your Email]  
**Repository:** [Repository URL]

---

## 📄 LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 ACKNOWLEDGMENTS

- **Ultralytics YOLOv8** - Object detection framework
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision library
- **RTX 3050 Ti** - Hardware for training and testing

---

## 📊 PROJECT STATS

- **Total Models Trained**: 4 (v5, v6, v7, v8)
- **Best Model**: v6_optimized (77.42% mAP50)
- **Dataset Size**: 10,405 images
- **Training Time**: ~15 hours (all models)
- **Lines of Code**: ~2,500+ (Python)
- **Documentation**: ~50,000+ tokens

---

## 🎯 FUTURE ROADMAP

- [ ] Train YOLOv8m/l for better accuracy
- [ ] Implement temporal detection (video sequences)
- [ ] Mobile deployment (TFLite/ONNX)
- [ ] Web interface (Flask/FastAPI)
- [ ] Multi-camera support
- [ ] Cloud deployment (AWS/Azure)
- [ ] Real-time alert system
- [ ] Dataset expansion (20k+ images)

---

**⭐ If you find this project useful, please give it a star!**

**Last Updated:** December 23, 2025  
**Version:** 1.0.0  
**Status:** Production Ready ✅
