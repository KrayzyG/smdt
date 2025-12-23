# CẤU TRÚC BÁO CÁO - SMOKING DETECTION PROJECT

## 📁 DANH MỤC FILES

### 📄 File chính
- **[README.md](../README.md)** - Báo cáo tổng quan toàn bộ dự án

### 📂 1. TỔNG QUAN
**Folder:** `1_TONG_QUAN/`

**Nội dung:**
- `BAO_CAO_TONG_KET_TRAINING.md` - Tổng kết quá trình training (v5/v6/v7/v8)
- `README.md` - Giới thiệu dự án
- `MODEL_GUIDE.md` - Hướng dẫn về models và cấu hình
- `TRAINING_OPTIMIZATION_SUMMARY.md` - Tối ưu hóa training

**Mục đích:** Cung cấp cái nhìn tổng quan về dự án, lịch sử phát triển, và kết quả đạt được.

---

### 📂 2. TRAINING SCRIPTS
**Folder:** `2_TRAINING_SCRIPTS/`

**Nội dung:**
- `train.py` - Main training script
- `train_v8_moderate.py` - Training v8 với moderate augmentation (current)
- `smoking_detector.py` - Core smoking detection logic
- `cigarette_filter.py` - False positive filtering

**Mục đích:** Chứa các scripts để training models và core detection logic.

**Sử dụng:**
```bash
cd "smoking_with_yolov8 + aug"
python train_v8_moderate.py  # Train v8
```

---

### 📂 3. PREDICTION SCRIPTS
**Folder:** `3_PREDICTION_SCRIPTS/`

**Nội dung:**
- `predict_image.py` - Prediction cho single/batch images
- `predict_video.py` - Video processing + frame extraction
- `predict_camera.py` - Real-time camera detection
- `smoking_detector.py` - Smoking detection logic (copy)
- `cigarette_filter.py` - Cigarette filtering (copy)

**Mục đích:** Scripts để sử dụng trained models cho prediction.

**Sử dụng:**
```bash
# Image
python predict_image.py --image test.jpg

# Video (chạy ngầm, lưu frames)
python predict_video.py --video test.mp4

# Camera
python predict_camera.py
```

---

### 📂 4. TRAINING RESULTS
**Folder:** `4_TRAINING_RESULTS/`

**Nội dung:** (Cần copy manually từ `runs/train/` và `ketquatrain/`)
- `v5_full/` - Baseline results
- `v6_optimized/` - Best model results ⭐
- `v7_improved/` - Failed aggressive aug
- `v8_moderate/` - Current training results

**Mỗi folder chứa:**
- `weights/best.pt` - Model weights
- `results.csv` - Training metrics
- `args.yaml` - Training config
- `*.png` - Plots (confusion matrix, curves, etc.)
- `MODEL_INFO.md` - Detailed analysis

**Mục đích:** Lưu trữ và so sánh kết quả các phiên bản training.

---

### 📂 5. HƯỚNG DẪN
**Folder:** `5_HUONG_DAN/`

**Nội dung:**
- `HUONG_DAN_SU_DUNG.md` - Hướng dẫn chi tiết sử dụng hệ thống

**Mục đích:** Documentation chi tiết về cách sử dụng, troubleshooting, best practices.

**Bao gồm:**
- Cài đặt môi trường
- Training guide
- Prediction guide
- Tùy chỉnh parameters
- Troubleshooting
- Tips & Best practices

---

## 🎯 CÁCH SỬ DỤNG BÁO CÁO

### Cho Người đọc nhanh:
1. Đọc [README.md](../README.md) - Tổng quan 10 phút
2. Xem `1_TONG_QUAN/BAO_CAO_TONG_KET_TRAINING.md` - Chi tiết training
3. Check `4_TRAINING_RESULTS/` - Xem kết quả cụ thể

### Cho Người muốn sử dụng:
1. Đọc `5_HUONG_DAN/HUONG_DAN_SU_DUNG.md` - Setup & usage
2. Copy scripts từ `3_PREDICTION_SCRIPTS/`
3. Sử dụng best model từ `4_TRAINING_RESULTS/v6_optimized/`

### Cho Người muốn phát triển:
1. Đọc toàn bộ `1_TONG_QUAN/`
2. Study scripts trong `2_TRAINING_SCRIPTS/`
3. Analyze results trong `4_TRAINING_RESULTS/`
4. Tham khảo `5_HUONG_DAN/` cho advanced usage

---

## 📊 THỐNG KÊ DỰ ÁN

**Dataset:**
- Total: 10,405 images
- Classes: 2 (Cigarette, Person)
- Split: 80/10/10

**Models Trained:**
- v5_full: Baseline (mAP50: 75.96%)
- v6_optimized: Best ⭐ (mAP50: 77.42%)
- v7_improved: Failed (mAP50: 75.65%)
- v8_moderate: Training... (Target: 79%+)

**System Capabilities:**
- Image prediction
- Video processing with frame extraction
- Real-time camera detection
- ~135 FPS inference (RTX 3050 Ti)

---

## 🔗 QUICK LINKS

**Main Documentation:**
- [📄 Tổng quan dự án](../README.md)
- [📚 Hướng dẫn sử dụng](5_HUONG_DAN/HUONG_DAN_SU_DUNG.md)
- [📊 Báo cáo training](1_TONG_QUAN/BAO_CAO_TONG_KET_TRAINING.md)

**Scripts:**
- [🔧 Training Scripts](2_TRAINING_SCRIPTS/)
- [🔮 Prediction Scripts](3_PREDICTION_SCRIPTS/)

**Results:**
- [📈 Training Results](4_TRAINING_RESULTS/)

---

**Cập nhật:** December 23, 2025  
**Version:** 1.0
