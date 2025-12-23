# PREDICTION SCRIPTS - SMOKING DETECTION

## 📂 Files trong folder này

### 1. `predict_image.py` 📸
**Chức năng:** Prediction cho single hoặc batch images

**Features:**
- ✅ Single image prediction
- ✅ Batch processing (folder)
- ✅ Auto-save với smoking status
- ✅ Custom confidence threshold
- ✅ Debug mode

**Output format:**
```
{YYYYMMDD_HHMMSS}_{smoking/non_smoking}_{original_name}.jpg
Example: 20251223_112530_smoking_test.jpg
```

**Usage:**
```bash
# Single image
python predict_image.py --image test.jpg

# Batch processing
python predict_image.py --image input_data/images/

# Custom conf
python predict_image.py --image test.jpg --conf 0.25

# Debug mode
python predict_image.py --image test.jpg --debug
```

**Example output:**
```
Input: test_smoking.jpg
Output: 20251223_112530_smoking_test_smoking.jpg

Detection Results:
  SMOKING DETECTED ✅
  - Cigarette (conf: 0.85) near Person head (45px)
  - Person (conf: 0.92)
  - Status: smoking
```

---

### 2. `predict_video.py` 🎬
**Chức năng:** Video processing với frame extraction

**Features:**
- ✅ Video processing với annotated output
- ✅ Tự động tạo folder lưu frames có smoking
- ✅ Chạy ngầm mặc định (no preview)
- ✅ Smoking status dựa trên % frames
- ✅ Tốc độ cao: ~54 FPS

**Output:**
```
Video:
  {YYYYMMDD_HHMMSS}_{smoking/non_smoking}_{videoname}.mp4

Frames folder: {videoname}_frames/
  {YYYYMMDD_HHMMSS_mmm}_smoking_frame_{framenum:04d}.jpg
```

**Usage:**
```bash
# Default: chạy ngầm, lưu video + frames
python predict_video.py --video test.mp4

# Với preview
python predict_video.py --video test.mp4 --show

# Chỉ lưu frames, không lưu video
python predict_video.py --video test.mp4 --no-save

# Không lưu frames
python predict_video.py --video test.mp4 --no-frames
```

**Example output:**
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
  💾 Video đã lưu: 20251223_112939_smoking_test.mp4
  📁 Frames đã lưu: 135 ảnh trong test_frames/
============================================================
```

**Smoking threshold:** ≥5% frames → "smoking" status

---

### 3. `predict_camera.py` 📹
**Chức năng:** Real-time camera detection

**Features:**
- ✅ Real-time detection từ webcam
- ✅ Auto-save khi phát hiện smoking
- ✅ Manual save với 's' key
- ✅ FPS display
- ✅ Live annotation

**Output format:**
```
{YYYYMMDD_HHMMSS}_{smoking/non_smoking}_camera.jpg
Example: 20251223_112530_smoking_camera.jpg
```

**Usage:**
```bash
# Default webcam (camera 0)
python predict_camera.py

# Custom camera
python predict_camera.py --camera 1

# Custom confidence
python predict_camera.py --conf 0.25

# Custom model
python predict_camera.py --model runs/train/custom/weights/best.pt
```

**Controls:**
- `s`: Save current frame (manual)
- `q`: Quit

**Example output:**
```
🎥 Camera: 0
📸 Auto-save: ON (saves when smoking detected)

Frame 150:
  SMOKING DETECTED ✅
  - Cigarette (0.87) near Person head (42px)
  💾 Auto-saved: 20251223_112530_smoking_camera.jpg

Press 's' to save, 'q' to quit
```

---

### 4. `smoking_detector.py` 🔍
**Core Module:** Smoking detection logic

**Key Functions:**

**A. `is_smoking_detected()`**
```python
is_smoking_detected(
    results,
    head_threshold=80,      # Distance to head for visualization
    upper_threshold=150,    # Distance to upper body for detection
    conf_threshold=0.20,
    strict_face_only=False,
    debug=False
)

Returns: (is_smoking, smoking_persons, details)
```

**Logic:**
1. Extract Cigarette và Person detections
2. Tính distance từ cigarette đến person
3. Nếu distance ≤ upper_threshold → SMOKING ✅

**B. `get_smoking_label()`**
```python
get_smoking_label(is_smoking, details)

Returns: (label_text, color)
  - "🚬 SMOKING" (red) nếu is_smoking=True
  - "✅ NO SMOKING" (green) nếu False
```

**Distance calculation:**
```python
# From cigarette center to person's upper body
cig_center = (cig_x1 + cig_x2) / 2, (cig_y1 + cig_y2) / 2
person_head = (person_x1 + person_x2) / 2, person_y1
distance = sqrt((cig_x - head_x)^2 + (cig_y - head_y)^2)

if distance <= upper_threshold:
    SMOKING ✅
```

---

### 5. `cigarette_filter.py` 🔬
**Core Module:** False positive filtering

**Key Functions:**

**A. `filter_cigarette_detections()`**
```python
filter_cigarette_detections(
    results,
    min_size_px=8,
    aspect_ratio_range=(2.0, 6.0),
    debug=False
)

Returns: Filtered YOLOv8 results
```

**Filtering criteria:**
```python
✅ KEEP if:
  - Size ≥ min_size_px (default: 8px)
  - Aspect ratio in range (2.0-6.0)
  - Elongated shape (width/height or height/width)

❌ REMOVE if:
  - Too small (<8px) → Noise
  - Wrong aspect ratio → Not cigarette shape
```

**B. `get_recommended_thresholds()`**
```python
get_recommended_thresholds(image_size)

Returns: Dict with dynamic thresholds
```

**Dynamic adjustment:**
```python
# For 1920x1080
min_size_px = 8
aspect_ratio_range = (2.0, 6.0)

# For 640x480  
min_size_px = 5
aspect_ratio_range = (2.5, 7.0)
```

---

## 🎯 PREDICTION WORKFLOW

### Standard workflow:

**1. Prepare Input**
```bash
# Images
input_data/images/test.jpg

# Videos
input_data/videos/test.mp4

# Camera
# Webcam plugged in
```

**2. Run Prediction**
```bash
# Image
python predict_image.py --image input_data/images/test.jpg

# Video (background processing)
python predict_video.py --video input_data/videos/test.mp4

# Camera (real-time)
python predict_camera.py
```

**3. Check Results**
```bash
# Images
results/image/{timestamp}_smoking_test.jpg

# Videos
results/video/{timestamp}_smoking_test.mp4
results/video/test_frames/ (frames có smoking)

# Camera
results/camera/{timestamp}_smoking_camera.jpg
```

---

## ⚙️ PARAMETERS GUIDE

### Confidence Threshold

**Default: 0.20** (optimal for mAP50)

```bash
# High Precision (ít FP)
--conf 0.30

# Balanced (default)
--conf 0.20

# High Recall (nhiều detections)
--conf 0.15
```

**Recommendation:**
- Production: 0.20-0.25
- Testing: 0.15-0.20
- Demo: 0.25-0.30

### Distance Thresholds

**head_threshold (visualization):**
- Default: 80px
- Chỉ ảnh hưởng line drawing
- Không ảnh hưởng detection

**upper_threshold (detection):**
- Default: 150px
- Ảnh hưởng SMOKING detection
- Cigarette trong 150px → SMOKING ✅

```bash
# Strict detection
--head-dist 60 --upper-dist 100

# Default
--head-dist 80 --upper-dist 150

# Loose detection
--head-dist 100 --upper-dist 200
```

### Strict Face-only Mode

```bash
--strict-face
```

**Effect:**
- Chỉ phát hiện cigarette GẦN MẶT
- Bỏ qua detections xa hơn
- Giảm false positives

**Use case:**
- Môi trường đông người
- Cần Precision cao
- Chỉ quan tâm smoking near mouth

---

## 📊 PERFORMANCE

### Speed Benchmarks (RTX 3050 Ti)

**Image:**
```
Single image: ~7.4ms
  - Preprocess: 0.4ms
  - Inference: 5.8ms
  - Postprocess: 1.2ms
  
Throughput: ~135 FPS
```

**Video:**
```
Without preview: ~54 FPS
With preview: ~31 FPS

720p video (441 frames): 8-14s
1080p video (900 frames): 16-25s
```

**Camera:**
```
Real-time: 25-35 FPS (live display)
Inference only: ~135 FPS
```

### Optimization Tips

**Faster inference:**
```bash
# Reduce image size
--imgsz 416  # Default: 640

# Increase conf threshold
--conf 0.30

# Use TensorRT (GPU)
model.export(format='engine')
```

**Batch processing:**
```python
# Instead of loop
results = model.predict(images, batch=16)
```

---

## 🔧 TROUBLESHOOTING

### Common Issues

**1. Low FPS:**
```bash
# Check GPU usage
nvidia-smi

# Reduce image size
python predict_camera.py --imgsz 416

# Close preview
python predict_video.py  # No --show
```

**2. Nhiều False Positives:**
```bash
# Tăng confidence
--conf 0.30

# Strict mode
--strict-face

# Check cigarette_filter settings
```

**3. Missing Detections:**
```bash
# Giảm confidence
--conf 0.15

# Tăng distance threshold
--upper-dist 200

# Debug mode
--debug
```

---

## 💡 BEST PRACTICES

### DO ✅

- Test với nhiều conf thresholds
- Verify outputs trước deploy
- Use batch processing cho nhiều images
- Monitor FPS trong real-time
- Save important frames

### DON'T ❌

- Use single conf cho mọi scenario
- Ignore false positives
- Process large videos without checking
- Deploy without testing
- Forget to backup results

---

**Cập nhật:** December 23, 2025  
**Version:** 1.0
