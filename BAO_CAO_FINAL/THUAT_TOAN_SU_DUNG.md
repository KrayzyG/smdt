# CÁC THUẬT TOÁN ĐANG SỬ DỤNG TRONG DỰ ÁN SMOKING DETECTION

## 📋 Tổng quan

Dự án sử dụng **hybrid approach** kết hợp:
- **Deep Learning** (YOLOv8) cho object detection
- **Classical Computer Vision** (distance, geometry) cho reasoning
- **Rule-based Logic** (thresholds, filters) cho refinement

---

## 🎯 **1. OBJECT DETECTION - YOLOv8**

### **YOLO (You Only Look Once) v8**

**Type**: Single-stage object detector  
**Architecture**: CSPDarknet backbone + PAN-FPN neck + Detection head  
**Method**: Anchor-free detection  
**Loss**: DFL (Distribution Focal Loss) + CIoU  

```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.predict(source=img, conf=0.20)
```

### Đặc điểm YOLOv8:

- **Single-pass detection**: Không dùng region proposals như R-CNN
- **Grid-based**: Chia ảnh thành grid, mỗi cell dự đoán bounding boxes
- **Anchor-free**: Dự đoán trực tiếp center, width, height (không cần anchor boxes)
- **Real-time**: Tốc độ cao, phù hợp cho ứng dụng thực tế

### Architecture Components:

```
Input Image (640x640)
    ↓
[Backbone - CSPDarknet]
    - Feature extraction
    - Multi-scale features
    ↓
[Neck - PAN-FPN]  
    - Feature fusion
    - Top-down + Bottom-up paths
    ↓
[Head - Detection Head]
    - Classification branch
    - Regression branch
    ↓
Output: Bounding boxes + Classes + Confidences
```

---

## 🔍 **2. POST-PROCESSING - DISTANCE-BASED DETECTION**

### Logic phát hiện hút thuốc

Thuật toán tùy chỉnh dựa trên **khoảng cách Euclidean** giữa cigarette và vùng đầu/thân trên của person.

### a) **Bounding Box Region Extraction**

Trích xuất vùng đầu từ person bounding box:

```python
def get_head_region(person_box):
    """
    Lấy vùng đầu (20% phần trên của person box)
    
    Args:
        person_box: [x1, y1, x2, y2]
    Returns:
        [x1, y1, x2, y2_head]
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    head_height = height * 0.2  # 20% top
    y2_head = y1 + head_height
    return [x1, y1, x2, y2_head]
```

Trích xuất nửa trên cơ thể:

```python
def get_upper_body_region(person_box):
    """
    Lấy vùng nửa trên cơ thể (50% phần trên)
    
    Args:
        person_box: [x1, y1, x2, y2]
    Returns:
        [x1, y1, x2, y2_upper]
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    y2_upper = y1 + height * 0.5
    return [x1, y1, x2, y2_upper]
```

### b) **Euclidean Distance Calculation**

Tính khoảng cách từ cigarette đến target region:

```python
def calculate_distance_to_box(point_box, target_box):
    """
    Tính khoảng cách Euclidean từ tâm point_box đến target_box
    Nếu overlap → khoảng cách = 0
    
    Args:
        point_box: [x1, y1, x2, y2] - cigarette box
        target_box: [x1, y1, x2, y2] - head/upper body box
    Returns:
        float: Khoảng cách (pixels)
    """
    # Tâm của cigarette
    cx = (point_box[0] + point_box[2]) / 2
    cy = (point_box[1] + point_box[3]) / 2
    
    # Kiểm tra overlap
    if (target_box[0] <= cx <= target_box[2] and 
        target_box[1] <= cy <= target_box[3]):
        return 0.0  # Cigarette nằm trong target box
    
    # Tìm điểm gần nhất trên target_box
    closest_x = max(target_box[0], min(cx, target_box[2]))
    closest_y = max(target_box[1], min(cy, target_box[3]))
    
    # Euclidean distance
    distance = sqrt((cx - closest_x)² + (cy - closest_y)²)
    return distance
```

**Công thức Euclidean Distance:**

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

### c) **Two-tier Detection Logic**

Hệ thống phát hiện 2 cấp độ:

```python
def is_smoking_detected(results, head_threshold=80, upper_threshold=150):
    """
    Phát hiện smoking dựa trên khoảng cách
    
    Args:
        head_threshold: Khoảng cách tối đa đến đầu (80px)
        upper_threshold: Khoảng cách tối đa đến nửa trên (150px)
    """
    for person in persons:
        head_region = get_head_region(person)
        upper_region = get_upper_body_region(person)
        
        for cigarette in cigarettes:
            dist_to_head = calculate_distance(cigarette, head_region)
            dist_to_upper = calculate_distance(cigarette, upper_region)
            
            # Tier 1: Gần đầu (strict) - VẼ ĐƯỜNG NỐI
            if dist_to_head < head_threshold:
                return True, "SMOKING", draw_line=True
            
            # Tier 2: Gần nửa trên cơ thể (lenient) - KHÔNG VẼ
            elif dist_to_upper < upper_threshold:
                return True, "SMOKING", draw_line=False
    
    return False, "NON-SMOKING"
```

**Diagram:**

```
Person Box
┌─────────────┐
│   HEAD 20%  │ ← head_threshold (80px)
├─────────────┤
│             │
│   UPPER     │ ← upper_threshold (150px)
│   50%       │
├─────────────┤
│   LOWER     │
│   50%       │
└─────────────┘
```

---

## 🎨 **3. FALSE POSITIVE FILTERING**

### Thuật toán lọc dựa trên heuristics

Loại bỏ cigarette detections không hợp lệ:

```python
def filter_cigarette_detections(results, 
                                min_conf_cigarette=0.35,
                                min_aspect_ratio=2.0,
                                max_aspect_ratio=7.0,
                                min_area=30,
                                max_area=8000,
                                max_distance_to_person=400):
    """
    Lọc cigarette detections dựa trên các tiêu chí:
    1. Confidence threshold
    2. Aspect ratio (cigarette phải dài, mỏng)
    3. Size (area phải hợp lý)
    4. Distance to person (cigarette phải gần người)
    """
    filtered_boxes = []
    
    for box in cigarette_boxes:
        # 1. Confidence filtering
        if box.conf < min_conf_cigarette:
            continue
        
        # 2. Aspect ratio filtering
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        aspect_ratio = width / height
        
        if not (min_aspect_ratio < aspect_ratio < max_aspect_ratio):
            continue  # Cigarette phải dài, mỏng (2:1 đến 7:1)
        
        # 3. Size filtering
        area = width * height
        if not (min_area < area < max_area):
            continue  # Loại bỏ quá nhỏ hoặc quá lớn
        
        # 4. Distance to person filtering
        if person_boxes:
            min_dist = min(distance(box, p) for p in person_boxes)
            if min_dist > max_distance_to_person:
                continue  # Cigarette phải gần người
        
        filtered_boxes.append(box)
    
    return filtered_boxes
```

### Adaptive Thresholds

Tự động điều chỉnh thresholds theo kích thước ảnh:

```python
def get_recommended_thresholds(image_size):
    """
    Tính thresholds tối ưu dựa trên kích thước ảnh
    
    Args:
        image_size: (width, height)
    Returns:
        dict: Recommended thresholds
    """
    width, height = image_size
    img_area = width * height
    
    # Small image (< 200k pixels)
    if img_area < 200_000:
        return {
            'min_conf_cigarette': 0.35,
            'min_aspect_ratio': 2.0,
            'max_aspect_ratio': 7.0,
            'min_area': 30,
            'max_area': 3000,
            'max_distance_to_person': 200
        }
    
    # Medium image (200k - 500k pixels)
    elif img_area < 500_000:
        return {
            'min_conf_cigarette': 0.35,
            'min_aspect_ratio': 2.0,
            'max_aspect_ratio': 7.0,
            'min_area': 50,
            'max_area': 5000,
            'max_distance_to_person': 300
        }
    
    # Large image (> 500k pixels)
    else:
        return {
            'min_conf_cigarette': 0.35,
            'min_aspect_ratio': 2.0,
            'max_aspect_ratio': 7.0,
            'min_area': 100,
            'max_area': 8000,
            'max_distance_to_person': 400
        }
```

---

## 📊 **4. NON-MAXIMUM SUPPRESSION (NMS)**

### Loại bỏ duplicate detections

YOLOv8 tự động áp dụng NMS để loại bỏ các bounding boxes trùng lặp:

```python
results = model.predict(
    source=img,
    iou=0.7,    # IoU threshold cho NMS
    conf=0.20   # Confidence threshold
)
```

### NMS Algorithm:

```
1. Sort tất cả boxes theo confidence (cao → thấp)
2. Chọn box có confidence cao nhất → thêm vào output
3. Tính IoU giữa box đã chọn với các boxes còn lại
4. Loại bỏ các boxes có IoU > threshold (0.7)
5. Lặp lại bước 2-4 cho đến khi hết boxes
```

**Pseudocode:**

```python
def non_maximum_suppression(boxes, iou_threshold=0.7):
    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x.conf, reverse=True)
    
    keep = []
    while boxes:
        # Pick highest confidence box
        best = boxes[0]
        keep.append(best)
        boxes = boxes[1:]
        
        # Remove overlapping boxes
        boxes = [box for box in boxes 
                if iou(best, box) < iou_threshold]
    
    return keep
```

---

## 🧮 **5. IoU (INTERSECTION OVER UNION)**

### Đo độ overlap giữa 2 bounding boxes

```python
def calculate_iou(box1, box2):
    """
    Tính IoU (Intersection over Union)
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    Returns:
        float: IoU value (0-1)
    """
    # Tính toạ độ vùng giao nhau
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Kiểm tra có giao nhau không
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    # Diện tích giao nhau (Intersection)
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Diện tích từng box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Diện tích hợp nhau (Union)
    union_area = box1_area + box2_area - inter_area
    
    # IoU = Intersection / Union
    iou = inter_area / (union_area + 1e-6)
    return iou
```

**Công thức IoU:**

$$IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}$$

**Ví dụ:**

```
Box A: [0, 0, 4, 4] → Area = 16
Box B: [2, 2, 6, 6] → Area = 16

Intersection: [2, 2, 4, 4] → Area = 4
Union: 16 + 16 - 4 = 28

IoU = 4/28 = 0.143
```

---

## 🎓 **6. TRAINING ALGORITHMS**

### a) **Optimizer: AdamW**

**Adam with Weight Decay (Decoupled)**

```python
optimizer='AdamW'
lr0=0.01          # Initial learning rate
lrf=0.001         # Final learning rate factor
momentum=0.937    # Momentum
weight_decay=0.0005
```

**AdamW Components:**

1. **Adaptive Learning Rates** (Adam)
   - Mỗi parameter có learning rate riêng
   - Tự động điều chỉnh dựa trên gradient history

2. **Momentum**
   - Sử dụng exponentially weighted averages của gradients
   - Giúp thoát khỏi local minima

3. **Weight Decay** (L2 Regularization)
   - Decoupled từ gradient descent
   - Prevent overfitting

**Update Rule:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon} - \lambda\theta_{t-1}$$

### b) **Loss Functions**

YOLOv8 sử dụng **multi-task loss**:

```python
Total_Loss = box_loss × 7.5 + cls_loss × 2.0 + dfl_loss × 1.5
```

#### **1. Box Loss: CIoU (Complete IoU)**

```python
box_loss = 7.5  # Weight
```

**CIoU = IoU Loss + Distance + Aspect Ratio**

$$\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

Trong đó:
- $IoU$: Intersection over Union
- $\rho(b, b^{gt})$: Euclidean distance giữa centers
- $c$: Diagonal length của smallest enclosing box
- $v$: Aspect ratio consistency
- $\alpha$: Trade-off parameter

#### **2. Classification Loss: BCE (Binary Cross Entropy)**

```python
cls_loss = 2.0  # Weight (tăng cho cigarette detection)
```

**Binary Cross Entropy:**

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

Trong đó:
- $y_i$: Ground truth label (0 hoặc 1)
- $\hat{y}_i$: Predicted probability

#### **3. DFL Loss: Distribution Focal Loss**

```python
dfl_loss = 1.5  # Weight
```

Dùng cho **bounding box regression**:
- Thay vì predict 1 giá trị cố định
- Predict distribution của possible values
- Tăng accuracy cho box coordinates

### c) **Learning Rate Scheduling**

**Cosine Annealing với Warmup:**

```python
warmup_epochs = 5
lr0 = 0.01        # Initial LR
lrf = 0.001       # Final LR factor

# Warmup phase (epochs 0-5)
lr = warmup_bias_lr → lr0

# Main training (epochs 5-50)
lr = lr0 → (lr0 * lrf)  # Cosine decay
```

**Công thức Cosine Annealing:**

$$lr_t = lr_{final} + \frac{1}{2}(lr_{initial} - lr_{final})(1 + \cos(\frac{t\pi}{T}))$$

---

## 🎨 **7. DATA AUGMENTATION ALGORITHMS**

### Augmentation Techniques

```python
# Mosaic Augmentation
mosaic = 1.0

# MixUp
mixup = 0.15

# Copy-Paste  
copy_paste = 0.1

# HSV Color Space
hsv_h = 0.015      # Hue shift
hsv_s = 0.7        # Saturation
hsv_v = 0.4        # Value (brightness)

# Geometric Transforms
degrees = 10       # Rotation (-10° to +10°)
translate = 0.1    # Translation (±10%)
scale = 0.5        # Scaling (0.5x to 1.5x)
flipud = 0.0       # Vertical flip
fliplr = 0.5       # Horizontal flip (50%)
```

### a) **Mosaic Augmentation**

Ghép 4 ảnh thành 1 ảnh training:

```
┌─────────┬─────────┐
│ Image 1 │ Image 2 │
│         │         │
├─────────┼─────────┤
│ Image 3 │ Image 4 │
│         │         │
└─────────┴─────────┘
```

**Lợi ích:**
- Tăng diversity
- Học multi-object context
- Tăng small object detection

### b) **MixUp Augmentation**

Trộn 2 ảnh với alpha blending:

$$Image_{mixed} = \lambda \times Image_1 + (1-\lambda) \times Image_2$$
$$Label_{mixed} = \lambda \times Label_1 + (1-\lambda) \times Label_2$$

**Lợi ích:**
- Tăng regularization
- Giảm overfitting
- Smooth decision boundaries

### c) **HSV Color Jittering**

Thay đổi màu sắc trong không gian HSV:

```python
def hsv_augment(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    # Convert RGB → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Random gains
    h = hsv[:, :, 0] * (1 + random.uniform(-h_gain, h_gain))
    s = hsv[:, :, 1] * (1 + random.uniform(-s_gain, s_gain))
    v = hsv[:, :, 2] * (1 + random.uniform(-v_gain, v_gain))
    
    # Clip and merge
    hsv = np.stack([h, s, v], axis=2).astype(np.uint8)
    
    # Convert back HSV → RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

### d) **Geometric Transforms**

**Rotation:**
```python
angle = random.uniform(-degrees, degrees)
M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
image = cv2.warpAffine(image, M, (width, height))
```

**Translation:**
```python
tx = random.uniform(-translate, translate) * width
ty = random.uniform(-translate, translate) * height
M = np.array([[1, 0, tx], [0, 1, ty]])
image = cv2.warpAffine(image, M, (width, height))
```

**Scaling:**
```python
scale_factor = random.uniform(1-scale, 1+scale)
new_size = (int(width*scale_factor), int(height*scale_factor))
image = cv2.resize(image, new_size)
```

---

## 📈 **8. COMPLETE PIPELINE FLOWCHART**

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT IMAGE                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│            DATA AUGMENTATION (Training only)            │
│  - Mosaic, MixUp, Copy-Paste                           │
│  - HSV Jittering, Geometric Transforms                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  YOLOv8 DETECTION                       │
│  ┌─────────────────────────────────────────────┐       │
│  │ Backbone (CSPDarknet)                       │       │
│  │   - Multi-scale feature extraction         │       │
│  └──────────────┬──────────────────────────────┘       │
│                 │                                        │
│  ┌──────────────▼──────────────────────────────┐       │
│  │ Neck (PAN-FPN)                              │       │
│  │   - Feature pyramid                         │       │
│  │   - Top-down + Bottom-up paths              │       │
│  └──────────────┬──────────────────────────────┘       │
│                 │                                        │
│  ┌──────────────▼──────────────────────────────┐       │
│  │ Head (Detection)                            │       │
│  │   - Bounding box regression                 │       │
│  │   - Classification                          │       │
│  └──────────────┬──────────────────────────────┘       │
└─────────────────┼───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              NON-MAXIMUM SUPPRESSION                    │
│  - Remove duplicate detections                          │
│  - IoU threshold: 0.7                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│           CIGARETTE FALSE POSITIVE FILTER               │
│  - Confidence check (conf > 0.35)                       │
│  - Aspect ratio check (2:1 to 7:1)                      │
│  - Size check (area in valid range)                     │
│  - Distance to person check                             │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│         PERSON + CIGARETTE DETECTIONS                   │
│         (Filtered, High-confidence)                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          DISTANCE CALCULATION (Euclidean)               │
│  For each (person, cigarette) pair:                     │
│    - Get head region (top 20%)                          │
│    - Get upper body region (top 50%)                    │
│    - Calculate distance cigarette → head               │
│    - Calculate distance cigarette → upper body         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│            SMOKING DETECTION LOGIC                      │
│  ┌───────────────────────────────────────────┐         │
│  │ IF distance_to_head < 80px:               │         │
│  │    → SMOKING (draw line)                  │         │
│  │ ELSE IF distance_to_upper < 150px:        │         │
│  │    → SMOKING (no line)                    │         │
│  │ ELSE:                                      │         │
│  │    → NON-SMOKING                           │         │
│  └───────────────────────────────────────────┘         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│         OUTPUT: SMOKING / NON-SMOKING                   │
│         + Annotated Image with Bounding Boxes           │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 **TỔNG HỢP CÁC THUẬT TOÁN**

| Component | Algorithm | Type | Purpose |
|-----------|-----------|------|---------|
| **Object Detection** | YOLOv8 | Deep Learning (CNN) | Detect Person & Cigarette |
| **Backbone** | CSPDarknet | CNN | Feature extraction |
| **Neck** | PAN-FPN | Feature Pyramid | Multi-scale fusion |
| **Post-processing** | NMS | Greedy algorithm | Remove duplicates |
| **Filtering** | Rule-based Heuristics | Logic | Remove false positives |
| **Distance Calculation** | Euclidean Distance | Geometric | Measure proximity |
| **Smoking Detection** | Distance Thresholding | Logic-based | Final decision |
| **IoU Calculation** | Intersection over Union | Geometric | Measure overlap |
| **Optimizer** | AdamW | Gradient Descent | Parameter updates |
| **Box Loss** | CIoU Loss | Regression | Bounding box accuracy |
| **Classification Loss** | BCE Loss | Classification | Class prediction |
| **DFL Loss** | Distribution Focal Loss | Regression | Box coordinate refinement |
| **LR Scheduling** | Cosine Annealing + Warmup | Optimization | Learning rate decay |
| **Augmentation** | Mosaic + MixUp + Transforms | Data Processing | Increase diversity |

---

## 💡 **ĐẶC ĐIỂM NỔI BẬT**

### 1. **Hybrid Approach**
Kết hợp Deep Learning (YOLOv8) với Classical CV (distance, geometry) cho kết quả tốt hơn thuần DL.

### 2. **Domain-Specific Logic**
- Two-tier detection (head + upper body)
- Adaptive thresholds based on image size
- Cigarette filtering heuristics

### 3. **Efficient Architecture**
- Single-stage detector (fast)
- Anchor-free (simpler)
- Multi-scale features (robust)

### 4. **Robust Post-processing**
- NMS loại duplicate
- Heuristic filters loại false positives
- Distance-based verification

### 5. **Advanced Training**
- AdamW optimizer (state-of-the-art)
- Multi-loss training
- Rich augmentation pipeline

---

## 📚 **REFERENCES**

1. **YOLOv8**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
2. **AdamW**: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
3. **CIoU Loss**: "Distance-IoU Loss" (Zheng et al., 2020)
4. **Mosaic Augmentation**: YOLOv4 paper (Bochkovskiy et al., 2020)
5. **MixUp**: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)

---

**Ngày cập nhật**: 23/12/2025  
**Version**: 1.0  
**Model**: smoking_detection_v7_improved  
**Framework**: YOLOv8 (Ultralytics)
