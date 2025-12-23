# 📊 ẢNH HƯỞNG CỦA DATA TRAIN/VAL/TEST ĐẾN KẾT QUẢ TRAIN

**Dự án:** Smoking Detection System (YOLOv8s)  
**Dataset hiện tại:** 11,910 train + 312 val + 122 test = 12,344 images

---

## 📋 MỤC LỤC

1. [Tổng Quan Data Splits](#1-tổng-quan-data-splits)
2. [Train Data - Dữ Liệu Huấn Luyện](#2-train-data---dữ-liệu-huấn-luyện)
3. [Validation Data - Dữ Liệu Xác Thực](#3-validation-data---dữ-liệu-xác-thực)
4. [Test Data - Dữ Liệu Kiểm Tra](#4-test-data---dữ-liệu-kiểm-tra)
5. [Tỷ Lệ Phân Chia Data](#5-tỷ-lệ-phân-chia-data)
6. [Vấn Đề Phổ Biến](#6-vấn-đề-phổ-biến)
7. [Phân Tích Dataset Hiện Tại](#7-phân-tích-dataset-hiện-tại)
8. [Khuyến Nghị Cải Thiện](#8-khuyến-nghị-cải-thiện)

---

## 1. TỔNG QUAN DATA SPLITS

### 1.1. Ba Tập Dữ Liệu Trong Machine Learning

```
┌─────────────────────────────────────────────────────────────┐
│                    TOÀN BỘ DATASET                          │
│                    (12,344 images)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   TRAIN DATA     │  │  VAL DATA   │  │  TEST DATA   │  │
│  │   11,910 (96.5%) │  │  312 (2.5%) │  │  122 (1.0%)  │  │
│  │                  │  │             │  │              │  │
│  │  Model HỌC       │  │  Model ĐIỀU │  │  Đánh giá    │  │
│  │  từ data này     │  │  CHỈNH ở đây│  │  CUỐI CÙNG   │  │
│  └──────────────────┘  └─────────────┘  └──────────────┘  │
│         ↓                     ↓                  ↓         │
│    Học patterns         Tối ưu HP         Đánh giá thật   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2. Vai Trò Của Từng Tập

| Tập | % | Vai Trò | Khi Nào Sử Dụng | Ảnh Hưởng |
|-----|---|---------|-----------------|-----------|
| **Train** | 70-80% | Model **HỌC** patterns | Mỗi epoch | ⭐⭐⭐⭐⭐ QUAN TRỌNG NHẤT |
| **Val** | 10-15% | **ĐIỀU CHỈNH** hyperparameters | Sau mỗi epoch | ⭐⭐⭐⭐ RẤT QUAN TRỌNG |
| **Test** | 10-15% | **ĐÁNH GIÁ** cuối cùng | Sau khi train xong | ⭐⭐⭐ QUAN TRỌNG |

---

## 2. TRAIN DATA - DỮ LIỆU HUẤN LUYỆN

### 2.1. Vai Trò

**Train data là nguồn kiến thức của model:**
- Model **HỌC TẤT CẢ** từ train data
- Mỗi epoch, model xem toàn bộ train data
- Model điều chỉnh weights dựa trên train data
- **QUYẾT ĐỊNH 80%** chất lượng model

### 2.2. Ảnh Hưởng Đến Kết Quả

#### ✅ **Train Data NHIỀU** (10,000+ images)

**Ưu điểm:**
```
✅ Model học được nhiều patterns
✅ Giảm overfitting
✅ Tổng quát hóa tốt
✅ Robust với unseen data
✅ Chính xác cao
```

**Dataset hiện tại:** 11,910 images ✅ RẤT TỐT

#### ❌ **Train Data ÍT** (<1,000 images)

**Nhược điểm:**
```
❌ Model học không đủ patterns
❌ Dễ overfitting (học thuộc)
❌ Không tổng quát
❌ Sai khi gặp trường hợp mới
❌ Chính xác thấp
```

### 2.3. Ảnh Hưởng Cụ Thể

| Số Lượng Train Data | mAP50 Dự Đoán | Overfitting | Khả Năng Tổng Quát |
|---------------------|---------------|-------------|-------------------|
| < 500 images | 30-50% | ❌ CAO | ❌ KÉM |
| 500-1,000 images | 50-65% | ⚠️ TRUNG BÌNH | ⚠️ TRUNG BÌNH |
| 1,000-5,000 images | 65-75% | ✅ THẤP | ✅ TỐT |
| 5,000-10,000 images | 75-85% | ✅ RẤT THẤP | ✅ RẤT TỐT |
| **> 10,000 images** | **80-90%** | **✅ CỰC THẤP** | **✅ XUẤT SẮC** |

**Dataset hiện tại: 11,910 → Nằm ở level XUẤT SẮC ✅**

### 2.4. Quality vs Quantity

**Chất lượng QUAN TRỌNG HƠN số lượng:**

```python
# BAD: 10,000 images chất lượng kém
- Ảnh mờ, tối
- Labels sai
- Bị duplicate
- Không đa dạng
→ mAP50: 60-70% ❌

# GOOD: 5,000 images chất lượng tốt
- Ảnh rõ nét
- Labels chính xác 100%
- Đa dạng (góc độ, ánh sáng, môi trường)
- Không duplicate
→ mAP50: 75-85% ✅
```

### 2.5. Cân Bằng Classes

**CRITICAL cho cigarette detection:**

```python
# Dataset hiện tại (ước tính):
Train Data:
  - Person: ~70-80% instances      ← DOMINANT class
  - Cigarette: ~20-30% instances   ← MINORITY class

# Vấn đề:
Model học THIÊN VỆ về Person
→ Cigarette detection YẾU (54.17% mAP50 trong v2)

# Giải pháp đã áp dụng trong train.py:
cls=2.0          # Tăng 4x class loss
copy_paste=0.3   # Tạo thêm cigarette instances
scale=0.8        # Giữ size cigarette
```

---

## 3. VALIDATION DATA - DỮ LIỆU XÁC THỰC

### 3.1. Vai Trò

**Val data giúp ĐIỀU CHỈNH model trong quá trình train:**
- Model **KHÔNG HỌC** từ val data
- Sau mỗi epoch, test model trên val data
- Dựa vào val metrics để early stopping
- Chọn best.pt dựa trên val mAP50

### 3.2. Ảnh Hưởng Đến Kết Quả

#### ✅ **Val Data ĐỦ** (300+ images)

**Ưu điểm:**
```
✅ Đại diện cho real-world data
✅ Metrics đáng tin cậy
✅ Early stopping chính xác
✅ Chọn best epoch đúng
✅ Tránh overfitting
```

**Dataset hiện tại:** 312 images ✅ ĐỦ

#### ❌ **Val Data ÍT** (<100 images)

**Nhược điểm:**
```
❌ Metrics không ổn định (biến động cao)
❌ Early stopping sai
❌ Chọn sai best epoch
❌ Không phát hiện overfitting
❌ Kết quả không đáng tin
```

### 3.3. Tần Suất Validation

```python
# Trong train.py:
epochs=100  # 100 lần training

# YOLOv8 tự động validate sau MỖI epoch:
Epoch 1: Train → Validate (312 images)
Epoch 2: Train → Validate (312 images)
...
Epoch 100: Train → Validate (312 images)

# Chọn best.pt:
best_epoch = epoch with highest val/mAP50
# Ví dụ: Epoch 87 có mAP50=83.93% → Save best.pt
```

### 3.4. Early Stopping

```python
# Config trong train.py:
patience=30  # Dừng nếu 30 epochs không cải thiện

# Ví dụ:
Epoch 70: val/mAP50 = 83.5%  ← Peak
Epoch 71-100: val/mAP50 ≤ 83.5%  (giảm hoặc bằng)
→ STOP tại epoch 100 (hoặc 70+30=100)
```

**Val data QUÁ ÍT → Early stopping SAI → Train thừa/thiếu epochs**

### 3.5. Phân Tích Val Data Hiện Tại

```
Val Data: 312 images (2.5% dataset)

Tỷ lệ: Train/Val = 11,910/312 = 38:1

Standard practice: 5:1 đến 10:1
→ 38:1 HƠI CAO ⚠️

Lý tưởng: Val nên ~800-1,200 images (10%)
```

**Ảnh hưởng:**
- ✅ Val metrics tương đối ổn định (312 images đủ lớn)
- ⚠️ Nhưng có thể không đại diện đầy đủ
- ⚠️ Một số edge cases có thể bị miss

---

## 4. TEST DATA - DỮ LIỆU KIỂM TRA

### 4.1. Vai Trò

**Test data là "kỳ thi cuối kỳ" của model:**
- Model **KHÔNG BAO GIỜ THẤY** test data trong quá trình train
- Chỉ test **MỘT LẦN DUY NHẤT** sau khi train xong
- Đánh giá **CHÍNH THỨC** performance
- So sánh với baseline/other models

### 4.2. Ảnh Hưởng Đến Kết Quả

**Test data KHÔNG ảnh hưởng đến quá trình training:**
- ✅ Model không học từ test data
- ✅ Không ảnh hưởng đến weights
- ✅ Chỉ dùng để đánh giá

**Nhưng ảnh hưởng đến ĐÁNH GIÁ:**

#### ✅ **Test Data ĐỦ LỚN** (100+ images)

```
✅ Đánh giá chính xác
✅ Metrics ổn định
✅ Tin cậy được kết quả
✅ Có thể public/báo cáo
```

**Dataset hiện tại:** 122 images ✅ ĐỦ

#### ❌ **Test Data QUÁ ÍT** (<50 images)

```
❌ Metrics không đáng tin
❌ Biến động cao
❌ May mắn/không may
❌ Không đại diện
```

### 4.3. Test vs Val Metrics

```python
# Trường hợp LÝ TƯỞNG:
Val mAP50:  83.5%
Test mAP50: 82.8%  ← Chênh lệch ~1%
→ ✅ Model TỔNG QUÁT TỐT

# Trường hợp OVERFITTING:
Val mAP50:  90.0%
Test mAP50: 70.0%  ← Chênh lệch 20%
→ ❌ Model HỌC THUỘC val data

# Trường hợp VAL DATA KHÔNG ĐẠI DIỆN:
Val mAP50:  75.0%
Test mAP50: 85.0%  ← Test tốt hơn val
→ ⚠️ Val data không đại diện tốt
```

### 4.4. Phân Tích Test Data Hiện Tại

```
Test Data: 122 images (1.0% dataset)

Tỷ lệ: Train/Test = 11,910/122 = 97:1

Standard practice: 5:1 đến 10:1
→ 97:1 RẤT CAO ⚠️

Lý tưởng: Test nên ~1,000-1,500 images (10-15%)
```

**Ảnh hưởng:**
- ⚠️ 122 images CÓ THỂ không đại diện đầy đủ
- ⚠️ Edge cases có thể bị miss
- ⚠️ Metrics có thể biến động

---

## 5. TỶ LỆ PHÂN CHIA DATA

### 5.1. Standard Practice

| Tổng Dataset | Train | Val | Test | Use Case |
|--------------|-------|-----|------|----------|
| **< 1,000** | 70% | 15% | 15% | Small projects |
| **1,000-10,000** | 75% | 15% | 10% | Medium projects |
| **> 10,000** | 80% | 10% | 10% | Large projects |
| **> 100,000** | 90% | 5% | 5% | Huge datasets |

### 5.2. Dataset Hiện Tại

```python
Total: 12,344 images

Actual:
  Train: 11,910 (96.5%)  ← QUÁ CAO ⚠️
  Val:   312    (2.5%)   ← QUÁ THẤP ⚠️
  Test:  122    (1.0%)   ← QUÁ THẤP ⚠️

Recommended (80/10/10):
  Train: 9,875  (80%)
  Val:   1,234  (10%)
  Test:  1,235  (10%)

Difference:
  Train: +2,035 images
  Val:   -922 images   ← THIẾU RẤT NHIỀU
  Test:  -1,113 images ← THIẾU RẤT NHIỀU
```

### 5.3. Ảnh Hưởng Của Phân Chia Hiện Tại

#### ⚠️ **96.5% Train - QUÁ CAO**

**Hậu quả:**
```
⚠️ Val/Test quá ít → Không đại diện
⚠️ Metrics không tin cậy
⚠️ Có thể overfitting mà không phát hiện
⚠️ Early stopping không chính xác
```

**Nhưng:**
```
✅ Model học được RẤT NHIỀU patterns
✅ Hiếm khi underfitting
✅ Chất lượng training cao
```

#### ⚠️ **2.5% Val - QUÁ THẤP**

**Hậu quả:**
```
⚠️ Val metrics biến động
⚠️ Best.pt có thể không phải best thật
⚠️ Early stopping không chính xác
⚠️ Không phát hiện overfitting tốt
```

**Đề xuất:** Val nên ≥ 1,000 images (10%)

#### ⚠️ **1.0% Test - QUÁ THẤP**

**Hậu quả:**
```
⚠️ Test metrics không đáng tin
⚠️ Có thể lucky/unlucky
⚠️ Không đại diện real-world
⚠️ Khó so sánh với other models
```

**Đề xuất:** Test nên ≥ 1,000 images (10%)

---

## 6. VẤN ĐỀ PHỔ BIẾN

### 6.1. Data Leakage

**Vấn đề NGHIÊM TRỌNG:**

```python
# BAD: Cùng một người/scene trong nhiều splits
Train: person_A_frame_001.jpg
Val:   person_A_frame_002.jpg  ← Data leakage!
Test:  person_A_frame_003.jpg  ← Data leakage!

→ Model HỌC THUỘC person A
→ Metrics GIẢ TẠO cao
```

**Giải pháp:**

```python
# GOOD: Phân chia theo PERSON/SCENE
Train: person_A_*.jpg, person_B_*.jpg
Val:   person_C_*.jpg, person_D_*.jpg
Test:  person_E_*.jpg, person_F_*.jpg

→ Model phải TỔNG QUÁT thật sự
```

### 6.2. Imbalanced Classes

**Dataset hiện tại (ước tính):**

```python
# Toàn dataset:
Person:    ~70-80% instances
Cigarette: ~20-30% instances

# Nếu phân chia NGẪU NHIÊN:
Train: Person 75%, Cigarette 25%  ← OK
Val:   Person 75%, Cigarette 25%  ← OK
Test:  Person 75%, Cigarette 25%  ← OK

# Nhưng nếu Val/Test QUÁ ÍT:
Val (312 images):
  Person instances: ~500-600
  Cigarette instances: ~100-150  ← QUÁ ÍT ⚠️

→ Val metrics cho CIGARETTE không tin cậy
→ Giải thích tại sao Cigarette mAP50 thấp hơn Person
```

### 6.3. Low-Quality Data

**Ảnh hưởng RẤT LỚN:**

```python
# 10% labels SAI trong train data:
10% x 11,910 = 1,191 images SAI

Ảnh hưởng:
- Model học SAI patterns
- mAP50 giảm 5-10%
- Cigarette detection kém
- Precision giảm

# CRITICAL: Kiểm tra labels!
Công cụ: Roboflow Label Quality Check
```

### 6.4. Overfitting vs Underfitting

```python
# Overfitting (học thuộc):
Train loss: 0.05  ← RẤT THẤP
Val loss:   0.30  ← CAO

→ Model học thuộc train data
→ Không tổng quát

# Underfitting (học chưa đủ):
Train loss: 0.50  ← CAO
Val loss:   0.48  ← CAO

→ Model chưa học đủ
→ Cần train thêm epochs

# Good fit (lý tưởng):
Train loss: 0.15  ← THẤP
Val loss:   0.18  ← GẦN train loss

→ Model tổng quát tốt ✅
```

---

## 7. PHÂN TÍCH DATASET HIỆN TẠI

### 7.1. Tổng Quan

```python
Dataset: Roboflow smoking-tasfx v4
Total: 12,344 images

Split:
├── Train: 11,910 (96.5%)  ⚠️ QUÁ CAO
├── Val:   312    (2.5%)   ⚠️ QUÁ THẤP
└── Test:  122    (1.0%)   ⚠️ QUÁ THẤP

Classes:
├── Cigarette (class 0): ~20-30% instances
└── Person (class 1):    ~70-80% instances
```

### 7.2. Điểm Mạnh ✅

1. **Train data RẤT LỚN (11,910)**
   - Model học được nhiều patterns
   - Hiếm overfitting
   - Chất lượng training cao

2. **Tổng dataset LỚN (12,344)**
   - Đủ để train model tốt
   - Đa dạng scenarios

3. **Data quality (Roboflow)**
   - Labels chính xác
   - Format chuẩn YOLO
   - Preprocessing tốt

### 7.3. Điểm Yếu ⚠️

1. **Val data QUÁ ÍT (312 = 2.5%)**
   ```
   Standard: 10% (1,200+ images)
   Actual:   2.5% (312 images)
   → Thiếu 900+ images
   
   Ảnh hưởng:
   - Val metrics không ổn định
   - Best.pt có thể không best thật
   - Early stopping không chính xác
   ```

2. **Test data QUÁ ÍT (122 = 1.0%)**
   ```
   Standard: 10% (1,200+ images)
   Actual:   1.0% (122 images)
   → Thiếu 1,100+ images
   
   Ảnh hưởng:
   - Test metrics không đáng tin
   - Không đại diện real-world
   - Edge cases bị miss
   ```

3. **Class imbalance (Cigarette 20-30%)**
   ```
   Person:    70-80% ← Dominant
   Cigarette: 20-30% ← Minority
   
   Ảnh hưởng:
   - Model thiên vệ Person
   - Cigarette detection yếu
   - Đã fix bằng cls=2.0 trong train.py ✅
   ```

### 7.4. Kết Quả Training v3

```python
Model: YOLOv8s
Epochs: 60/100
Time: 3.85 hours

Final Metrics:
├── mAP50:     83.93%  ✅ TỐT
├── Precision: 82.36%  ✅ TỐT
├── Recall:    79.21%  ✅ TỐT
└── mAP50-95:  ~60%    ✅ OK

Losses:
├── Box loss: 0.81  ✅ THẤP
├── Cls loss: 3.42  ⚠️ CAO (do cigarette khó)
└── DFL loss: 1.15  ✅ THẤP
```

**Phân tích:**
- Overall metrics RẤT TỐT (83.93%)
- Nhưng chưa rõ per-class breakdown
- Cls loss CAO → Cigarette vẫn challenging
- Cần validate per-class để xác nhận

---

## 8. KHUYẾN NGHỊ CẢI THIỆN

### 8.1. Re-split Dataset (Khuyến Nghị Cao)

**Phân chia lại 80/10/10:**

```python
# Script để re-split
import os
import shutil
import random
from pathlib import Path

def resplit_dataset():
    """
    Re-split dataset 80/10/10 thay vì 96.5/2.5/1.0
    """
    
    # Paths
    dataset_root = Path(r"e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\dataset\smoking_train_image")
    
    # Collect all images
    all_images = []
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_root / split / 'images'
        for img in img_dir.glob('*.jpg'):
            all_images.append(img.stem)  # Filename without extension
    
    print(f"Total images: {len(all_images)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_images)
    
    # Calculate splits (80/10/10)
    total = len(all_images)
    train_end = int(total * 0.80)
    val_end = train_end + int(total * 0.10)
    
    train_files = all_images[:train_end]           # 9,875
    val_files = all_images[train_end:val_end]      # 1,234
    test_files = all_images[val_end:]              # 1,235
    
    print(f"\nNew split:")
    print(f"  Train: {len(train_files)} ({len(train_files)/total*100:.1f}%)")
    print(f"  Val:   {len(val_files)} ({len(val_files)/total*100:.1f}%)")
    print(f"  Test:  {len(test_files)} ({len(test_files)/total*100:.1f}%)")
    
    # TODO: Move files to new directories
    # (Implementation needed)

resplit_dataset()
```

**Lợi ích:**
```
✅ Val metrics đáng tin hơn (1,234 thay vì 312)
✅ Test metrics chính xác hơn (1,235 thay vì 122)
✅ Early stopping chính xác
✅ Best.pt thật sự best
✅ Phát hiện overfitting tốt hơn
```

**Nhược điểm:**
```
⚠️ Mất thời gian re-train
⚠️ Không thể compare với v2/v3 cũ
⚠️ Cần validate lại toàn bộ
```

### 8.2. Cross-Validation (Alternative)

**Nếu không muốn re-split:**

```python
# K-Fold Cross-Validation (K=5)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
    print(f"\n=== FOLD {fold+1}/5 ===")
    
    # Train model on train_idx
    # Validate on val_idx
    
    results.append(model.val())

# Average results
avg_mAP50 = sum([r.box.map50 for r in results]) / 5
print(f"\nCross-validation mAP50: {avg_mAP50*100:.2f}%")
```

**Lợi ích:**
```
✅ Sử dụng TẤT CẢ data
✅ Metrics đáng tin nhất
✅ Không waste data
✅ Phát hiện overfitting tốt
```

**Nhược điểm:**
```
❌ Tốn thời gian (train 5 lần)
❌ Cần GPU resources
```

### 8.3. Augmentation Thay Vì Thêm Data

**Nếu không thể thêm data:**

```python
# Trong train.py, tăng augmentation:
mosaic=1.0,          # Mosaic
mixup=0.3,           # ↑ Mixup (0.2 → 0.3)
copy_paste=0.5,      # ↑ Copy-paste (0.3 → 0.5)
scale=0.9,           # ↑ Scale range
degrees=15,          # ↑ Rotation

# Lợi ích:
→ Tạo ra "virtual data" từ train data
→ Tăng diversity
→ Giảm overfitting
→ Không cần data thật mới
```

### 8.4. Per-Class Validation

**CRITICAL để hiểu cigarette performance:**

```python
# Script: validate_per_class.py
from ultralytics import YOLO

model = YOLO('runs/train/smoking_detection_v3_improved/weights/best.pt')

results = model.val(
    data='e:\\LEARN\\@ki1 nam 4\\MACHINE LEARNING\\smoke\\wsf1\\dataset\\smoking_train_image\\data.yaml',
    split='test'
)

# Extract per-class metrics
per_class_map50 = results.box.maps  # Per-class mAP50

print(f"\nPer-Class mAP50:")
print(f"  Cigarette: {per_class_map50[0]*100:.2f}%")
print(f"  Person:    {per_class_map50[1]*100:.2f}%")
print(f"  Average:   {results.box.map50*100:.2f}%")
```

**Chạy script:**
```bash
python validate_per_class.py
```

### 8.5. Collect More Data (Long-term)

**Nếu cần cải thiện lâu dài:**

```
1. Scrape thêm images từ internet
2. Record video → Extract frames
3. Augmentation synthetic data
4. FOCUS: Cigarette minority class

Target:
- Total: 20,000+ images
- Cigarette: Tăng tỷ lệ lên 40-50%
- Train/Val/Test: 80/10/10
```

---

## 📊 BẢNG TỔNG KẾT

### Ảnh Hưởng Của Data Splits

| Factor | Impact on Training | Impact on Validation | Impact on Testing | Priority |
|--------|-------------------|---------------------|-------------------|----------|
| **Train Size** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | 🔴 CRITICAL |
| **Train Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 🔴 CRITICAL |
| **Train Balance** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 🟠 HIGH |
| **Val Size** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 🟠 HIGH |
| **Val Quality** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 🟠 HIGH |
| **Test Size** | - | - | ⭐⭐⭐⭐⭐ | 🟡 MEDIUM |
| **Test Quality** | - | - | ⭐⭐⭐⭐⭐ | 🟡 MEDIUM |

### Dataset Hiện Tại vs Lý Tưởng

| Metric | Hiện Tại | Lý Tưởng | Đánh Giá |
|--------|----------|----------|----------|
| **Total Images** | 12,344 | 15,000+ | ✅ TỐT |
| **Train %** | 96.5% | 80% | ⚠️ QUÁ CAO |
| **Val %** | 2.5% | 10% | ❌ QUÁ THẤP |
| **Test %** | 1.0% | 10% | ❌ QUÁ THẤP |
| **Train Size** | 11,910 | 9,875 | ✅ XUẤT SẮC |
| **Val Size** | 312 | 1,234 | ❌ THIẾU 922 |
| **Test Size** | 122 | 1,235 | ❌ THIẾU 1,113 |
| **Class Balance** | 70/30 | 50/50 | ⚠️ IMBALANCED |

---

## 🎯 CHECKLIST ĐÁNH GIÁ DATASET

### Trước Khi Train:
- [ ] Train data ≥ 80% dataset
- [ ] Val data ≥ 10% dataset (1,000+ images)
- [ ] Test data ≥ 10% dataset (1,000+ images)
- [ ] Không có data leakage giữa splits
- [ ] Classes cân bằng trong mỗi split
- [ ] Labels chính xác 100%
- [ ] Images chất lượng cao (rõ nét, đủ sáng)
- [ ] Đa dạng scenarios (góc độ, môi trường, lighting)

### Sau Khi Train:
- [ ] Val mAP50 và Test mAP50 chênh lệch < 5%
- [ ] Per-class mAP50 đều > 70%
- [ ] Train loss và Val loss gần nhau (< 20% chênh lệch)
- [ ] Confusion matrix không có bias rõ rệt
- [ ] Test trên real-world data OK

---

**Kết luận:** Dataset hiện tại có điểm mạnh là train data RẤT LỚN (11,910), nhưng val/test QUÁ ÍT (312/122). Khuyến nghị re-split về 80/10/10 để có metrics đáng tin hơn, hoặc dùng cross-validation để tận dụng tối đa data hiện có.

---

**Last Updated:** December 11, 2025  
**Dataset:** Roboflow smoking-tasfx v4  
**Current Split:** 96.5% / 2.5% / 1.0% (Train/Val/Test)  
**Recommended Split:** 80% / 10% / 10%
