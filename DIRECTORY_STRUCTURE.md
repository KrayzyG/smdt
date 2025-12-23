# 📂 CẤU TRÚC THƯ MỤC DỰ ÁN

**YOLOv8 Smoking Detection System - Directory Structure**

**Last Updated:** 23/12/2025

---

## 📋 TỔNG QUAN CẤU TRÚC

```
smoking_with_yolov8 + aug/
│
├── 🔧 CORE SCRIPTS (6 files)
├── 🎓 TRAINING SCRIPTS (4 files)
├── 📊 DATASET TOOLS (1 file)
├── 📁 INPUT/OUTPUT FOLDERS
├── 🏆 MODELS & WEIGHTS
├── 📚 DOCUMENTATION
└── 📋 CONFIGURATION FILES
```

---

## 🗂️ CHI TIẾT CẤU TRÚC

### 1. 🔧 CORE SCRIPTS - Production Ready

**Location:** `./` (Root directory)

| File | Size | Purpose | Usage |
|------|------|---------|-------|
| `predict_image.py` | ~8 KB | Phát hiện smoking trong ảnh | `python predict_image.py --image test.jpg` |
| `predict_video.py` | ~12 KB | Phát hiện smoking trong video | `python predict_video.py --video test.mp4` |
| `predict_camera.py` | ~9 KB | Phát hiện real-time từ camera | `python predict_camera.py --camera 0` |
| `smoking_detector.py` | ~6 KB | Core detector class | Imported by predict scripts |
| `cigarette_filter.py` | ~3 KB | Smoking detection logic | Filters cigarette-person pairs |
| `optimize_dataset_v6.py` | ~4 KB | Dataset optimization tool | `python optimize_dataset_v6.py` |

**Dependencies:**
```
predict_*.py → smoking_detector.py → cigarette_filter.py
```

**Key Features:**
- ✅ Batch processing support
- ✅ Auto save results with timestamps
- ✅ Configurable confidence thresholds
- ✅ Background mode for video (no preview)
- ✅ Frame extraction for smoking detections

---

### 2. 🎓 TRAINING SCRIPTS

**Location:** `./` (Root directory)

| File | Status | Epochs | mAP50 | Purpose |
|------|--------|--------|-------|---------|
| `train.py` | ✅ Stable | 80 | - | Standard training template |
| `train_v6.py` | ⭐ **BEST** | 80 | 77.42% | Production model (recommended) |
| `train_v7_improved.py` | ❌ Failed | 100 | 75.65% | Aggressive augmentation (don't use) |
| `train_v8_moderate.py` | ⏸️ Testing | 50 | 72.95% | Moderate augmentation experiment |

**Usage:**
```bash
# Train best model (recommended)
python train_v6.py

# Train with custom epochs
# Edit file: epochs = 100

# Monitor training
tensorboard --logdir runs/train
```

**Training Flow:**
```
1. Load pretrained YOLOv8s weights
2. Configure hyperparameters
3. Train with early stopping
4. Save checkpoints every 10 epochs
5. Generate plots and metrics
6. Best model saved to runs/train/<name>/weights/best.pt
```

---

### 3. 📁 INPUT/OUTPUT FOLDERS

#### A. `input_data/` - Test data
```
input_data/
├── images/          # 24 test images (.jpg)
│   ├── WIN_20251005_20_31_46_Pro.jpg
│   ├── WIN_20251113_04_31_41_Pro.jpg
│   └── ...
└── videos/          # 3 test videos (.mp4)
    ├── WIN_20250718_17_07_14_Pro.mp4
    ├── WIN_20251113_04_32_53_Pro.mp4
    └── WIN_20251223_11_18_44_Pro.mp4
```

**Purpose:** Sample data for testing predictions

#### B. `results/` - Detection outputs
```
results/
├── image/           # Image detection results
│   ├── 20251223_105423_smoking_WIN_*.jpg
│   ├── 20251223_110019_non_smoking_WIN_*.jpg
│   └── ...
├── video/           # Video detection results
│   ├── 20251223_112622_smoking_*.mp4
│   ├── WIN_20251223_11_18_44_Pro_frames/  # Extracted smoking frames
│   │   ├── 20251223_112939_434_smoking_frame_0002.jpg
│   │   └── ...
│   └── ...
└── camera/          # Camera snapshots
    ├── 20251223_105244_smoking_detected.jpg
    └── ...
```

**Naming Convention:**
- Images: `{timestamp}_{smoking/non_smoking}_{original_name}.jpg`
- Videos: `{timestamp}_{smoking/non_smoking}_{original_name}.mp4`
- Frames: `{timestamp}_{ms}_smoking_frame_{num:04d}.jpg`
- Camera: `{timestamp}_smoking_detected.jpg`

#### C. `runs/` - YOLOv8 training runs
```
runs/
├── detect/          # Detection runs (validation)
│   └── val*/
└── train/           # Training runs
    ├── smoking_detection_v5_full/
    ├── smoking_detection_v6_optimized/
    ├── smoking_detection_v7_improved/
    └── smoking_detection_v8_moderate/
        ├── weights/
        │   ├── best.pt      # Best model checkpoint
        │   ├── last.pt      # Last epoch checkpoint
        │   └── epoch*.pt    # Epoch checkpoints
        ├── results.csv      # Training metrics
        ├── args.yaml        # Training config
        └── *.png            # Training plots
```

**Important Files:**
- `weights/best.pt` - Best model (lowest validation loss)
- `results.csv` - Epoch-by-epoch metrics
- `confusion_matrix.png` - Class confusion
- `results.png` - Training curves

---

### 4. 🏆 MODELS & WEIGHTS

#### A. `ketquatrain/` - Organized training results
```
ketquatrain/
├── v5_full/         # Baseline model
│   ├── weights/
│   │   ├── best.pt  (mAP50: 75.96%)
│   │   └── last.pt
│   ├── plots/
│   ├── results.csv
│   ├── args.yaml
│   └── MODEL_INFO.md
│
├── v6_optimized/    # ⭐ BEST MODEL
│   ├── weights/
│   │   ├── best.pt  (mAP50: 77.42%) ⭐
│   │   └── last.pt
│   ├── plots/
│   ├── results.csv
│   ├── args.yaml
│   └── MODEL_INFO.md
│
├── v7_improved/     # Failed experiment
│   ├── weights/
│   │   └── best.pt  (mAP50: 75.65%)
│   ├── results.csv
│   └── MODEL_INFO.md
│
├── README.md        # Models overview
└── BAO_CAO_TONG_KET_TRAINING.md  # Training summary
```

**Model Comparison:**
| Model | mAP50 | Precision | Recall | Status |
|-------|-------|-----------|--------|--------|
| v5_full | 75.96% | 87.67% | 70.64% | ✅ Baseline |
| **v6_optimized** | **77.42%** | **87.62%** | **73.93%** | ⭐ **BEST** |
| v7_improved | 75.65% | 85.88% | 70.46% | ❌ Failed |

#### B. Pretrained weights
```
yolov8s.pt   # YOLOv8 small (11.1M params, COCO pretrained)
yolo11n.pt   # YOLO11 nano (2.6M params, latest version)
```

---

### 5. 📚 DOCUMENTATION

#### A. `BAO_CAO_FINAL/` - Complete project report

**Structure:**
```
BAO_CAO_FINAL/
├── README.md              # ⭐ Main comprehensive report (13K tokens)
├── INDEX.md               # Navigation guide
├── CHECKLIST.md           # Submission checklist
│
├── 1_TONG_QUAN/           # Overview & Analysis
│   ├── BAO_CAO_TONG_KET_TRAINING.md         # Training summary
│   ├── PHAN_TICH_CHI_TIET_CAC_MODEL.md      # ⭐ Detailed model analysis (35K tokens)
│   ├── MODEL_GUIDE.md                       # Model comparison
│   └── TRAINING_OPTIMIZATION_SUMMARY.md     # Optimization guide
│
├── 2_TRAINING_SCRIPTS/    # Training code & docs
│   ├── README.md          # Training scripts documentation
│   ├── train.py           # Standard training
│   ├── train_v8_moderate.py  # Latest experiment
│   ├── smoking_detector.py   # Core module
│   └── cigarette_filter.py   # Filter module
│
├── 3_PREDICTION_SCRIPTS/  # Prediction code & docs
│   ├── README.md          # Prediction scripts documentation
│   ├── predict_image.py   # Image detection
│   ├── predict_video.py   # Video detection
│   ├── predict_camera.py  # Camera detection
│   ├── smoking_detector.py   # Core module
│   └── cigarette_filter.py   # Filter module
│
├── 4_TRAINING_RESULTS/    # Training results (manual copy needed)
│   └── (Copy from runs/train/ and ketquatrain/)
│
└── 5_HUONG_DAN/           # Usage guides
    └── HUONG_DAN_SU_DUNG.md  # ⭐ Comprehensive usage guide (5.8K tokens)
```

**Key Documents:**

1. **README.md** (Main Report)
   - 8 sections: Overview, Architecture, Dataset, Training, Results, Usage, Structure, Conclusions
   - ~13,000 tokens
   - Perfect for academic report

2. **PHAN_TICH_CHI_TIET_CAC_MODEL.md** (Model Analysis)
   - Detailed analysis of 4 models (v5, v6, v7, v8)
   - Pre-training vs Post-training comparison
   - Hyperparameter analysis
   - ~35,000 tokens

3. **HUONG_DAN_SU_DUNG.md** (Usage Guide)
   - Installation, Training, Prediction
   - Troubleshooting, Tips & Best Practices
   - Advanced usage
   - ~5,800 tokens

**Usage:**
- For report: Start with `BAO_CAO_FINAL/README.md`
- For model details: Read `1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md`
- For usage: Read `5_HUONG_DAN/HUONG_DAN_SU_DUNG.md`
- For navigation: Check `INDEX.md`

#### B. Root documentation files
```
README.md                          # Quick start guide
PROJECT_README.md                  # ⭐ Complete project README
DIRECTORY_STRUCTURE.md             # This file
MODEL_GUIDE.md                     # Model versions comparison
PATH_STRUCTURE.md                  # Path organization
DATA_SPLITS_IMPACT_GUIDE.md        # Dataset splitting guide
GOOGLE_COLAB_TRAINING_GUIDE.md     # Training on Colab
```

---

### 6. 📋 CONFIGURATION FILES

```
requirements.txt       # Python dependencies
.gitignore            # Git ignore rules
```

**requirements.txt:**
```txt
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
```

**.gitignore:**
```
__pycache__/
*.pyc
*.pyo
*.pt (except yolov8s.pt, yolo11n.pt)
runs/
results/
venv/
.vscode/
```

---

## 🎯 WORKFLOW & DATA FLOW

### 1. Training Workflow
```
┌─────────────────┐
│ Load pretrained │
│   yolov8s.pt    │
└────────┬────────┘
         │
┌────────▼────────┐
│ Configure       │
│ hyperparameters │
└────────┬────────┘
         │
┌────────▼────────┐
│ Train on        │
│ dataset v6      │
└────────┬────────┘
         │
┌────────▼────────┐
│ Save checkpoints│
│ every 10 epochs │
└────────┬────────┘
         │
┌────────▼────────┐
│ Best model →    │
│ runs/train/*/   │
│ weights/best.pt │
└────────┬────────┘
         │
┌────────▼────────┐
│ Copy to         │
│ ketquatrain/v*/ │
└─────────────────┘
```

### 2. Prediction Workflow
```
┌─────────────────┐
│ Load best model │
│ v6_optimized    │
└────────┬────────┘
         │
┌────────▼────────┐
│ Input:          │
│ image/video/cam │
└────────┬────────┘
         │
┌────────▼────────┐
│ YOLOv8 detect   │
│ Cigarette +     │
│ Person          │
└────────┬────────┘
         │
┌────────▼────────┐
│ Filter smoking  │
│ (distance check)│
└────────┬────────┘
         │
┌────────▼────────┐
│ Save result to  │
│ results/*/      │
└─────────────────┘
```

### 3. Documentation Workflow
```
Training → runs/train/* 
         ↓
Copy → ketquatrain/v*
         ↓
Analyze → BAO_CAO_FINAL/1_TONG_QUAN/
         ↓
Document → BAO_CAO_FINAL/README.md
         ↓
Package → BAO_CAO_FINAL/ (complete)
```

---

## 📊 FILE SIZES & STATISTICS

### Storage Usage

| Category | Size | Files | Description |
|----------|------|-------|-------------|
| Models (weights) | ~46 MB | 8 | All .pt files |
| Training results | ~500 MB | ~200 | runs/train/* |
| Documentation | ~2 MB | 20+ | Markdown files |
| Input data | ~50 MB | ~30 | Test images/videos |
| Output results | ~100 MB | ~300 | Predictions |
| Code | ~200 KB | 10+ | Python scripts |

**Total Project Size:** ~700 MB (with all training runs)

**Minimal Size:** ~50 MB (code + best model + docs only)

### File Counts

```
Python scripts:    10 files
Documentation:     20+ files
Model weights:     8 files (.pt)
Training runs:     4 folders
Test data:         27 files (images + videos)
Result files:      300+ files
```

---

## 🔍 IMPORTANT PATHS

### Production Use

**Best Model:**
```
ketquatrain/v6_optimized/weights/best.pt
```

**Prediction Scripts:**
```
predict_image.py
predict_video.py
predict_camera.py
```

**Core Modules:**
```
smoking_detector.py
cigarette_filter.py
```

### Documentation

**Main Report:**
```
BAO_CAO_FINAL/README.md
```

**Model Analysis:**
```
BAO_CAO_FINAL/1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md
```

**Usage Guide:**
```
BAO_CAO_FINAL/5_HUONG_DAN/HUONG_DAN_SU_DUNG.md
```

### Configuration

**Training Config:**
```
ketquatrain/v6_optimized/args.yaml
```

**Dependencies:**
```
requirements.txt
```

---

## 🚀 QUICK NAVIGATION

### For Users (Prediction)
```
1. Read: PROJECT_README.md (this file)
2. Check: requirements.txt
3. Use: predict_*.py scripts
4. Results: results/*/
```

### For Developers (Training)
```
1. Read: BAO_CAO_FINAL/5_HUONG_DAN/HUONG_DAN_SU_DUNG.md
2. Check: train_v6.py (best training script)
3. Monitor: runs/train/*/results.csv
4. Weights: runs/train/*/weights/best.pt
```

### For Reviewers (Report)
```
1. Start: BAO_CAO_FINAL/README.md (overview)
2. Details: BAO_CAO_FINAL/1_TONG_QUAN/PHAN_TICH_CHI_TIET_CAC_MODEL.md
3. Results: ketquatrain/v6_optimized/
4. Usage: BAO_CAO_FINAL/5_HUONG_DAN/HUONG_DAN_SU_DUNG.md
```

---

## 📝 NOTES

### Folder Naming Conventions

- **Lowercase + underscore:** Python modules (`smoking_detector.py`)
- **Timestamp prefix:** Results files (`20251223_105423_*.jpg`)
- **Version suffix:** Models (`v5_full`, `v6_optimized`)
- **Uppercase:** Documentation folders (`BAO_CAO_FINAL`)

### File Naming Patterns

**Results:**
- `{timestamp}_{status}_{original}.{ext}`
- Example: `20251223_110019_smoking_WIN_20251113_04_32_10_Pro.jpg`

**Frames:**
- `{timestamp}_{ms}_smoking_frame_{num:04d}.jpg`
- Example: `20251223_112939_434_smoking_frame_0002.jpg`

**Models:**
- `smoking_detection_v{n}_{variant}/weights/best.pt`
- Example: `smoking_detection_v6_optimized/weights/best.pt`

### Cleanup Tips

**Clean training cache:**
```bash
Remove-Item -Recurse -Force runs/train/smoking_detection_v*
# Keep only final results in ketquatrain/
```

**Clean prediction results:**
```bash
Remove-Item -Recurse -Force results/image/*
Remove-Item -Recurse -Force results/video/*
Remove-Item -Recurse -Force results/camera/*
```

**Clean Python cache:**
```bash
Remove-Item -Recurse -Force __pycache__
Remove-Item -Force *.pyc
```

---

## ✅ CHECKLIST

### Before Training
- [ ] Dataset prepared in `dataset/smoking_train_image_v6/`
- [ ] Pretrained weights downloaded (`yolov8s.pt`)
- [ ] GPU available (check `nvidia-smi`)
- [ ] Enough disk space (≥5GB)

### Before Prediction
- [ ] Best model exists (`ketquatrain/v6_optimized/weights/best.pt`)
- [ ] Input data in `input_data/images/` or `input_data/videos/`
- [ ] Results folder created (`results/`)

### Before Submission
- [ ] Training results copied to `BAO_CAO_FINAL/4_TRAINING_RESULTS/`
- [ ] Documentation complete in `BAO_CAO_FINAL/`
- [ ] All scripts tested and working
- [ ] README files updated

---

**Document Version:** 1.0  
**Last Updated:** 23/12/2025  
**Maintainer:** Project Team  
**Status:** Complete ✅
