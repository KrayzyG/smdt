"""
Script huấn luyện YOLOv8 cho phát hiện Smoking Detection
Dataset: Roboflow smoking-tasfx v4 (đã làm sạch)
Classes: Cigarette (0), Person (1)

Logic phát hiện smoking (post-processing):
- Phát hiện Person và Cigarette
- Tính khoảng cách từ Cigarette đến vùng ĐẦU của Person
- Nếu Cigarette gần đầu Person → SMOKING
- Ngược lại → NON-SMOKING
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def main():
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Sử dụng device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ✅ TRAIN FROM SCRATCH: Dùng yolov8s.pt (COCO pretrained)
    # Fine-tune v3 không hiệu quả do dataset khác nhau quá nhiều
    # Train from scratch cho dataset 80/10/10 mới sẽ tốt hơn
    print(f"🚀 Training from scratch với yolov8s.pt (COCO pretrained)")
    model = YOLO('yolov8s.pt')
    
    # ✅ IMPROVED: Dataset đã được re-split (80/10/10) cho validation đáng tin cậy hơn
    # Old: 11,910 train / 312 val / 122 test (96.5/2.5/1.0) - val set quá nhỏ
    # New: 9,875 train / 1,234 val / 1,235 test (80/10/10) - balanced split ✅
    data_yaml = r"e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\dataset\smoking_train_image_improved\data.yaml"
    
    # Kiểm tra file tồn tại
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"❌ Không tìm thấy data.yaml tại: {data_yaml}")
    
    # Validate dataset directories
    dataset_root = Path(data_yaml).parent
    train_dir = dataset_root / 'train' / 'images'
    valid_dir = dataset_root / 'valid' / 'images'
    test_dir = dataset_root / 'test' / 'images'
    
    print(f"📂 Dataset validation:")
    print(f"   data.yaml: {data_yaml}")
    print(f"   Train images: {len(list(train_dir.glob('*')))} ({'✅' if train_dir.exists() else '❌'})")
    print(f"   Valid images: {len(list(valid_dir.glob('*')))} ({'✅' if valid_dir.exists() else '❌'})")
    print(f"   Test images: {len(list(test_dir.glob('*')))} ({'✅' if test_dir.exists() else '❌'})")
    
    # ✅ FULL TRAINING CONFIG: Train đầy đủ từ đầu
    results = model.train(
        data=data_yaml,              # File cấu hình dataset
        epochs=80,                   # ✅ FULL: 80 epochs cho model học đầy đủ hơn
        imgsz=640,                   # Kích thước ảnh input
        batch=14,                    # ✅ RTX 3050Ti 4GB optimal
        device=device,               # GPU hoặc CPU
        workers=8,                   # ✅ Tăng từ 4 → 8 workers (faster data loading)
        patience=20,                 # ✅ FULL: 20 patience (không converge quá sớm)
        save=True,                   # Lưu checkpoint
        save_period=10,              # ✅ FULL: Lưu mỗi 10 epochs (total 80 epochs)
        project='runs/train',        # Thư mục lưu kết quả
        name='smoking_detection_v5_full',  # ✅ v5: Train from scratch dataset 80/10/10
        exist_ok=True,               # Ghi đè để tiếp tục train
        pretrained=True,             # ✅ FULL: Dùng COCO pretrained weights
        cache=False,                 # ❌ DISABLED: Tắt để tránh OOM (stable training)
        optimizer='AdamW',           # ✅ AdamW tốt hơn Adam (weight decay separation)
        lr0=0.01,                    # ✅ FULL: Learning rate chuẩn cho full training
        lrf=0.001,                   # ✅ FULL: final_lr = 0.01 * 0.001 = 0.00001
        momentum=0.937,              # Momentum
        weight_decay=0.0005,         # Weight decay
        warmup_epochs=5,             # ✅ FULL: Warmup đầy đủ cho stable start
        warmup_momentum=0.8,         # Warmup momentum
        warmup_bias_lr=0.1,          # ✅ FULL: Warmup bias LR chuẩn
        box=7.5,                     # ✅ Giảm từ 10.0 → 7.5 (balance với cls loss)
        cls=2.0,                     # ✅ CRITICAL: Tăng từ 0.5 → 2.0 (cigarette cần học class tốt hơn!)
        dfl=1.5,                     # DFL loss gain
        # ✅ NEW: Class weights để cân bằng Cigarette vs Person
        # Format: [weight_class_0, weight_class_1] = [Cigarette, Person]
        # Cigarette khó hơn → weight cao hơn
        # Note: YOLOv8 không có tham số class_weights trực tiếp, phải tune qua cls loss
        # ✅ IMPROVED: Data Augmentation tối ưu cho small objects (cigarette)
        hsv_h=0.015,                 # ✅ Giảm từ 0.02 → 0.015 (màu cigarette quan trọng)
        hsv_s=0.7,                   # ✅ Giảm từ 0.8 → 0.7 (giữ màu cigarette realistic)
        hsv_v=0.4,                   # ✅ Giảm từ 0.5 → 0.4 (brightness quan trọng)
        degrees=10,                  # ✅ Giảm từ 15 → 10 (cigarette nhỏ, rotate nhiều mất shape)
        translate=0.1,               # ✅ Giảm từ 0.2 → 0.1 (giữ cigarette trong frame)
        scale=0.8,                   # ✅ CRITICAL: Tăng từ 0.6 → 0.8 (KHÔNG scale down quá → cigarette mất)
        shear=2,                     # ✅ Giảm từ 5 → 2 (shear nhiều → cigarette bị méo)
        perspective=0.0005,          # ✅ Giảm từ 0.001 → 0.0005 (perspective ít ảnh hưởng)
        flipud=0.0,                  # ✅ TẮT vertical flip (cigarette không đảo ngược)
        fliplr=0.5,                  # Horizontal flip (OK cho cigarette)
        mosaic=1.0,                  # Mosaic augmentation (tốt cho small objects)
        mixup=0.2,                   # ✅ Tăng từ 0.15 → 0.2 (tạo thêm challenging examples)
        copy_paste=0.3,              # ✅ CRITICAL: Tăng từ 0.1 → 0.3 (copy cigarette vào nhiều scenes)
        # ✅ NEW: Multi-scale training cho small objects
        rect=False,                  # Không dùng rectangular training (dùng square để giữ cigarette)
        close_mosaic=10,             # Tắt mosaic 10 epochs cuối để fine-tune
        # ✅ NEW: Label smoothing
        label_smoothing=0.1,         # Label smoothing giúp generalization tốt hơn
        plots=True,                  # Lưu plots kết quả
        verbose=True                 # Hiển thị chi tiết
    )
    
    print("\n" + "="*60)
    print("✅ Training hoàn tất!")
    print("="*60)
    print(f"📁 Kết quả lưu tại: runs/train/smoking_detection_v5_full")
    print(f"🏆 Best model: runs/train/smoking_detection_v5_full/weights/best.pt")
    print(f"📊 Results CSV: runs/train/smoking_detection_v5_full/results.csv")
    print("\n🎯 FULL TRAINING v5:")
    print("   📊 Base model: yolov8s.pt (COCO pretrained)")
    print("   ✅ Dataset: 80/10/10 split (9,875/1,234/1,235)")
    print("   ✅ Epochs: 80 (full training extended)")
    print("   ✅ Learning rate: 0.01 (chuẩn)")
    print("   ✅ Patience: 20 (early stopping)")
    print("   ✅ Config: Optimized cho small objects (cigarette)")
    print("\n🔥 KỲ VỌNG:")
    print("   🎯 Target mAP50: 83-86% (bằng hoặc cao hơn v3)")
    print("   🎯 Better validation: Val set 4x lớn hơn → reliable metrics")
    print("   🎯 Training time: ~4.5-5 giờ (80 epochs, cache=False)")
    print("\n📊 So sánh với v4 fine-tune:")
    print("   v4 fine-tune: mAP50 = 77.32% (không tốt)")
    print("   v5 from scratch: Expected 83-86% (tốt hơn nhiều)")
    print("="*60)

if __name__ == '__main__':
    main()
