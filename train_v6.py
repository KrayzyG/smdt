"""
Training Script v6 - Optimized for Smoking Detection
Improvements:
- Dataset v6: 85/10/5 split, filtered hard negatives
- Enhanced augmentation for small objects
- Focal loss weights optimized
- Multi-scale training
- Longer training schedule
"""

import torch
from ultralytics import YOLO
import os
from pathlib import Path

def main():
    print("="*60)
    print("🚀 TRAINING v6 - OPTIMIZED CONFIGURATION")
    print("="*60)
    
    # Check CUDA
    device = '0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"🚀 Sử dụng device: cuda")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA không khả dụng, sử dụng CPU")
    
    # Dataset path
    data_yaml = r"e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\dataset\smoking_train_image_v6\data.yaml"
    
    if not os.path.exists(data_yaml):
        print(f"❌ Không tìm thấy data.yaml: {data_yaml}")
        return
    
    print(f"📂 Dataset: {data_yaml}")
    
    # Load model from scratch
    print(f"🚀 Training from scratch với yolov8s.pt (COCO pretrained)")
    model = YOLO('yolov8s.pt')
    
    # Training với config tối ưu
    print("\n" + "="*60)
    print("⚙️  TRAINING CONFIGURATION v6:")
    print("="*60)
    print("✅ Dataset v6: 85/10/5 split (8,844/1,040/521)")
    print("✅ Filtered: 705 hard negatives removed")
    print("✅ Epochs: 80 (stable training)")
    print("✅ Learning rate: 0.012 → 0.000012 (balanced)")
    print("✅ Batch size: 14 (same as v5, stable)")
    print("✅ Patience: 25 (early stopping)")
    print("❌ Multi-scale: OFF (stable training)")
    print("✅ Balanced augmentation: copy_paste=0.35, scale=0.6, mixup=0.2")
    print("✅ Focal loss weights: box=10.0, cls=2.5, dfl=2.0")
    print("="*60 + "\n")
    
    results = model.train(
        # Dataset
        data=data_yaml,
        
        # Training schedule
        epochs=80,                   # ✅ Giống v5 (stable)
        patience=25,                 # ✅ Cao hơn v5 một chút (20 → 25)
        
        # Image settings
        imgsz=640,
        multi_scale=False,           # ❌ TẮT multi-scale (stable training)
        rect=False,                  # Shuffle enabled
        
        # Batch & workers
        batch=14,                    # ✅ Batch 14 như v5 (stable, tested)
        workers=8,
        device=device,
        
        # Learning rate
        lr0=0.012,                   # ✅ Cao hơn v5 (0.01) nhưng không quá (0.012)
        lrf=0.001,                   # ✅ final_lr = 0.000012 (vừa phải)
        warmup_epochs=5,             # ✅ Giống v5 (stable)
        warmup_bias_lr=0.1,
        warmup_momentum=0.8,
        
        # Optimizer
        optimizer='AdamW',
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights - ✅ FOCAL LOSS cho small objects
        box=10.0,                    # ✅ Tăng từ 7.5 → 10.0
        cls=2.5,                     # ✅ Tăng từ 2.0 → 2.5
        dfl=2.0,                     # ✅ Tăng từ 1.5 → 2.0
        
        # Augmentation - ✅ ENHANCED cho cigarette nhỏ
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.6,                   # ✅ Thấp hơn v5 (0.8) cho small objects nhưng không quá aggressive
        shear=2,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,                   # ✅ Giống v5 (stable augmentation)
        copy_paste=0.35,             # ✅ Cao hơn v5 (0.3) nhưng không quá (0.35)
        auto_augment='randaugment',
        erasing=0.4,
        
        # Validation & saving
        val=True,
        save=True,
        save_period=10,              # Save mỗi 10 epochs
        
        # Output
        project='runs/train',
        name='smoking_detection_v6_optimized',
        exist_ok=True,
        
        # Other settings
        pretrained=True,             # COCO pretrained
        cache=False,                 # Tắt để tránh OOM
        
        # Visualization
        plots=True,
    )
    
    print("\n" + "="*60)
    print("✅ Training hoàn tất!")
    print("="*60)
    print(f"📁 Kết quả lưu tại: runs/train/smoking_detection_v6_optimized")
    print(f"🏆 Best model: runs/train/smoking_detection_v6_optimized/weights/best.pt")
    print(f"📊 Results CSV: runs/train/smoking_detection_v6_optimized/results.csv")
    print("\n🎯 OPTIMIZATIONS v6:")
    print("   ✅ Dataset: 85/10/5 split, filtered 705 hard negatives")
    print("   ✅ Training: 80 epochs, lr=0.012, batch=14")
    print("   ✅ Augmentation: copy_paste=0.35, scale=0.6, mixup=0.2")
    print("   ✅ Focal loss: box=10.0, cls=2.5, dfl=2.0")
    print("   ❌ Multi-scale: disabled (VRAM limit 4GB)")
    print("\n🔥 KỲ VỌNG:")
    print("   🎯 Target mAP50: 80-83% (cao hơn v5: 75.96%)")
    print("   🎯 Target Recall: ≥75% (cao hơn v5: 70.64%)")
    print("   🎯 Better small object detection với focal loss")
    print("   🎯 Training time: ~4-5 giờ (80 epochs)")
    print("\n📊 So sánh:")
    print("   v5_full: mAP50 = 75.96%")
    print("   v6_optimized: Expected 80-83% (+4-7%)")
    print("="*60)

if __name__ == "__main__":
    main()
