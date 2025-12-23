"""
Training v8_moderate: MODERATE AUGMENTATION
Giải quyết vấn đề v7 aggressive aug failed
Strategy: Tăng augmentation VỪA PHẢI từ v6 để tăng Recall
Target: mAP50 79-80%, Recall 76-78%
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path
from datetime import datetime

def train_v8_moderate():
    """
    Training với moderate augmentation
    Cân bằng giữa v6 (baseline) và v7 (quá mạnh)
    """
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{'='*70}")
    print("🚀 SMOKING DETECTION - TRAINING v8_moderate")
    print(f"{'='*70}\n")
    print(f"📍 Device: {device}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {vram_total:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        print(f"   ✅ GPU cache cleared")
    
    # Model
    model_path = 'yolov8s.pt'
    print(f"\n🎯 Model: {model_path} (COCO pretrained)")
    print(f"   Strategy: Train from scratch with moderate augmentation")
    
    # Dataset
    script_dir = Path(__file__).parent.parent
    data_yaml = script_dir / 'dataset' / 'smoking_train_image_v6' / 'data.yaml'
    
    print(f"\n📂 Dataset validation:")
    print(f"   Root: {data_yaml.parent}")
    
    # Validate dataset structure
    train_img_dir = data_yaml.parent / 'train' / 'images'
    val_img_dir = data_yaml.parent / 'val' / 'images'
    test_img_dir = data_yaml.parent / 'test' / 'images'
    
    if not train_img_dir.exists():
        print(f"❌ Train images not found: {train_img_dir}")
        return
    
    if not val_img_dir.exists():
        print(f"❌ Val images not found: {val_img_dir}")
        return
    
    train_count = len(list(train_img_dir.glob('*')))
    val_count = len(list(val_img_dir.glob('*')))
    test_count = len(list(test_img_dir.glob('*'))) if test_img_dir.exists() else 0
    total_count = train_count + val_count + test_count
    
    print(f"   Train: {train_count:,} images ({train_count/total_count*100:.1f}%) ✅")
    print(f"   Val:   {val_count:,} images ({val_count/total_count*100:.1f}%) ✅")
    print(f"   Test:  {test_count:,} images ({test_count/total_count*100:.1f}%) ✅")
    print(f"   Total: {total_count:,} images")
    
    # Training config
    print(f"\n⚙️ Training Configuration (v8_moderate):\n")
    
    print(f"   📊 BASIC SETTINGS:")
    print(f"      epochs: 50 (reduced for faster iteration)")
    print(f"      batch: 12 (moderate aug, safe for 4GB VRAM)")
    print(f"      imgsz: 640")
    print(f"      patience: 25")
    print(f"      close_mosaic: 10")
    
    print(f"\n   🎓 OPTIMIZER & LEARNING RATE:")
    print(f"      optimizer: AdamW")
    print(f"      lr0: 0.013 (v6: 0.012, v7: 0.015) 🔥 Tăng nhẹ")
    print(f"      lrf: 0.0005 (v6: 0.001, v7: 0.0001) 🔥 Giữa v6 và v7")
    print(f"      cos_lr: True 🔥")
    print(f"      warmup_epochs: 6 (v6: 5, v7: 8)")
    print(f"      warmup_momentum: 0.8")
    print(f"      momentum: 0.937")
    print(f"      weight_decay: 0.0005")
    
    print(f"\n   ⚖️ LOSS WEIGHTS (BALANCED FOR RECALL):")
    print(f"      box: 11.0 (v6: 10.0, v7: 12.0) 🔥 Tăng nhẹ")
    print(f"      cls: 2.2 (v6: 2.5, v7: 2.0) 🔥 Giảm nhẹ → Tăng Recall")
    print(f"      dfl: 2.3 (v6: 2.0, v7: 2.5) 🔥 Tăng nhẹ cho small objects")
    print(f"      → Strategy: Cân bằng Detection và Classification")
    
    print(f"\n   🎨 AUGMENTATION (MODERATE - VỪA PHẢI):")
    print(f"      scale: 0.55 (v6: 0.6, v7: 0.5) 🔥")
    print(f"      copy_paste: 0.4 (v6: 0.35, v7: 0.5) 🔥")
    print(f"      mixup: 0.22 (v6: 0.2, v7: 0.25) 🔥")
    print(f"      translate: 0.15 (v6: 0.1, v7: 0.2) 🔥")
    print(f"      degrees: 12 (v6: 10, v7: 15) 🔥")
    print(f"      shear: 2.5 (v6: 2, v7: 3)")
    print(f"      mosaic: 1.0")
    print(f"      flipud: 0.0")
    print(f"      fliplr: 0.5")
    print(f"      hsv_h: 0.018 (v6: 0.015, v7: 0.02)")
    print(f"      hsv_s: 0.75 (v6: 0.7, v7: 0.8)")
    print(f"      hsv_v: 0.45 (v6: 0.4, v7: 0.5)")
    
    print(f"\n   📈 MODERATE AUGMENTATION RATIONALE:")
    print(f"      • Tăng nhẹ từ v6 để expose thêm edge cases")
    print(f"      • KHÔNG quá mạnh như v7 (tránh overfitting)")
    print(f"      • copy_paste=0.4: Thêm cigarette instances nhưng không quá nhiều")
    print(f"      • mixup=0.22: Hard negatives vừa phải")
    print(f"      • scale=0.55: Cigarettes nhỏ hơn nhưng không quá nhỏ")
    
    print(f"\n🎯 EXPECTED RESULTS:")
    print(f"   Baseline (v6): mAP50=77.42%, P=87.08%, R=73.58%")
    print(f"   Failed (v7):   mAP50=75.65%, P=84.15%, R=72.12%")
    print(f"   Target (v8):   mAP50≥79%, P=85-87%, R≥76-78%")
    print(f"   Strategy:      MODERATE increase in Recall, maintain Precision")
    print(f"   Success rate:  70-80% (moderate risk)")
    
    # Auto-start training
    print(f"\n{'='*70}")
    print("🔥 STARTING TRAINING v8_moderate")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Estimated time: 2.5-3 hours (50 epochs)")
    print(f"{'='*70}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Train model
    results = model.train(
        # Data
        data=str(data_yaml),
        
        # Basic settings
        epochs=50,
        batch=12,              # Moderate aug, safe for 4GB
        imgsz=640,
        patience=25,
        close_mosaic=10,
        
        # Device
        device=device,
        workers=8,
        
        # Optimizer & LR schedule
        optimizer='AdamW',
        lr0=0.013,             # v6: 0.012, v7: 0.015
        lrf=0.0005,            # v6: 0.001, v7: 0.0001
        cos_lr=True,
        warmup_epochs=6,       # v6: 5, v7: 8
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights
        box=11.0,              # v6: 10.0, v7: 12.0
        cls=2.2,               # v6: 2.5, v7: 2.0 - Lower for recall
        dfl=2.3,               # v6: 2.0, v7: 2.5
        
        # Augmentation (MODERATE)
        scale=0.55,            # v6: 0.6, v7: 0.5
        copy_paste=0.4,        # v6: 0.35, v7: 0.5
        mixup=0.22,            # v6: 0.2, v7: 0.25
        translate=0.15,        # v6: 0.1, v7: 0.2
        degrees=12,            # v6: 10, v7: 15
        shear=2.5,             # v6: 2, v7: 3
        mosaic=1.0,
        flipud=0.0,
        fliplr=0.5,
        perspective=0.0005,
        hsv_h=0.018,           # v6: 0.015, v7: 0.02
        hsv_s=0.75,            # v6: 0.7, v7: 0.8
        hsv_v=0.45,            # v6: 0.4, v7: 0.5
        
        # Other settings
        amp=True,
        deterministic=True,
        seed=0,
        
        # Output
        project='runs/train',
        name='smoking_detection_v8_moderate',
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
        val=True
    )
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETED!")
    print(f"{'='*70}")
    
    # Print results
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Best model: runs/train/smoking_detection_v8_moderate/weights/best.pt")
    print(f"   Last model: runs/train/smoking_detection_v8_moderate/weights/last.pt")
    print(f"   Results CSV: runs/train/smoking_detection_v8_moderate/results.csv")
    print(f"   Plots: runs/train/smoking_detection_v8_moderate/*.png")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Check results: code runs/train/smoking_detection_v8_moderate/results.csv")
    print(f"   2. Compare with v6: python check_v8_results.py")
    print(f"   3. If SUCCESS (mAP≥79%, R≥76%):")
    print(f"      → Backup to ketquatrain/v8_moderate/")
    print(f"      → Test model: python predict_image.py --model runs/train/smoking_detection_v8_moderate/weights/best.pt")
    print(f"   4. If FAILED:")
    print(f"      → Try GIẢI PHÁP 2: YOLOv8m (model lớn hơn)")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    train_v8_moderate()
