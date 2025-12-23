"""
Script huấn luyện YOLOv8 Version 7 - IMPROVED
==============================================

Baseline: v6_optimized (mAP50: 77.27%, Recall: 70.64%)
Target: mAP50 ≥ 79%, Recall ≥ 74%

KEY IMPROVEMENTS (Mức 1):
- Loss weights: box=12.0 (+20%), dfl=2.5 (+25%)
- Augmentation: scale=0.5, copy_paste=0.4, translate=0.15
- LR schedule: Cosine LR, lr0=0.015, lrf=0.0001
- Training: 100 epochs, warmup=8

Expected improvement: +2-3% mAP, +3-5% Recall
Training time: ~4-5 giờ (RTX 3050Ti 4GB)
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path
from datetime import datetime

def main():
    print("="*70)
    print("🚀 SMOKING DETECTION - TRAINING v7_IMPROVED")
    print("="*70)
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📍 Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
        print(f"   ✅ GPU cache cleared")
    
    # Model: Train from scratch với YOLOv8s
    print(f"\n🎯 Model: yolov8s.pt (COCO pretrained)")
    print(f"   Strategy: Train from scratch")
    model = YOLO('yolov8s.pt')
    
    # Dataset path
    data_yaml = r"e:\LEARN\@ki1 nam 4\MACHINE LEARNING\smoke\wsf1\dataset\smoking_train_image_v6\data.yaml"
    
    # Validate dataset
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"❌ Không tìm thấy data.yaml: {data_yaml}")
    
    dataset_root = Path(data_yaml).parent
    train_dir = dataset_root / 'train' / 'images'
    val_dir = dataset_root / 'val' / 'images'  # Fixed: 'val' not 'valid'
    test_dir = dataset_root / 'test' / 'images'
    
    print(f"\n📂 Dataset validation:")
    print(f"   Root: {dataset_root}")
    train_count = len(list(train_dir.glob('*'))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob('*'))) if val_dir.exists() else 0
    test_count = len(list(test_dir.glob('*'))) if test_dir.exists() else 0
    total_count = train_count + val_count + test_count
    
    print(f"   Train: {train_count:,} images ({train_count/total_count*100:.1f}%) {'✅' if train_dir.exists() else '❌'}")
    print(f"   Val:   {val_count:,} images ({val_count/total_count*100:.1f}%) {'✅' if val_dir.exists() else '❌'}")
    print(f"   Test:  {test_count:,} images ({test_count/total_count*100:.1f}%) {'✅' if test_dir.exists() else '❌'}")
    print(f"   Total: {total_count:,} images")
    
    if train_count == 0 or val_count == 0:
        raise ValueError(f"❌ Dataset không hợp lệ! Train: {train_count}, Val: {val_count}")
    
    # Verify split ratio
    expected_train_pct = 80.0
    expected_val_pct = 10.0
    actual_train_pct = train_count/total_count*100
    actual_val_pct = val_count/total_count*100
    
    if abs(actual_train_pct - expected_train_pct) > 5:
        print(f"   ⚠️ WARNING: Train split {actual_train_pct:.1f}% != expected {expected_train_pct}%")
    if abs(actual_val_pct - expected_val_pct) > 2:
        print(f"   ⚠️ WARNING: Val split {actual_val_pct:.1f}% != expected {expected_val_pct}%")
    
    # Training configuration
    print(f"\n⚙️ Training Configuration (v7_improved):")
    print(f"\n   📊 BASIC SETTINGS:")
    print(f"      epochs: 100 (v6: 80, +25%)")
    print(f"      batch: 10 (Aggressive aug optimized) ⚠️ Reduced for stability")
    print(f"      imgsz: 640")
    print(f"      patience: 30 (v6: 25, +20%)")
    print(f"      close_mosaic: 10 (last 10 epochs)")
    print(f"      Note: Batch reduced 14→10 due to aggressive augmentation memory")
    
    print(f"\n   🎓 OPTIMIZER & LEARNING RATE:")
    print(f"      optimizer: AdamW")
    print(f"      lr0: 0.015 (v6: 0.012, +25%) 🔥")
    print(f"      lrf: 0.0001 (v6: 0.001, -90%) 🔥")
    print(f"      cos_lr: True (v6: False) 🔥 NEW!")
    print(f"      warmup_epochs: 8 (v6: 5, +60%)")
    print(f"      warmup_momentum: 0.8")
    print(f"      momentum: 0.937")
    print(f"      weight_decay: 0.0005")
    
    print(f"\n   ⚖️ LOSS WEIGHTS (OPTIMIZED FOR CIGARETTE DETECTION):")
    print(f"      box: 12.0 (v6: 10.0, +20%) 🔥🔥 Localization")
    print(f"      cls: 2.0 (v6: 2.5, -20%) 🔥 DETECT > Classify")
    print(f"      dfl: 2.5 (v6: 2.0, +25%) 🔥🔥 Small objects")
    print(f"      → Strategy: Maximize RECALL (detect more cigarettes)")
    print(f"      → Accept slightly lower precision for higher recall")
    
    print(f"\n   🎨 AUGMENTATION (AGGRESSIVE FOR CIGARETTE):")
    print(f"      scale: 0.5 (v6: 0.6, -17%) 🔥 Small cigarettes")
    print(f"      copy_paste: 0.5 (v6: 0.35, +43%) 🔥🔥 Max instances")
    print(f"      mixup: 0.25 (v6: 0.2, +25%) 🔥 Hard examples")
    print(f"      translate: 0.2 (v6: 0.1, +100%) 🔥 Edge cases")
    print(f"      degrees: 15 (v6: 10, +50%) 🔥 More rotations")
    print(f"      shear: 3 (v6: 2, +50%)")
    print(f"      mosaic: 1.0")
    print(f"      flipud: 0.0 (cigarettes không đảo)")
    print(f"      fliplr: 0.5")
    print(f"      hsv_h: 0.02 (v6: 0.015, +33%) 🔥 Color variation")
    print(f"      hsv_s: 0.8 (v6: 0.7, +14%) 🔥 Contrast")
    print(f"      hsv_v: 0.5 (v6: 0.4, +25%) 🔥 Lighting")
    
    print(f"\n⚠️ AGGRESSIVE AUGMENTATION WARNINGS:")
    print(f"   - scale=0.5 + copy_paste=0.5 + mixup=0.25 RẤT MẠNH!")
    print(f"   - cls=2.0 thấp → Recall cao nhưng Precision có thể giảm")
    print(f"   - Monitor first 20 epochs:")
    print(f"     • Box loss phải giảm < 1.5 sau epoch 20")
    print(f"     • Recall phải > 0.65 sau epoch 30")
    print(f"   - Nếu loss plateau → rollback copy_paste=0.4, mixup=0.2")
    print(f"   - Strategy: MAXIMIZE RECALL, accept lower Precision")
    
    print(f"\n🎯 EXPECTED RESULTS (AGGRESSIVE CONFIG):")
    print(f"   Baseline (v6): mAP50=77.27%, P=87.67%, R=70.64%")
    print(f"   Target (v7):   mAP50≥79%, P=84-86%, R≥75-77%")
    print(f"   Strategy:      PRIORITIZE RECALL (detect more cigarettes)")
    print(f"   Trade-off:     Precision may drop 1-3% for +5-7% Recall")
    print(f"   Improvement:   +2-3% mAP, +5-7% Recall 🎯")
    
    # Confirm training
    print(f"\n{'='*70}")
    
    # Check if previous training exists
    resume_path = Path('runs/train/smoking_detection_v7_improved/weights/last.pt')
    if resume_path.exists():
        print(f"⚠️ Found existing training: {resume_path}")
        user_input = input("Resume from checkpoint? (y/n, default=n): ")
        if user_input.lower() == 'y':
            print(f"✅ Resuming from {resume_path}")
            model = YOLO(str(resume_path))
            # Will use resume=True in training
        else:
            print(f"🔄 Starting fresh training (old results will be overwritten)")
    
    user_input = input("🚀 Bắt đầu training? (y/n): ")
    if user_input.lower() != 'y':
        print("❌ Training cancelled")
        return
    
    print(f"\n{'='*70}")
    print(f"🔥 STARTING TRAINING v7_improved")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Estimated time: 4-5 hours")
    print(f"{'='*70}\n")
    
    # Training callbacks
    def on_train_epoch_end(trainer):
        """Monitor training sau mỗi epoch"""
        epoch = trainer.epoch
        if epoch % 10 == 0:
            metrics = trainer.metrics
            print(f"\n📊 Epoch {epoch}: mAP50={metrics.get('metrics/mAP50(B)', 0):.4f}, "
                  f"Recall={metrics.get('metrics/recall(B)', 0):.4f}")
    
    # Train model
    results = model.train(
        # Data
        data=data_yaml,
        
        # Basic settings
        epochs=100,
        batch=10,            # Reduced from 14 for aggressive augmentation
        imgsz=640,
        patience=30,
        close_mosaic=10,
        
        # Device
        device=device,
        workers=8,
        
        # Optimizer & LR schedule (IMPROVED)
        optimizer='AdamW',
        lr0=0.015,           # 🔥 Tăng từ 0.012
        lrf=0.0001,          # 🔥 Giảm từ 0.001
        cos_lr=True,         # 🔥 NEW: Cosine LR schedule
        warmup_epochs=8,     # 🔥 Tăng từ 5
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights (OPTIMIZED FOR RECALL)
        box=12.0,            # 🔥🔥 High - Focus cigarette localization
        cls=2.0,             # 🔥 LOWER - Prioritize detection over classification
        dfl=2.5,             # 🔥🔥 High - Small cigarette objects
        
        # Augmentation (AGGRESSIVE FOR CIGARETTE)
        scale=0.5,           # 🔥🔥 Small objects (50% scale)
        copy_paste=0.5,      # 🔥🔥 VERY HIGH - Max cigarette instances
        mixup=0.25,          # 🔥🔥 HIGH - Hard negative examples
        translate=0.2,       # 🔥🔥 20% shift - Edge cases & corners
        degrees=15,          # 🔥 More rotation angles
        shear=3,             # 🔥 Increased skew
        mosaic=1.0,
        flipud=0.0,          # Cigarettes không đảo ngược
        fliplr=0.5,
        perspective=0.0005,
        hsv_h=0.02,          # 🔥 Color variation (white/yellow cigarettes)
        hsv_s=0.8,           # 🔥 Saturation (low contrast backgrounds)
        hsv_v=0.5,           # 🔥 Lighting conditions
        
        # Other settings
        amp=True,            # Automatic Mixed Precision
        deterministic=True,  # Reproducible results
        seed=0,
        
        # Output
        project='runs/train',
        name='smoking_detection_v7_improved',
        exist_ok=True,
        save=True,
        save_period=10,      # Save checkpoint mỗi 10 epochs
        plots=True,
        verbose=True,
    )
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETED!")
    print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # Print results
    print(f"\n📊 FINAL RESULTS:")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"   Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"   Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}")
    
    # Comparison với v6
    v6_map50 = 77.27
    v6_precision = 87.67
    v6_recall = 70.64
    
    map50_diff = (results.results_dict.get('metrics/mAP50(B)', 0) * 100) - v6_map50
    precision_diff = (results.results_dict.get('metrics/precision(B)', 0) * 100) - v6_precision
    recall_diff = (results.results_dict.get('metrics/recall(B)', 0) * 100) - v6_recall
    
    print(f"\n📈 COMPARISON với v6_optimized:")
    print(f"   mAP50: {map50_diff:+.2f}% (target: +2-3%)")
    print(f"   Precision: {precision_diff:+.2f}%")
    print(f"   Recall: {recall_diff:+.2f}% (target: +3-5%)")
    
    # Success criteria
    v7_map50 = results.results_dict.get('metrics/mAP50(B)', 0) * 100
    v7_recall = results.results_dict.get('metrics/recall(B)', 0) * 100
    
    print(f"\n🎯 SUCCESS CRITERIA:")
    if v7_map50 >= 79 and v7_recall >= 74:
        print(f"   ✅ PASS: mAP50={v7_map50:.2f}%, Recall={v7_recall:.2f}%")
        print(f"   → Continue to v8 (Progressive training)")
    elif v7_map50 >= 78 and v7_recall >= 72:
        print(f"   ⚠️ PARTIAL: mAP50={v7_map50:.2f}%, Recall={v7_recall:.2f}%")
        print(f"   → Analyze & adjust, retry v7")
    else:
        print(f"   ❌ FAIL: mAP50={v7_map50:.2f}%, Recall={v7_recall:.2f}%")
        print(f"   → Rollback changes, investigate issues")
    
    # Save location
    print(f"\n💾 Results saved to:")
    print(f"   runs/train/smoking_detection_v7_improved/")
    print(f"   - results.csv (metrics per epoch)")
    print(f"   - args.yaml (training config)")
    print(f"   - weights/best.pt (best model)")
    print(f"   - weights/last.pt (last epoch)")
    
    print(f"\n📝 NEXT STEPS:")
    print(f"   1. Copy results to ketquatrain/v7_improved/")
    print(f"   2. Update BAO_CAO_TONG_KET_TRAINING.md")
    print(f"   3. Analyze training curves")
    print(f"   4. Compare with v6 baseline")
    print(f"   5. Decide: Continue to v8 or adjust v7")
    
    print(f"\n{'='*70}")
    print(f"🎉 Training script completed!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
