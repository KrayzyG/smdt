"""
Dataset Optimization for v6 Training
- Re-split: 85/10/5 (tăng train set)
- Filter hard negatives (cigarettes quá nhỏ, aspect ratio lệch)
- Enhanced data quality
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml

def parse_label(label_path):
    """Parse YOLO label file"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    objects = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        
        objects.append({
            'class': class_id,
            'x': x_center,
            'y': y_center,
            'w': width,
            'h': height,
            'area': width * height,
            'aspect_ratio': width / height if height > 0 else 0
        })
    
    return objects

def should_keep_image(label_path, min_area=0.0005, max_aspect=8, min_aspect=0.1):
    """
    Quyết định có nên giữ image hay không dựa trên chất lượng labels
    
    Filter rules:
    - Cigarette area < 0.0005 (quá nhỏ, <20px @ 640x640)
    - Aspect ratio > 8 hoặc < 0.1 (quá lệch)
    """
    if not os.path.exists(label_path):
        return False
    
    objects = parse_label(label_path)
    if not objects:
        return False
    
    for obj in objects:
        # Filter cigarettes quá nhỏ
        if obj['class'] == 0 and obj['area'] < min_area:
            return False
        
        # Filter aspect ratio quá lệch
        if obj['aspect_ratio'] > max_aspect or obj['aspect_ratio'] < min_aspect:
            return False
    
    return True

def optimize_dataset(
    source_dir="e:/LEARN/@ki1 nam 4/MACHINE LEARNING/smoke/wsf1/dataset/smoking_train_image_improved",
    output_dir="e:/LEARN/@ki1 nam 4/MACHINE LEARNING/smoke/wsf1/dataset/smoking_train_image_v6",
    train_ratio=0.85,
    val_ratio=0.10,
    test_ratio=0.05,
    seed=42
):
    """
    Tối ưu dataset cho v6:
    - Re-split 85/10/5
    - Filter hard negatives
    - Validate labels
    """
    
    print("="*60)
    print("🔧 DATASET OPTIMIZATION FOR V6")
    print("="*60)
    
    random.seed(seed)
    
    # Create output structure
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (output_path / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Collect all images from source
    source_path = Path(source_dir)
    all_images = []
    
    print("\n📂 Collecting images from source...")
    for split in ['train', 'val', 'test']:
        img_dir = source_path / split / 'images'
        if img_dir.exists():
            for img_file in img_dir.glob('*.*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_file = source_path / split / 'labels' / (img_file.stem + '.txt')
                    all_images.append({
                        'image': img_file,
                        'label': label_file,
                        'original_split': split
                    })
    
    print(f"   ✅ Found {len(all_images)} images")
    
    # Filter hard negatives
    print("\n🔍 Filtering hard negatives...")
    filtered_images = []
    filtered_count = 0
    
    for item in all_images:
        if should_keep_image(item['label']):
            filtered_images.append(item)
        else:
            filtered_count += 1
    
    print(f"   ✅ Kept {len(filtered_images)} images")
    print(f"   ❌ Filtered {filtered_count} hard negatives")
    
    # Shuffle and split
    random.shuffle(filtered_images)
    
    total = len(filtered_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_images = filtered_images[:train_end]
    val_images = filtered_images[train_end:val_end]
    test_images = filtered_images[val_end:]
    
    print(f"\n📊 New split:")
    print(f"   Train: {len(train_images)} ({len(train_images)/total*100:.1f}%)")
    print(f"   Val:   {len(val_images)} ({len(val_images)/total*100:.1f}%)")
    print(f"   Test:  {len(test_images)} ({len(test_images)/total*100:.1f}%)")
    
    # Copy files
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    print("\n📁 Copying files...")
    for split_name, images in splits.items():
        for item in images:
            # Copy image
            dst_img = output_path / split_name / 'images' / item['image'].name
            shutil.copy2(item['image'], dst_img)
            
            # Copy label
            if item['label'].exists():
                dst_label = output_path / split_name / 'labels' / item['label'].name
                shutil.copy2(item['label'], dst_label)
    
    print("   ✅ Files copied successfully")
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_path.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['Cigarette', 'Person']
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"\n✅ data.yaml created: {yaml_path}")
    
    # Validate
    print("\n🔍 Validating dataset...")
    for split_name in ['train', 'val', 'test']:
        img_count = len(list((output_path / split_name / 'images').glob('*.*')))
        label_count = len(list((output_path / split_name / 'labels').glob('*.txt')))
        print(f"   {split_name}: {img_count} images, {label_count} labels")
        
        if img_count != label_count:
            print(f"   ⚠️  Warning: Mismatch in {split_name}")
    
    # Statistics
    print("\n📊 Dataset Statistics:")
    for split_name in ['train', 'val', 'test']:
        cigarette_count = 0
        person_count = 0
        
        for label_file in (output_path / split_name / 'labels').glob('*.txt'):
            objects = parse_label(label_file)
            for obj in objects:
                if obj['class'] == 0:
                    cigarette_count += 1
                elif obj['class'] == 1:
                    person_count += 1
        
        total_obj = cigarette_count + person_count
        if total_obj > 0:
            print(f"   {split_name}:")
            print(f"      Cigarette: {cigarette_count} ({cigarette_count/total_obj*100:.1f}%)")
            print(f"      Person: {person_count} ({person_count/total_obj*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ DATASET OPTIMIZATION COMPLETED!")
    print("="*60)
    print(f"📂 Output: {output_path}")
    print(f"📄 Config: {yaml_path}")
    print("\n🚀 Ready for v6 training!")

if __name__ == "__main__":
    optimize_dataset()
