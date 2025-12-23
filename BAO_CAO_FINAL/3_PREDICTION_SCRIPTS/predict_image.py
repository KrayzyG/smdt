"""
Dự đoán Smoking Detection trên ảnh đơn
Dataset: 2 classes (Cigarette, Person)
Logic: Phát hiện Cigarette gần vùng đầu Person → SMOKING
"""

from ultralytics import YOLO
import cv2
import os
import torch
from pathlib import Path
from datetime import datetime
from smoking_detector import is_smoking_detected, get_smoking_label
from cigarette_filter import filter_cigarette_detections, get_recommended_thresholds

def predict_image(model_path, image_path, output_dir=None, 
                  conf_threshold=0.3, head_threshold=80, upper_threshold=150,
                  strict_face_only=False, save_result=True, show_result=False, debug=False):
    """
    Dự đoán smoking detection trên ảnh
    
    Args:
        model_path: Đường dẫn đến model weights (.pt)
        image_path: Đường dẫn ảnh input
        output_dir: Thư mục lưu kết quả (mặc định: ./results/image)
        conf_threshold: Ngưỡng confidence (0-1)
        head_threshold: Khoảng cách tối đa từ cigarette đến đầu (pixels)
        upper_threshold: Khoảng cách tối đa từ cigarette đến nửa trên cơ thể (pixels)
        save_result: Lưu ảnh kết quả
        show_result: Hiển thị ảnh kết quả
        debug: Hiển thị thông tin debug
    """
    # Set default output_dir if not provided
    if output_dir is None:
        output_dir = str(Path(__file__).parent / 'results' / 'image')
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    
    # Đọc ảnh
    print(f"📷 Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return
    
    # Dự đoán
    results = model.predict(
        source=img,
        conf=conf_threshold,
        verbose=False
    )
    
    # Lọc cigarette detections để giảm false positives
    if debug:
        print(f"\n🔍 Lọc cigarette detections...")
    
    # Lấy recommended thresholds dựa trên kích thước ảnh
    img_height, img_width = img.shape[:2]
    filter_params = get_recommended_thresholds((img_width, img_height))
    
    if debug:
        print(f"   Kích thước ảnh: {img_width}x{img_height}")
        print(f"   Filter params: min_conf={filter_params['min_conf_cigarette']}, "
              f"aspect_ratio={filter_params['min_aspect_ratio']}-{filter_params['max_aspect_ratio']}, "
              f"area={filter_params['min_area']}-{filter_params['max_area']}px, "
              f"max_dist={filter_params['max_distance_to_person']}px")
    
    results = filter_cigarette_detections(results, debug=debug, **filter_params)
    
    # Phát hiện smoking
    is_smoking, smoking_persons, details = is_smoking_detected(
        results, 
        head_threshold=head_threshold,
        upper_threshold=upper_threshold,
        conf_threshold=conf_threshold,
        strict_face_only=strict_face_only,
        debug=debug
    )
    
    # Vẽ kết quả
    annotated_img = results[0].plot()  # Vẽ tất cả detections
    
    # Thêm label smoking/non-smoking
    label, color = get_smoking_label(is_smoking, details)
    
    # Vẽ text lớn ở góc trái trên
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Lấy kích thước text để vẽ background
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Vẽ background cho text
    cv2.rectangle(annotated_img, 
                  (10, 10), 
                  (20 + text_width, 30 + text_height), 
                  color, 
                  -1)  # Filled
    
    # Vẽ text
    cv2.putText(annotated_img, 
                label, 
                (15, 25 + text_height), 
                font, 
                font_scale, 
                (255, 255, 255),  # Trắng
                thickness)
    
    # In kết quả
    print(f"\n{'='*60}")
    print(f"🎯 KẾT QUẢ PHÁT HIỆN")
    print(f"{'='*60}")
    print(f"  Trạng thái: {label}")
    print(f"  👤 Số người phát hiện: {details['total_persons']}")
    print(f"  🚬 Số cigarette phát hiện: {details['total_cigarettes']}")
    if is_smoking:
        print(f"  ⚠️  Số người đang smoking: {details['smoking_count']}")
        for i, match in enumerate(details['matches'], 1):
            print(f"     Person #{match['person_idx']}: distance = {match['distance']:.1f}px")
    print(f"{'='*60}\n")
    
    # Lưu kết quả
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "smoking" if is_smoking else "non_smoking"
        base_name = Path(image_path).stem
        ext = Path(image_path).suffix
        output_path = os.path.join(output_dir, f"{timestamp}_{status}_{base_name}{ext}")
        cv2.imwrite(output_path, annotated_img)
        print(f"💾 Đã lưu kết quả: {output_path}")
    
    # Hiển thị
    if show_result:
        try:
            cv2.imshow('Smoking Detection Result', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print(f"⚠️  Không thể hiển thị ảnh (OpenCV GUI không hỗ trợ)")
            print(f"📂 Xem kết quả tại: {output_path}")
            # Mở bằng default image viewer
            import subprocess
            subprocess.run(['start', output_path], shell=True)
    
    return is_smoking, annotated_img, details

def main():
    """Main function"""
    import argparse
    import glob
    from pathlib import Path
    
    # Auto-detect model path (trong thư mục hiện tại)
    script_dir = Path(__file__).parent
    # Workspace root là 3 cấp trên: BAO_CAO_FINAL/3_PREDICTION_SCRIPTS -> BAO_CAO_FINAL -> smoking_with_yolov8 + aug -> wsf1
    workspace_root = script_dir.parent.parent.parent
    default_model = workspace_root / 'runs' / 'train' / 'smoking_detection_v7_improved' / 'weights' / 'best.pt'
    
    parser = argparse.ArgumentParser(description='Smoking Detection - Image Prediction')
    parser.add_argument('--model', type=str, default=str(default_model), 
                       help='Path to model weights (.pt)')
    parser.add_argument('--image', type=str, default=None, help='Path to input image (nếu không có sẽ xử lý tất cả ảnh trong input_data/images)')
    parser.add_argument('--input-dir', type=str, default=str(workspace_root / 'smoking_with_yolov8 + aug' / 'input_data' / 'images'), help='Input directory chứa ảnh')
    parser.add_argument('--output', type=str, default=str(script_dir / 'results' / 'image'), help='Output directory')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold (optimal: 0.20 for best mAP50=66.07%%, Cigarette mAP50=54.17%%)')
    parser.add_argument('--head-dist', type=int, default=80, help='Max distance to face/head to DRAW line (pixels)')
    parser.add_argument('--upper-dist', type=int, default=150, help='Max distance to upper body to DETECT (pixels)')
    parser.add_argument('--strict-face', action='store_true', help='Chỉ phát hiện gần mặt (bỏ qua nửa trên cơ thể)')
    parser.add_argument('--show', action='store_true', help='Show result image')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    
    args = parser.parse_args()
    
    # Kiểm tra model
    if not os.path.exists(args.model):
        print(f"❌ Model không tồn tại: {args.model}")
        print(f"   Vui lòng train model trước: python train.py")
        return
    
    # Xử lý ảnh
    if args.image is not None:
        # Xử lý 1 ảnh cụ thể
        if not os.path.exists(args.image):
            print(f"❌ Ảnh không tồn tại: {args.image}")
            return
        
        image_list = [args.image]
        print(f"📷 Xử lý 1 ảnh: {args.image}")
    else:
        # Xử lý tất cả ảnh trong input_data/images
        image_list = glob.glob(f'{args.input_dir}/*.jpg') + glob.glob(f'{args.input_dir}/*.png') + glob.glob(f'{args.input_dir}/*.jpeg')
        
        if not image_list:
            # Copy một số ảnh test vào input_data
            test_images = glob.glob(str(workspace_root / 'dataset' / 'smoking_train_image_v6' / 'test' / 'images' / '*.jpg'))[:5]
            if test_images:
                import shutil
                os.makedirs(args.input_dir, exist_ok=True)
                for test_img in test_images:
                    shutil.copy(test_img, args.input_dir)
                print(f"📋 Đã copy {len(test_images)} ảnh test vào {args.input_dir}")
                image_list = glob.glob(f'{args.input_dir}/*.jpg')
            else:
                print(f"❌ Không tìm thấy ảnh trong {args.input_dir}")
                print(f"   Vui lòng copy ảnh vào thư mục {args.input_dir} hoặc dùng --image <path>")
                return
        
        print(f"📂 Tìm thấy {len(image_list)} ảnh trong {args.input_dir}")
        print(f"🚀 Bắt đầu xử lý...")
    
    # Xử lý từng ảnh
    results_summary = {
        'total': len(image_list),
        'smoking': 0,
        'non_smoking': 0,
        'processed': []
    }
    
    for idx, img_path in enumerate(image_list, 1):
        print(f"\n{'='*60}")
        print(f"📷 [{idx}/{len(image_list)}] Processing: {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        is_smoking, annotated_img, details = predict_image(
            model_path=args.model,
            image_path=img_path,
            output_dir=args.output,
            conf_threshold=args.conf,
            head_threshold=args.head_dist,
            upper_threshold=args.upper_dist,
            strict_face_only=args.strict_face,
            save_result=True,
            show_result=False,  # Không show từng ảnh
            debug=args.debug
        )
        
        # Cập nhật summary
        if is_smoking:
            results_summary['smoking'] += 1
        else:
            results_summary['non_smoking'] += 1
        
        results_summary['processed'].append({
            'image': os.path.basename(img_path),
            'status': 'SMOKING' if is_smoking else 'NON-SMOKING',
            'persons': details['total_persons'],
            'cigarettes': details['total_cigarettes']
        })
    
    # Hiển thị tổng kết
    print(f"\n{'='*60}")
    print(f"📊 TỔNG KẾT XỬ LÝ")
    print(f"{'='*60}")
    print(f"  Tổng số ảnh: {results_summary['total']}")
    print(f"  ❌ SMOKING: {results_summary['smoking']}")
    print(f"  ✅ NON-SMOKING: {results_summary['non_smoking']}")
    print(f"  📁 Kết quả lưu tại: {os.path.abspath(args.output)}")
    print(f"{'='*60}")
    
    # Chi tiết từng ảnh
    if args.debug:
        print(f"\n📋 Chi tiết:")
        for r in results_summary['processed']:
            status_icon = "❌" if r['status'] == 'SMOKING' else "✅"
            print(f"  {status_icon} {r['image']}: {r['status']} (Persons: {r['persons']}, Cigarettes: {r['cigarettes']})")
    
    # Mở folder kết quả
    if args.show:
        import subprocess
        subprocess.run(['explorer', os.path.abspath(args.output)], shell=True)
        print(f"\n📂 Đã mở folder kết quả")


if __name__ == "__main__":
    main()
