"""
Dự đoán Smoking Detection realtime từ camera
Dataset: 2 classes (Cigarette, Person)
Logic: Phát hiện Cigarette gần vùng đầu Person → SMOKING
"""

from ultralytics import YOLO
import cv2
import os
import torch
from datetime import datetime
from smoking_detector import is_smoking_detected, get_smoking_label
from cigarette_filter import filter_cigarette_detections, get_recommended_thresholds
import time

def predict_camera(model_path, camera_id=0, output_dir='results/camera',
                   conf_threshold=0.3, head_threshold=80, upper_threshold=150,
                   strict_face_only=False, save_screenshots=True, debug=False):
    """
    Dự đoán smoking detection realtime từ camera
    
    Args:
        model_path: Đường dẫn model weights
        camera_id: ID camera (0 = webcam mặc định)
        output_dir: Thư mục lưu screenshots
        conf_threshold: Ngưỡng confidence
        head_threshold: Khoảng cách tối đa cigarette-đầu
        upper_threshold: Khoảng cách tối đa cigarette-nửa trên cơ thể
        save_screenshots: Tự động lưu ảnh khi phát hiện smoking
        debug: Hiển thị debug info
    """
    # Load model
    print(f"📦 Loading model: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # Mở camera
    print(f"📷 Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở camera {camera_id}")
        return
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera opened: {width}x{height}")
    
    # Statistics
    frame_count = 0
    smoking_detections = 0
    last_save_time = 0
    save_cooldown = 1  # Lưu ảnh mỗi 1 giây khi phát hiện smoking
    
    if save_screenshots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 Screenshots will be saved to: {output_dir}")
    
    print(f"\n{'='*60}")
    print("📹 CAMERA REALTIME DETECTION")
    print(f"{'='*60}")
    print("  Nhấn 'q' để thoát")
    print("  Nhấn 's' để lưu screenshot")
    print("  Nhấn 'd' để toggle debug mode")
    print(f"{'='*60}\n")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    debug_mode = debug
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không thể đọc frame từ camera")
            break
        
        frame_count += 1
        fps_frame_count += 1
        
        # Tính FPS
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
        
        # Dự đoán
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False
        )
        
        # Lọc cigarette detections
        h, w = frame.shape[:2]
        filter_params = get_recommended_thresholds((w, h))
        results = filter_cigarette_detections(results, debug=False, **filter_params)
        
        # Phát hiện smoking
        is_smoking, smoking_persons, details = is_smoking_detected(
            results,
            head_threshold=head_threshold,
            upper_threshold=upper_threshold,
            conf_threshold=conf_threshold,
            strict_face_only=strict_face_only,
            debug=debug_mode
        )
        
        if is_smoking:
            smoking_detections += 1
        
        # Vẽ kết quả
        annotated_frame = results[0].plot()
        
        # Label chính
        label, color = get_smoking_label(is_smoking, details)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated_frame,
                      (10, 10),
                      (20 + text_width, 30 + text_height),
                      color,
                      -1)
        
        cv2.putText(annotated_frame,
                    label,
                    (15, 25 + text_height),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness)
        
        # FPS counter
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(annotated_frame,
                    fps_text,
                    (width - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)
        
        # Stats
        stats_y = height - 60
        cv2.putText(annotated_frame,
                    f"Frames: {frame_count}",
                    (10, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
        
        cv2.putText(annotated_frame,
                    f"Smoking detections: {smoking_detections}",
                    (10, stats_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
        
        # Auto-save screenshot khi phát hiện smoking
        current_time = time.time()
        if save_screenshots and is_smoking and (current_time - last_save_time) >= save_cooldown:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(output_dir, f"{timestamp}_smoking_camera.jpg")
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"📸 Auto-saved: {screenshot_path}")
            last_save_time = current_time
        
        # Hiển thị (nếu có GUI support)
        try:
            cv2.imshow('Smoking Detection - Camera', annotated_frame)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            # OpenCV không có GUI support, chỉ lưu ảnh
            key = 0xFF
            if frame_count % 30 == 0:
                print(f"⏳ Frame {frame_count} | Smoking: {smoking_detections}")
        
        # Xử lý phím
        
        if key == ord('q'):
            print("\n⚠️  Thoát...")
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "smoking" if is_smoking else "non_smoking"
            screenshot_path = os.path.join(output_dir, f"{timestamp}_{status}_camera.jpg")
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"📸 Manual screenshot saved: {screenshot_path}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print("📊 THỐNG KÊ")
    print(f"{'='*60}")
    print(f"  Tổng frames: {frame_count}")
    print(f"  Smoking detections: {smoking_detections}")
    if frame_count > 0:
        print(f"  Tỷ lệ smoking: {(smoking_detections/frame_count)*100:.1f}%")
    print(f"{'='*60}\n")

def main():
    """Main function"""
    import argparse
    from pathlib import Path
    
    # Auto-detect model path (trong thư mục hiện tại)
    script_dir = Path(__file__).parent
    default_model = script_dir / 'ketquatrain' / 'v6_optimized' / 'weights' / 'best.pt'
    
    parser = argparse.ArgumentParser(description='Smoking Detection - Camera Realtime')
    parser.add_argument('--model', type=str, default=str(default_model),
                       help='Path to model weights (.pt)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--output', type=str, default=str(script_dir / 'results' / 'camera'), help='Output directory')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold (optimal: 0.20 for best mAP50=66.07% and Cigarette detection)')
    parser.add_argument('--head-dist', type=int, default=80, help='Max distance to face/head to DRAW line (pixels)')
    parser.add_argument('--upper-dist', type=int, default=150, help='Max distance to upper body to DETECT (pixels)')
    parser.add_argument('--strict-face', action='store_true', help='Chỉ phát hiện gần mặt (bỏ qua nửa trên cơ thể)')
    parser.add_argument('--no-save', action='store_true', help='Do not auto-save screenshots')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model không tồn tại: {args.model}")
        print(f"   Vui lòng train model trước: python train.py")
        return
    
    predict_camera(
        model_path=args.model,
        camera_id=args.camera,
        output_dir=args.output,
        conf_threshold=args.conf,
        head_threshold=args.head_dist,
        upper_threshold=args.upper_dist,
        strict_face_only=args.strict_face,
        save_screenshots=not args.no_save,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
