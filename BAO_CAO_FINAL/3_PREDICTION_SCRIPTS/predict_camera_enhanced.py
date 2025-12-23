"""
Dự đoán Smoking Detection realtime từ camera với ENHANCEMENT
Cải thiện cho camera chất lượng kém: low light, blur, noise, low contrast
"""

from ultralytics import YOLO
import cv2
import os
import torch
from datetime import datetime
from smoking_detector import is_smoking_detected, get_smoking_label
from cigarette_filter import filter_cigarette_detections, get_recommended_thresholds
from camera_enhancement import CameraEnhancer, AdaptiveConfidenceAdjuster, check_camera_quality, recommend_camera_settings
import time
import numpy as np

def predict_camera_enhanced(model_path, camera_id=0, output_dir='results/camera',
                   conf_threshold=0.25, head_threshold=80, upper_threshold=150,
                   strict_face_only=False, save_screenshots=True, debug=False,
                   enable_enhancement=True, adaptive_conf=True):
    """
    Dự đoán smoking detection realtime với camera enhancement
    
    Args:
        model_path: Đường dẫn model weights
        camera_id: ID camera (0 = webcam mặc định)
        output_dir: Thư mục lưu screenshots
        conf_threshold: Ngưỡng confidence base
        head_threshold: Khoảng cách tối đa cigarette-đầu
        upper_threshold: Khoảng cách tối đa cigarette-nửa trên cơ thể
        save_screenshots: Tự động lưu ảnh khi phát hiện smoking
        debug: Hiển thị debug info
        enable_enhancement: Bật enhancement cho camera kém
        adaptive_conf: Điều chỉnh confidence theo điều kiện ánh sáng
    """
    # Load model
    print(f"📦 Loading model: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # Mở camera
    print(f"\n📷 Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở camera {camera_id}")
        return
    
    # Camera settings (tối ưu cho detection)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Camera opened: {width}x{height} @ {camera_fps}fps")
    
    # Check camera quality
    print(f"\n🔍 Analyzing camera quality...")
    quality_info = check_camera_quality(cap)
    if quality_info:
        print(f"\n📊 Camera Quality Report:")
        print(f"   Brightness: {quality_info['brightness']:.1f} (optimal: 100-150)")
        print(f"   Blur Score: {quality_info['blur_score']:.1f} (higher = sharper)")
        print(f"   Contrast: {quality_info['contrast']:.1f} (higher = better)")
        print(f"   Overall: {quality_info['quality']}")
        
        if quality_info['issues']:
            print(f"   ⚠️  Issues: {', '.join(quality_info['issues'])}")
            
            print(f"\n💡 Recommendations:")
            for rec in recommend_camera_settings(quality_info):
                print(f"   {rec}")
            
            if enable_enhancement:
                print(f"\n✅ Auto-enhancement ENABLED to compensate!")
            else:
                print(f"\n⚠️  Auto-enhancement DISABLED - consider enabling with --enhance")
    
    # Initialize enhancer
    enhancer = None
    conf_adjuster = None
    
    if enable_enhancement:
        enhancer = CameraEnhancer(
            auto_enhance=True,
            denoise=True,      # Giảm noise
            sharpen=True,      # Tăng sharpness
            clahe=True,        # Tăng contrast
            auto_wb=True       # Auto white balance
        )
        print(f"🎨 Camera Enhancer initialized")
    
    if adaptive_conf:
        conf_adjuster = AdaptiveConfidenceAdjuster(base_conf=conf_threshold)
        print(f"⚙️  Adaptive Confidence Adjuster initialized")
    
    # Statistics
    frame_count = 0
    smoking_detections = 0
    last_save_time = 0
    save_cooldown = 1  # Lưu ảnh mỗi 1 giây
    
    if save_screenshots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 Screenshots will be saved to: {output_dir}")
    
    print(f"\n{'='*70}")
    print("📹 ENHANCED CAMERA REALTIME DETECTION")
    print(f"{'='*70}")
    print("  Nhấn 'q' để thoát")
    print("  Nhấn 's' để lưu screenshot")
    print("  Nhấn 'd' để toggle debug mode")
    print("  Nhấn 'e' để toggle enhancement")
    print("  Nhấn 'a' để toggle adaptive confidence")
    print("  Nhấn 'i' để show quality info")
    print(f"{'='*70}\n")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    debug_mode = debug
    show_quality_info = False
    enhancement_enabled = enable_enhancement
    adaptive_enabled = adaptive_conf
    
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
        
        # Store original for comparison
        original_frame = frame.copy()
        
        # Apply enhancement
        if enhancement_enabled and enhancer:
            frame = enhancer.enhance(frame)
        
        # Get adaptive confidence
        current_conf = conf_threshold
        if adaptive_enabled and conf_adjuster:
            current_conf = conf_adjuster.get_adaptive_conf(frame)
        
        # Dự đoán
        results = model.predict(
            source=frame,
            conf=current_conf,
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
            conf_threshold=current_conf,
            strict_face_only=strict_face_only,
            debug=debug_mode
        )
        
        if is_smoking:
            smoking_detections += 1
        
        # Vẽ kết quả lên frame
        annotated_frame = results[0].plot() if results and len(results) > 0 else frame.copy()
        
        # Vẽ smoking persons
        for person_data in smoking_persons:
            x1, y1, x2, y2 = person_data['person_box']
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 0, 255), 3)
            
            label = f"SMOKING (dist: {person_data['distance']:.0f}px)"
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Status overlay
        status_color = (0, 0, 255) if is_smoking else (0, 255, 0)
        status_text = "SMOKING DETECTED" if is_smoking else "NON-SMOKING"
        
        # Background cho status
        cv2.rectangle(annotated_frame, (10, 10), (400, 70), (0, 0, 0), -1)
        cv2.putText(annotated_frame, status_text, (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # Info overlay
        info_y = 90
        cv2.rectangle(annotated_frame, (10, info_y), (420, info_y + 180), (0, 0, 0), -1)
        
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (20, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, info_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Persons: {details['total_persons']}", (20, info_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Cigarettes: {details['total_cigarettes']}", (20, info_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Conf: {current_conf:.2f}", (20, info_y + 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Enhancement status
        enh_text = "ENH: ON" if enhancement_enabled else "ENH: OFF"
        enh_color = (0, 255, 0) if enhancement_enabled else (128, 128, 128)
        cv2.putText(annotated_frame, enh_text, (20, info_y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, enh_color, 1)
        
        adapt_text = "ADAPT: ON" if adaptive_enabled else "ADAPT: OFF"
        adapt_color = (0, 255, 0) if adaptive_enabled else (128, 128, 128)
        cv2.putText(annotated_frame, adapt_text, (20, info_y + 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, adapt_color, 1)
        
        # Quality info overlay
        if show_quality_info and enhancer:
            quality = enhancer.get_quality_info(original_frame)
            q_y = 280
            cv2.rectangle(annotated_frame, (10, q_y), (380, q_y + 130), (0, 0, 0), -1)
            cv2.putText(annotated_frame, "QUALITY INFO:", (20, q_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(annotated_frame, f"Brightness: {quality['brightness']:.1f}", (20, q_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Blur Score: {quality['blur_score']:.1f}", (20, q_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Contrast: {quality['contrast']:.1f}", (20, q_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            q_color = (0, 255, 0) if quality['quality'] == 'Good' else (0, 165, 255) if quality['quality'] == 'Fair' else (0, 0, 255)
            cv2.putText(annotated_frame, f"Quality: {quality['quality']}", (20, q_y + 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 1)
        
        # Hiển thị
        cv2.imshow('Smoking Detection - Enhanced', annotated_frame)
        
        # Auto-save khi phát hiện smoking
        if is_smoking and save_screenshots:
            current_time = time.time()
            if current_time - last_save_time >= save_cooldown:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_smoking_detected.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, annotated_frame)
                print(f"💾 Saved: {filename}")
                last_save_time = current_time
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Manual save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_manual_save.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, annotated_frame)
            print(f"💾 Manual save: {filename}")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('e'):
            enhancement_enabled = not enhancement_enabled
            print(f"Enhancement: {'ON' if enhancement_enabled else 'OFF'}")
        elif key == ord('a'):
            adaptive_enabled = not adaptive_enabled
            print(f"Adaptive confidence: {'ON' if adaptive_enabled else 'OFF'}")
        elif key == ord('i'):
            show_quality_info = not show_quality_info
            print(f"Quality info: {'ON' if show_quality_info else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\n{'='*60}")
    print("📊 SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total frames: {frame_count}")
    print(f"  Smoking detections: {smoking_detections}")
    print(f"  Average FPS: {current_fps:.1f}")
    print(f"  Detection rate: {(smoking_detections/frame_count*100):.2f}%")
    print(f"{'='*60}\n")


def main():
    """Main function"""
    import argparse
    from pathlib import Path
    
    # Auto-detect model path
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    default_model = workspace_root / 'runs' / 'train' / 'smoking_detection_v7_improved' / 'weights' / 'best.pt'
    
    parser = argparse.ArgumentParser(description='Smoking Detection - Enhanced Camera Realtime')
    parser.add_argument('--model', type=str, default=str(default_model),
                       help='Path to model weights (.pt)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--output', type=str, default=str(script_dir / 'results' / 'camera_enhanced'), 
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Base confidence threshold (will be adjusted adaptively)')
    parser.add_argument('--head-dist', type=int, default=80, help='Max distance to face/head (pixels)')
    parser.add_argument('--upper-dist', type=int, default=150, help='Max distance to upper body (pixels)')
    parser.add_argument('--strict-face', action='store_true', help='Only detect near face')
    parser.add_argument('--no-save', action='store_true', help='Disable auto-save screenshots')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-enhance', action='store_true', help='Disable image enhancement')
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive confidence')
    
    args = parser.parse_args()
    
    # Kiểm tra model
    if not os.path.exists(args.model):
        print(f"❌ Model không tồn tại: {args.model}")
        return
    
    print(f"🚀 Starting Enhanced Camera Detection...")
    print(f"   Model: {args.model}")
    print(f"   Camera ID: {args.camera}")
    print(f"   Enhancement: {'DISABLED' if args.no_enhance else 'ENABLED'}")
    print(f"   Adaptive Conf: {'DISABLED' if args.no_adaptive else 'ENABLED'}")
    
    predict_camera_enhanced(
        model_path=args.model,
        camera_id=args.camera,
        output_dir=args.output,
        conf_threshold=args.conf,
        head_threshold=args.head_dist,
        upper_threshold=args.upper_dist,
        strict_face_only=args.strict_face,
        save_screenshots=not args.no_save,
        debug=args.debug,
        enable_enhancement=not args.no_enhance,
        adaptive_conf=not args.no_adaptive
    )


if __name__ == "__main__":
    main()
