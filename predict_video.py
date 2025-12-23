"""
Dự đoán Smoking Detection trên video
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
import time

def predict_video(model_path, video_path, output_dir='results/video',
                  conf_threshold=0.3, head_threshold=80, upper_threshold=150,
                  strict_face_only=False, save_result=True, show_result=True, 
                  save_frames=True, debug=False):
    """
    Dự đoán smoking detection trên video
    save_frames: Lưu các frames có smoking vào folder riêng
    show_result: Hiển thị video trong quá trình xử lý (default: True)
    """
    # Load model
    print(f"Loading model: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    
    # Open video
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return
    
    # Thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output paths
    base_name = Path(video_path).stem
    ext = Path(video_path).suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup video writer (status sẽ được thêm sau khi xử lý xong)
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        temp_output = os.path.join(output_dir, f"{timestamp}_temp_{base_name}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        print(f"Processing video...")
    
    # Setup frames folder
    if save_frames:
        frames_dir = os.path.join(output_dir, f"{base_name}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Frames folder: {frames_dir}")
    
    frame_count = 0
    smoking_frames = 0
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print("START PROCESSING VIDEO")
    print(f"{'='*60}\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Dự đoán
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False
        )
        
        filter_params = get_recommended_thresholds((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        results = filter_cigarette_detections(results, debug=False, **filter_params)
        
        is_smoking, smoking_persons, details = is_smoking_detected(
            results,
            head_threshold=head_threshold,
            upper_threshold=upper_threshold,
            conf_threshold=conf_threshold,
            strict_face_only=strict_face_only,
            debug=False
        )
        
        if is_smoking:
            smoking_frames += 1
        
        annotated_frame = results[0].plot()
        
        if save_frames and is_smoking:
            frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            frame_filename = f"{frame_timestamp}_smoking_frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, annotated_frame)
        
        label, color = get_smoking_label(is_smoking, details)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated_frame,
                      (10, 10),
                      (20 + text_width, 25 + text_height),
                      color,
                      -1)
        
        cv2.putText(annotated_frame,
                    label,
                    (15, 20 + text_height),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness)
        
        counter_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(annotated_frame,
                    counter_text,
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
        
        if save_result:
            out.write(annotated_frame)
        
        if show_result:
            cv2.imshow('Smoking Detection - Video', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
        
        if frame_count % 30 == 0 or frame_count == total_frames:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"FPS: {fps_current:.1f} | Smoking frames: {smoking_frames}")
    
    cap.release()
    if save_result:
        out.release()
    
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    smoking_percentage = (smoking_frames / frame_count) * 100 if frame_count > 0 else 0
    
    if save_result:
        status = "smoking" if smoking_percentage >= 5.0 else "non_smoking"
        output_path = os.path.join(output_dir, f"{timestamp}_{status}_{base_name}{ext}")
        os.rename(temp_output, output_path)
    
    print(f"\n{'='*60}")
    print("PROCESSING RESULTS")
    print(f"{'='*60}")
    print(f"  Total frames: {frame_count}")
    print(f"  Smoking frames: {smoking_frames} ({smoking_percentage:.1f}%)")
    print(f"  Processing time: {total_time:.1f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    if save_result:
        print(f"  Video saved: {output_path}")
    if save_frames and smoking_frames > 0:
        print(f"  Frames saved: {smoking_frames} images in {frames_dir}")
    print(f"{'='*60}\n")

def main():
    import argparse
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    default_model = script_dir / 'ketquatrain' / 'v6_optimized' / 'weights' / 'best.pt'
    
    parser = argparse.ArgumentParser(description='Smoking Detection - Video Prediction')
    parser.add_argument('--model', type=str, default=str(default_model),
                       help='Path to model weights (.pt)')
    parser.add_argument('--video', type=str, default=None, help='Path to input video (nếu không có sẽ xử lý video đầu tiên trong input_data/videos)')
    parser.add_argument('--input-dir', type=str, default=str(script_dir / 'input_data' / 'videos'), help='Input directory chứa video')
    parser.add_argument('--output', type=str, default=str(script_dir / 'results' / 'video'), help='Output directory')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold (optimal: 0.20 for best mAP50=66.07% and Cigarette detection)')
    parser.add_argument('--head-dist', type=int, default=80, help='Max distance to face/head to DRAW line (pixels)')
    parser.add_argument('--upper-dist', type=int, default=150, help='Max distance to upper body to DETECT (pixels)')
    parser.add_argument('--strict-face', action='store_true', help='Chỉ phát hiện gần mặt (bỏ qua nửa trên cơ thể)')
    parser.add_argument('--no-show', action='store_true', help='Do not show result while processing (default: show)')
    parser.add_argument('--no-save', action='store_true', help='Do not save result video')
    parser.add_argument('--no-frames', action='store_true', help='Do not save smoking frames')
    parser.add_argument('--debug', action='store_true', help='Show debug info')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print(f"   Please train model first: python train.py")
        return
    
    if args.video is not None:
        if not os.path.exists(args.video):
            print(f"Video not found: {args.video}")
            return
        video_list = [args.video]
        print(f"Processing 1 video: {args.video}")
    else:
        import glob
        video_list = glob.glob(f'{args.input_dir}/*.mp4') + glob.glob(f'{args.input_dir}/*.avi') + glob.glob(f'{args.input_dir}/*.mov')
        
        if not video_list:
            print(f"No videos found in {args.input_dir}")
            print(f"   Please copy videos to {args.input_dir} or use --video <path>")
            return
        
        print(f"Found {len(video_list)} videos in {args.input_dir}")
        print(f"Starting processing...")
    
    for idx, video_path in enumerate(video_list, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(video_list)}] Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}\n"))
        
        predict_video(
            model_path=args.model,
            video_path=video_path,
            output_dir=args.output,
            conf_threshold=args.conf,
            head_threshold=args.head_dist,
            upper_threshold=args.upper_dist,
            strict_face_only=args.strict_face,
            save_result=not args.no_save,
            show_result=not args.no_show,
            save_frames=not args.no_frames,
            debug=args.debug
        )

if __name__ == "__main__":
    main()
