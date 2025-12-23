"""
Module phát hiện hành vi hút thuốc thông minh
Dựa trên khoảng cách giữa Cigarette và vùng đầu của Person
"""

import numpy as np

def get_head_region(person_box):
    """
    Lấy vùng đầu của Person (20% phần trên của bounding box)
    
    Args:
        person_box: [x1, y1, x2, y2] - bounding box của Person
    
    Returns:
        [x1, y1, x2, y2_head] - bounding box vùng đầu
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    head_height = height * 0.2  # 20% phần trên là vùng đầu
    y2_head = y1 + head_height
    
    return [x1, y1, x2, y2_head]

def get_upper_body_region(person_box):
    """
    Lấy vùng nửa trên cơ thể (50% phần trên)
    
    Args:
        person_box: [x1, y1, x2, y2]
    
    Returns:
        [x1, y1, x2, y2_upper] - bounding box nửa trên cơ thể
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1
    y2_upper = y1 + height * 0.5
    
    return [x1, y1, x2, y2_upper]

def calculate_distance_to_box(point_box, target_box):
    """
    Tính khoảng cách từ tâm point_box đến target_box
    Nếu point_box overlap với target_box → khoảng cách = 0
    
    Args:
        point_box: [x1, y1, x2, y2] - box của điểm (cigarette)
        target_box: [x1, y1, x2, y2] - box mục tiêu (head/upper body)
    
    Returns:
        float: Khoảng cách (pixels)
    """
    # Tâm của point_box (cigarette)
    cx = (point_box[0] + point_box[2]) / 2
    cy = (point_box[1] + point_box[3]) / 2
    
    # Kiểm tra xem point có nằm trong target_box không
    if (target_box[0] <= cx <= target_box[2] and 
        target_box[1] <= cy <= target_box[3]):
        return 0.0  # Overlap → khoảng cách = 0
    
    # Tìm điểm gần nhất trên target_box
    closest_x = max(target_box[0], min(cx, target_box[2]))
    closest_y = max(target_box[1], min(cy, target_box[3]))
    
    # Tính khoảng cách Euclidean
    distance = np.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
    
    return distance

def calculate_iou(box1, box2):
    """
    Tính IoU (Intersection over Union) giữa 2 boxes
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU value (0-1)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou

def is_smoking_detected(results, 
                        head_threshold=80,       # Khoảng cách tối đa từ cigarette đến đầu để vẽ đường nối
                        upper_threshold=150,     # Khoảng cách tối đa từ cigarette đến nửa trên cơ thể để phát hiện
                        conf_threshold=0.3,      # Confidence tối thiểu
                        strict_face_only=False,  # False = cho phép phát hiện cả nửa trên cơ thể
                        debug=False):
    """
    Phát hiện hành vi hút thuốc dựa trên vị trí Cigarette gần đầu/mặt Person
    
    Logic ưu tiên (khi strict_face_only=True):
    1. Cigarette trong vùng đầu (20% trên) với khoảng cách <= head_threshold → SMOKING
    2. Ngược lại → NON-SMOKING
    
    Logic mở rộng (khi strict_face_only=False):
    1. Cigarette trong vùng đầu (20% trên) → SMOKING (độ ưu tiên cao)
    2. Cigarette trong nửa trên cơ thể (50% trên) và gần đầu → SMOKING
    3. Ngược lại → NON-SMOKING
    
    Args:
        results: YOLO detection results
        head_threshold: Khoảng cách tối đa (pixels) từ cigarette đến vùng đầu/mặt
        upper_threshold: Khoảng cách tối đa từ cigarette đến nửa trên cơ thể (chỉ dùng khi strict_face_only=False)
        conf_threshold: Ngưỡng confidence tối thiểu
        strict_face_only: True = chỉ phát hiện cigarette gần mặt, False = bao gồm cả nửa trên cơ thể
        debug: Hiển thị thông tin debug
    
    Returns:
        tuple: (is_smoking: bool, smoking_persons: list, details: dict)
    """
    smoking_persons = []  # Danh sách các person đang smoking
    details = {
        'total_persons': 0,
        'total_cigarettes': 0,
        'smoking_count': 0,
        'matches': []
    }
    
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        
        # Lọc theo confidence
        valid_mask = confs >= conf_threshold
        boxes = boxes[valid_mask]
        classes = classes[valid_mask]
        confs = confs[valid_mask]
        
        # Lấy Person boxes (class 1) và Cigarette boxes (class 0)
        person_indices = np.where(classes == 1)[0]
        cigarette_indices = np.where(classes == 0)[0]
        
        details['total_persons'] = len(person_indices)
        details['total_cigarettes'] = len(cigarette_indices)
        
        if debug:
            print(f"\n🔍 DEBUG - Detected objects:")
            print(f"   👤 Persons: {len(person_indices)}")
            print(f"   🚬 Cigarettes: {len(cigarette_indices)}")
        
        # Nếu không có cả Person và Cigarette → không smoking
        if len(person_indices) == 0 or len(cigarette_indices) == 0:
            return False, smoking_persons, details
        
        # Kiểm tra từng cặp Person-Cigarette
        for p_idx in person_indices:
            person_box = boxes[p_idx]
            person_conf = confs[p_idx]
            head_region = get_head_region(person_box)
            upper_region = get_upper_body_region(person_box)
            
            is_this_person_smoking = False
            closest_cigarette_dist = float('inf')
            closest_cigarette_box = None
            is_close_to_face = False  # Track nếu cigarette gần mặt (để vẽ đường nối)
            
            for c_idx in cigarette_indices:
                cigarette_box = boxes[c_idx]
                cigarette_conf = confs[c_idx]
                
                # Tính khoảng cách đến vùng đầu
                dist_to_head = calculate_distance_to_box(cigarette_box, head_region)
                
                if debug:
                    print(f"\n   📏 Person #{p_idx} ↔ Cigarette #{c_idx}:")
                    print(f"      Distance to head: {dist_to_head:.1f}px (threshold: {head_threshold}px)")
                
                # Ưu tiên 1: Cigarette gần mặt/đầu
                if dist_to_head <= head_threshold:
                    is_this_person_smoking = True
                    if dist_to_head < closest_cigarette_dist:
                        closest_cigarette_dist = dist_to_head
                        closest_cigarette_box = cigarette_box
                        is_close_to_face = True  # Đánh dấu là gần mặt
                    if debug:
                        print(f"      ✅ SMOKING detected (near face/head)!")
                
                # Ưu tiên 2: Cigarette trong nửa trên cơ thể (chỉ khi strict_face_only=False)
                elif not strict_face_only and upper_threshold is not None:
                    dist_to_upper = calculate_distance_to_box(cigarette_box, upper_region)
                    
                    if debug:
                        print(f"      Distance to upper body: {dist_to_upper:.1f}px (threshold: {upper_threshold}px)")
                    
                    if dist_to_upper <= upper_threshold:
                        is_this_person_smoking = True
                        if dist_to_upper < closest_cigarette_dist:
                            closest_cigarette_dist = dist_to_upper
                            closest_cigarette_box = cigarette_box
                            # KHÔNG đánh dấu is_close_to_face = True (không vẽ đường nối)
                        if debug:
                            print(f"      ✅ SMOKING detected (near upper body - no line)!")
                elif debug and not strict_face_only:
                    print(f"      ❌ Too far from face (distance: {dist_to_head:.1f}px > {head_threshold}px)")
            
            # Nếu person này đang smoking
            if is_this_person_smoking:
                smoking_persons.append({
                    'person_box': person_box.tolist(),
                    'person_conf': float(person_conf),
                    'cigarette_box': closest_cigarette_box.tolist(),
                    'distance': float(closest_cigarette_dist),
                    'is_close_to_face': is_close_to_face  # Thêm flag để biết có vẽ đường nối không
                })
                details['smoking_count'] += 1
                details['matches'].append({
                    'person_idx': int(p_idx),
                    'distance': float(closest_cigarette_dist)
                })
    
    is_smoking = len(smoking_persons) > 0
    
    return is_smoking, smoking_persons, details

def get_smoking_label(is_smoking, details=None):
    """
    Lấy label text và màu cho kết quả
    
    Returns:
        tuple: (label_text, color_bgr)
    """
    if is_smoking:
        if details and details['smoking_count'] > 1:
            label = f"⚠️ SMOKING ({details['smoking_count']} persons)"
        else:
            label = "⚠️ SMOKING"
        color = (0, 0, 255)  # Đỏ
    else:
        label = "✅ NON-SMOKING"
        color = (0, 255, 0)  # Xanh lá
    
    return label, color
