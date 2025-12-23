"""
Script để lọc cigarette detections dựa trên nhiều tiêu chí
Giảm false positives bằng cách:
1. Tăng confidence threshold cho cigarette
2. Kiểm tra aspect ratio (cigarettes thường dài và mỏng)
3. Kiểm tra kích thước tuyệt đối (không quá lớn hoặc quá nhỏ)
4. Chỉ chấp nhận cigarette gần người
"""

def filter_cigarette_detections(results, 
                                min_conf_cigarette=0.30,  # Cao hơn person
                                min_aspect_ratio=1.5,      # Chiều dài/rộng tối thiểu
                                max_aspect_ratio=8.0,      # Chiều dài/rộng tối đa
                                min_area=50,               # Diện tích tối thiểu (pixels)
                                max_area=5000,             # Diện tích tối đa (pixels)
                                max_distance_to_person=300, # Khoảng cách tối đa đến người
                                debug=False):
    """
    Lọc cigarette detections để giảm false positives
    
    Returns:
        filtered_results: Results object với chỉ cigarettes đã lọc và persons
    """
    if len(results) == 0 or results[0].boxes is None:
        return results
    
    boxes = results[0].boxes
    filtered_indices = []
    
    # Tìm tất cả persons để tính khoảng cách
    person_boxes = []
    for i, (box, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
        if int(cls) == 1:  # Person class
            person_boxes.append(box)
            filtered_indices.append(i)  # Giữ tất cả persons
    
    # Lọc cigarettes
    cigarette_count = 0
    filtered_cigarettes = 0
    
    for i, (box, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
        if int(cls) == 0:  # Cigarette class
            cigarette_count += 1
            x1, y1, x2, y2 = box
            
            # 1. Kiểm tra confidence
            if conf < min_conf_cigarette:
                if debug:
                    print(f"   ❌ Cigarette #{i}: Confidence quá thấp ({conf:.2f} < {min_conf_cigarette})")
                continue
            
            # 2. Tính aspect ratio và kích thước
            width = float(x2 - x1)
            height = float(y2 - y1)
            area = width * height
            
            # Aspect ratio (luôn > 1)
            aspect_ratio = max(width, height) / min(width, height)
            
            # 3. Kiểm tra aspect ratio
            if aspect_ratio < min_aspect_ratio:
                if debug:
                    print(f"   ❌ Cigarette #{i}: Aspect ratio quá nhỏ ({aspect_ratio:.2f} < {min_aspect_ratio}) - có thể là vật tròn")
                continue
            
            if aspect_ratio > max_aspect_ratio:
                if debug:
                    print(f"   ❌ Cigarette #{i}: Aspect ratio quá lớn ({aspect_ratio:.2f} > {max_aspect_ratio}) - có thể là que dài/dây")
                continue
            
            # 4. Kiểm tra kích thước
            if area < min_area:
                if debug:
                    print(f"   ❌ Cigarette #{i}: Diện tích quá nhỏ ({area:.0f}px < {min_area}px) - noise")
                continue
            
            if area > max_area:
                if debug:
                    print(f"   ❌ Cigarette #{i}: Diện tích quá lớn ({area:.0f}px > {max_area}px) - không phải cigarette")
                continue
            
            # 5. Kiểm tra khoảng cách đến person gần nhất
            if len(person_boxes) > 0:
                cig_center_x = (x1 + x2) / 2
                cig_center_y = (y1 + y2) / 2
                
                min_distance = float('inf')
                for person_box in person_boxes:
                    px1, py1, px2, py2 = person_box
                    person_center_x = (px1 + px2) / 2
                    person_center_y = (py1 + py2) / 2
                    
                    distance = ((cig_center_x - person_center_x)**2 + 
                               (cig_center_y - person_center_y)**2)**0.5
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_distance_to_person:
                    if debug:
                        print(f"   ❌ Cigarette #{i}: Quá xa người ({min_distance:.0f}px > {max_distance_to_person}px)")
                    continue
                
                if debug:
                    print(f"   ✅ Cigarette #{i}: Hợp lệ (conf={conf:.2f}, ratio={aspect_ratio:.2f}, area={area:.0f}px, dist={min_distance:.0f}px)")
            else:
                # Không có person → loại bỏ cigarette
                if debug:
                    print(f"   ❌ Cigarette #{i}: Không có người trong ảnh")
                continue
            
            filtered_indices.append(i)
            filtered_cigarettes += 1
    
    if debug and cigarette_count > 0:
        print(f"\n   📊 Lọc cigarettes: {filtered_cigarettes}/{cigarette_count} giữ lại ({cigarette_count - filtered_cigarettes} loại bỏ)")
    
    # Tạo results mới với chỉ filtered boxes
    if len(filtered_indices) > 0:
        import torch
        filtered_boxes = boxes[filtered_indices]
        results[0].boxes = filtered_boxes
    else:
        # Không có boxes nào pass filter
        results[0].boxes = None
    
    return results


def get_recommended_thresholds(image_size):
    """
    Đề xuất thresholds dựa trên kích thước ảnh
    
    Args:
        image_size: tuple (width, height)
    
    Returns:
        dict với recommended thresholds
    """
    width, height = image_size
    total_pixels = width * height
    
    # Thresholds scale với kích thước ảnh
    if total_pixels < 640*480:  # VGA
        return {
            'min_conf_cigarette': 0.35,
            'min_aspect_ratio': 2.0,
            'max_aspect_ratio': 7.0,
            'min_area': 30,
            'max_area': 3000,
            'max_distance_to_person': 200
        }
    elif total_pixels < 1920*1080:  # HD
        return {
            'min_conf_cigarette': 0.30,
            'min_aspect_ratio': 1.8,
            'max_aspect_ratio': 7.5,
            'min_area': 50,
            'max_area': 4000,
            'max_distance_to_person': 250
        }
    else:  # Full HD+
        return {
            'min_conf_cigarette': 0.28,
            'min_aspect_ratio': 1.5,
            'max_aspect_ratio': 8.0,
            'min_area': 80,
            'max_area': 5000,
            'max_distance_to_person': 300
        }
