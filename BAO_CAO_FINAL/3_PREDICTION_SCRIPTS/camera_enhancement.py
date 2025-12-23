"""
Module cải thiện chất lượng ảnh từ camera kém
Áp dụng các kỹ thuật xử lý ảnh để cải thiện detection trong điều kiện xấu
"""

import cv2
import numpy as np

class CameraEnhancer:
    """
    Cải thiện chất lượng frame từ camera
    Xử lý: Low light, Blur, Noise, Low contrast
    """
    
    def __init__(self, 
                 auto_enhance=True,
                 denoise=True,
                 sharpen=True,
                 clahe=True,
                 auto_wb=True):
        """
        Args:
            auto_enhance: Tự động detect và enhance
            denoise: Giảm noise
            sharpen: Tăng độ sắc nét
            clahe: CLAHE cho low contrast
            auto_wb: Auto white balance
        """
        self.auto_enhance = auto_enhance
        self.denoise = denoise
        self.sharpen = sharpen
        self.clahe = clahe
        self.auto_wb = auto_wb
        
        # CLAHE object
        if self.clahe:
            self.clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Sharpening kernel
        self.sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        # Statistics
        self.frame_brightness_history = []
        self.frame_blur_history = []
    
    def enhance(self, frame):
        """
        Main enhancement function
        
        Args:
            frame: BGR image from camera
        Returns:
            Enhanced frame
        """
        if not self.auto_enhance:
            return frame
        
        original = frame.copy()
        
        # 1. Detect image quality issues
        brightness = self._get_brightness(frame)
        blur_score = self._get_blur_score(frame)
        contrast = self._get_contrast(frame)
        
        # Store history
        self.frame_brightness_history.append(brightness)
        self.frame_blur_history.append(blur_score)
        if len(self.frame_brightness_history) > 30:
            self.frame_brightness_history.pop(0)
            self.frame_blur_history.pop(0)
        
        # 2. Apply enhancements based on quality
        
        # Low light enhancement
        if brightness < 80:
            frame = self._enhance_brightness(frame)
        
        # High brightness (overexposed)
        elif brightness > 180:
            frame = self._reduce_brightness(frame)
        
        # Low contrast enhancement
        if contrast < 40 and self.clahe:
            frame = self._enhance_contrast(frame)
        
        # Denoising for noisy frames
        if self.denoise and brightness < 100:
            frame = self._denoise(frame)
        
        # Sharpening for blurry frames
        if self.sharpen and blur_score < 100:
            frame = self._sharpen(frame)
        
        # Auto white balance
        if self.auto_wb:
            frame = self._auto_white_balance(frame)
        
        return frame
    
    def _get_brightness(self, frame):
        """Tính độ sáng trung bình"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _get_blur_score(self, frame):
        """
        Tính blur score (Laplacian variance)
        Score càng cao = càng sharp
        Score < 100 = blurry
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _get_contrast(self, frame):
        """Tính contrast (standard deviation)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def _enhance_brightness(self, frame):
        """
        Tăng độ sáng cho low light
        Dùng Gamma Correction
        """
        # Gamma correction (gamma < 1 = brighten)
        gamma = 1.5  # Tăng độ sáng
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(frame, table)
    
    def _reduce_brightness(self, frame):
        """Giảm độ sáng cho overexposed"""
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255 
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(frame, table)
    
    def _enhance_contrast(self, frame):
        """
        Tăng contrast với CLAHE
        (Contrast Limited Adaptive Histogram Equalization)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l = self.clahe_obj.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _denoise(self, frame):
        """
        Giảm noise với fastNlMeansDenoisingColored
        Tốt cho low light, high ISO
        """
        # h: Filter strength (10 = medium)
        # hColor: Color filter strength
        return cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
    
    def _sharpen(self, frame):
        """
        Tăng độ sắc nét với unsharp mask
        """
        # Gaussian blur
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        
        # Unsharp mask: original + (original - blurred) * amount
        amount = 1.5
        sharpened = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        
        return sharpened
    
    def _auto_white_balance(self, frame):
        """
        Auto white balance - Gray World Algorithm
        """
        result = frame.copy()
        
        # Calculate mean for each channel
        b, g, r = cv2.split(result)
        
        b_avg = np.mean(b)
        g_avg = np.mean(g)
        r_avg = np.mean(r)
        
        # Gray world assumption: average should be gray
        avg = (b_avg + g_avg + r_avg) / 3
        
        # Scale each channel
        b = np.clip(b * (avg / b_avg), 0, 255).astype(np.uint8)
        g = np.clip(g * (avg / g_avg), 0, 255).astype(np.uint8)
        r = np.clip(r * (avg / r_avg), 0, 255).astype(np.uint8)
        
        return cv2.merge([b, g, r])
    
    def get_quality_info(self, frame):
        """
        Lấy thông tin chất lượng frame
        
        Returns:
            dict: Quality metrics
        """
        brightness = self._get_brightness(frame)
        blur_score = self._get_blur_score(frame)
        contrast = self._get_contrast(frame)
        
        # Quality assessment
        quality = "Good"
        issues = []
        
        if brightness < 80:
            quality = "Poor"
            issues.append("Too Dark")
        elif brightness > 180:
            quality = "Fair"
            issues.append("Overexposed")
        
        if blur_score < 100:
            quality = "Poor" if quality == "Good" else quality
            issues.append("Blurry")
        
        if contrast < 40:
            quality = "Fair" if quality == "Good" else quality
            issues.append("Low Contrast")
        
        return {
            'brightness': round(brightness, 1),
            'blur_score': round(blur_score, 1),
            'contrast': round(contrast, 1),
            'quality': quality,
            'issues': issues
        }


class AdaptiveConfidenceAdjuster:
    """
    Điều chỉnh confidence threshold dựa trên điều kiện camera
    """
    
    def __init__(self, base_conf=0.25):
        self.base_conf = base_conf
        self.brightness_history = []
    
    def get_adaptive_conf(self, frame):
        """
        Tính confidence threshold tối ưu cho frame
        
        Low light → Lower confidence (chấp nhận nhiều detections hơn)
        Good light → Higher confidence (strict hơn)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > 30:
            self.brightness_history.pop(0)
        
        # Adaptive adjustment
        if brightness < 80:  # Very dark
            conf = max(0.15, self.base_conf - 0.10)
        elif brightness < 120:  # Dark
            conf = max(0.20, self.base_conf - 0.05)
        elif brightness > 180:  # Overexposed
            conf = min(0.35, self.base_conf + 0.05)
        else:  # Normal
            conf = self.base_conf
        
        return conf


# Utility functions
def check_camera_quality(cap):
    """
    Kiểm tra chất lượng camera trước khi chạy detection
    
    Args:
        cap: cv2.VideoCapture object
    Returns:
        dict: Camera quality info
    """
    ret, frame = cap.read()
    if not ret:
        return None
    
    enhancer = CameraEnhancer()
    quality_info = enhancer.get_quality_info(frame)
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    quality_info.update({
        'resolution': f"{width}x{height}",
        'fps': fps
    })
    
    return quality_info


def recommend_camera_settings(quality_info):
    """
    Đề xuất cài đặt camera dựa trên quality
    
    Args:
        quality_info: dict from check_camera_quality()
    Returns:
        dict: Recommended settings
    """
    recommendations = []
    
    if quality_info['brightness'] < 80:
        recommendations.append("⚠️ Tăng ánh sáng môi trường")
        recommendations.append("💡 Hoặc dùng camera có ISO cao hơn")
        recommendations.append("🔧 Tăng exposure compensation")
    
    if quality_info['brightness'] > 180:
        recommendations.append("⚠️ Giảm ánh sáng hoặc đổi góc camera")
        recommendations.append("🔧 Giảm exposure compensation")
    
    if quality_info['blur_score'] < 100:
        recommendations.append("⚠️ Camera bị mờ - kiểm tra focus")
        recommendations.append("🔧 Dùng camera có autofocus")
        recommendations.append("📐 Tăng khoảng cách hoặc đổi lens")
    
    if quality_info['contrast'] < 40:
        recommendations.append("⚠️ Contrast thấp")
        recommendations.append("🔧 Cải thiện lighting setup")
    
    if not recommendations:
        recommendations.append("✅ Chất lượng camera tốt!")
    
    return recommendations


if __name__ == "__main__":
    # Test enhancement
    cap = cv2.VideoCapture(0)
    
    # Check quality
    quality = check_camera_quality(cap)
    if quality:
        print("📊 Camera Quality:")
        print(f"   Resolution: {quality['resolution']}")
        print(f"   Brightness: {quality['brightness']}")
        print(f"   Blur Score: {quality['blur_score']}")
        print(f"   Contrast: {quality['contrast']}")
        print(f"   Quality: {quality['quality']}")
        if quality['issues']:
            print(f"   Issues: {', '.join(quality['issues'])}")
        
        print("\n💡 Recommendations:")
        for rec in recommend_camera_settings(quality):
            print(f"   {rec}")
    
    # Test enhancement
    enhancer = CameraEnhancer()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Original vs Enhanced
        enhanced = enhancer.enhance(frame)
        
        # Show side by side
        combined = np.hstack([frame, enhanced])
        cv2.imshow('Original (Left) vs Enhanced (Right)', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
