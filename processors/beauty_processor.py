import cv2
import numpy as np
import logging
import torch
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)

class BeautyProcessor:
    def __init__(self):
        self.pose_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_pose_model(self):
        if self.pose_model is None and YOLO is not None:
            logger.info("Loading YOLOv8-pose model for BeautyProcessor...")
            try:
                # Dùng yolov8s-pose.pt (Small) - chính xác hơn yolov8n (Nano)
                self.pose_model = YOLO("yolov8s-pose.pt")
                self.pose_model.to(self._device)
            except Exception as e:
                logger.error(f"Failed to load YOLOv8-pose: {e}")

    def apply_skin_retouch(self, img: np.ndarray, mask: np.ndarray, smooth_strength: float, tone_strength: float) -> np.ndarray:
        """
        Mịn da và trắng hồng.
        smooth_strength: 0-100 (được map sang thông số d, sigmaColor, sigmaSpace của bilateralFilter)
        tone_strength: 0-100 (tăng sáng và ám hồng)
        mask: float32 numpy array (0-1) từ PersonSegmenter.
        """
        if smooth_strength <= 0 and tone_strength <= 0:
            return img

        # Mịn da bằng Bilateral Filter
        smoothed = img
        if smooth_strength > 0:
            # Map 0-100 to d=9, sigma=10-150
            sigma = max(10, (smooth_strength / 100.0) * 150)
            smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=sigma)
            
        # Trắng hồng
        if tone_strength > 0:
            # Chuyển sang LAB để tăng sáng kênh L và tăng A (hồng/đỏ)
            lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            
            # Tăng sáng (L)
            l = l + (tone_strength / 100.0) * 20.0
            
            # Tăng hồng (A)
            a = a + (tone_strength / 100.0) * 10.0
            
            l = np.clip(l, 0, 255)
            a = np.clip(a, 0, 255)
            b = np.clip(b, 0, 255)
            
            lab_new = cv2.merge([l, a, b]).astype(np.uint8)
            smoothed = cv2.cvtColor(lab_new, cv2.COLOR_LAB2BGR)
            
        # Tạo mask da (Skin Mask) dùng không gian màu YCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask_img = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Chuyển về 0-1 và làm mềm viền
        skin_mask_img = skin_mask_img.astype(np.float32) / 255.0
        skin_mask_img = cv2.GaussianBlur(skin_mask_img, (5, 5), 0)
        
        # Kết hợp Mask người (tránh nhận nhầm bối cảnh) và Mask da
        final_mask = mask * skin_mask_img
            
        # Blend lại phần mịn da/trắng hồng chỉ lên vùng da của người
        mask_3d = np.repeat(final_mask[:, :, np.newaxis], 3, axis=2)
        result = (img * (1.0 - mask_3d) + smoothed * mask_3d).astype(np.uint8)
        return result

    def apply_body_slim(self, img: np.ndarray, strength: float) -> np.ndarray:
        """
        Thon gọn cơ thể và khuôn mặt.
        strength: 0-100
        """
        if strength <= 0 or YOLO is None:
            return img
            
        self._load_pose_model()
        if self.pose_model is None:
            return img

        # Nhận diện để lấy vị trí center của người
        # Tăng imgsz và giảm conf để nhận diện tốt hơn trên ảnh độ phân giải siêu cao (20MP+)
        results = self.pose_model(img, verbose=False, imgsz=1280, conf=0.15)
        bodies = []
        faces = []
        
        if len(results) > 0:
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    h = y2 - y1
                    bodies.append((cx, cy, h))
                    
            if results[0].keypoints is not None:
                kpts = results[0].keypoints.xy.cpu().numpy()
                for p in kpts:
                    if len(p) >= 5:
                        nose = p[0]
                        l_ear = p[3]
                        r_ear = p[4]
                        if nose[1] > 0 and l_ear[1] > 0 and r_ear[1] > 0:
                            # Tính độ rộng khuôn mặt dựa trên khoảng cách 2 tai
                            face_w = abs(l_ear[0] - r_ear[0]) * 1.5 
                            if face_w > 10:
                                faces.append((nose[0], nose[1], face_w))
                
        if not bodies and not faces:
            return img
            
        # Tối ưu hóa: Tạo grid 1 lần và tích lũy độ dời (displacement)
        h, w = img.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        map_x = x_coords.astype(np.float32).copy()
        map_y = y_coords.astype(np.float32).copy()
        
        # 1. Thon gọn mặt (Face Slimming)
        for cx, cy, f_width in faces:
            radius = f_width * 0.8
            slim_factor = (strength / 100.0) * 0.25 # max 25% squeeze for face
            
            dx = x_coords - cx
            dy = y_coords - cy
            dist = np.sqrt(dx**2 + dy**2)
            
            roi_mask = dist < radius
            factor = np.zeros_like(dist, dtype=np.float32)
            factor[roi_mask] = (1 - (dist[roi_mask] / radius)) ** 2
            
            map_x += dx * slim_factor * factor
            
        # 2. Thon gọn cơ thể (Body Slimming)
        for cx, cy, h_box in bodies:
            radius = h_box * 0.5 
            # Tăng độ móp body lên 0.6 (trước đó là 0.3 hơi nhẹ)
            slim_factor = (strength / 100.0) * 0.6 
            
            dx = x_coords - cx
            dy = y_coords - cy
            dist = np.sqrt(dx**2 + dy**2)
            
            roi_mask = dist < radius
            factor = np.zeros_like(dist, dtype=np.float32)
            factor[roi_mask] = (1 - (dist[roi_mask] / radius)) ** 2
            
            map_x += dx * slim_factor * factor
            
        # Thực hiện warp 1 lần duy nhất cho toàn bộ ảnh
        res = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return res

    def apply_leg_stretch(self, img: np.ndarray, stretch_pct: float) -> np.ndarray:
        """
        Kéo dài chân bằng cách dãn nửa dưới cơ thể theo trục dọc.
        Làm thay đổi kích thước ảnh (ảnh sẽ cao hơn).
        stretch_pct: 0-100 (100 = giãn 20%)
        """
        if stretch_pct <= 0 or YOLO is None:
            return img
            
        self._load_pose_model()
        if self.pose_model is None:
            return img
            
        results = self.pose_model(img, verbose=False)
        hips_y = []
        
        # Tìm vị trí hông của những người trong ảnh (keypoints 11, 12)
        if len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy.cpu().numpy()
            for p in kpts:
                if len(p) >= 13: # Cần ít nhất tới điểm 12
                    left_hip = p[11]
                    right_hip = p[12]
                    # Nếu có tọa độ y > 0
                    if left_hip[1] > 0 and right_hip[1] > 0:
                        hips_y.append( (left_hip[1] + right_hip[1]) / 2.0 )
                    elif left_hip[1] > 0:
                        hips_y.append(left_hip[1])
                    elif right_hip[1] > 0:
                        hips_y.append(right_hip[1])
                        
        h, w = img.shape[:2]
        if not hips_y:
            # Fallback nếu không thấy hông: cắt ở 60% chiều cao
            split_y = int(h * 0.6)
        else:
            # Lấy vị trí hông thấp nhất (gần dưới cùng nhất) hoặc trung bình
            split_y = int(np.mean(hips_y))
            
        # Ràng buộc split_y
        split_y = max(int(h*0.2), min(int(h*0.8), split_y))
        
        # Cắt ảnh
        top = img[0:split_y, :]
        bottom = img[split_y:h, :]
        
        # Stretch bottom
        max_stretch_factor = 0.2 # 20% max
        factor = 1.0 + (stretch_pct / 100.0) * max_stretch_factor
        new_bottom_h = int(bottom.shape[0] * factor)
        
        bottom_stretched = cv2.resize(bottom, (w, new_bottom_h), interpolation=cv2.INTER_CUBIC)
        
        # Ghép lại
        res = np.vstack([top, bottom_stretched])
        return res

    def detect_bodies(self, img: np.ndarray) -> list:
        """
        Quét ảnh và trả về danh sách bounding boxes của các cơ thể.
        Trả về list các list [x1, y1, x2, y2].
        """
        if YOLO is None:
            return []
            
        self._load_pose_model()
        if self.pose_model is None:
            return []
            
        results = self.pose_model(img, verbose=False, imgsz=1280, conf=0.15)
        bboxes = []
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        logger.info(f"YOLO detect_bodies: ảnh {img.shape[1]}x{img.shape[0]} -> tìm thấy {len(bboxes)} người")
        return bboxes
