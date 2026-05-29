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

    def _get_person_mask(self, img: np.ndarray) -> np.ndarray:
        """Lấy person mask (float32 0-1) dùng PersonSegmenter."""
        try:
            from processors.person_segmenter import PersonSegmenter
            mask = PersonSegmenter().get_person_mask(img, feather_amount=31)
            return mask
        except Exception as e:
            logger.warning(f"PersonSegmenter failed: {e}. Dùng mask toàn trắng.")
            return np.ones(img.shape[:2], dtype=np.float32)

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
            
        h, w = img.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Accumulate displacement (dx, dy) — chưa add vào map
        disp_x = np.zeros((h, w), dtype=np.float32)
        
        # 1. Thon gọn mặt (Face Slimming)
        for cx, cy, f_width in faces:
            radius = f_width * 0.8
            slim_factor = (strength / 100.0) * 0.25  # max 25% squeeze for face
            
            dx = x_coords - cx
            dy = y_coords - cy
            dist = np.sqrt(dx**2 + dy**2)
            
            roi_mask = dist < radius
            factor = np.zeros_like(dist, dtype=np.float32)
            factor[roi_mask] = (1 - (dist[roi_mask] / radius)) ** 2
            
            disp_x += dx * slim_factor * factor
            
        # 2. Thon gọn cơ thể (Body Slimming)
        for (cx, cy, h_box), box_info in zip(bodies, results[0].boxes.xyxy.cpu().numpy()):
            bx1, by1, bx2, by2 = box_info
            radius = h_box * 0.5
            base_factor = (strength / 100.0) * 0.6
            
            # Giảm dần cường độ slim cho người ở gần rìa ảnh
            # Người bị cắt ở rìa → mask thiếu → artifact → giảm effect
            edge_margin = radius * 0.5  # vùng an toàn = nửa bán kính
            left_dist  = max(0.0, bx1)           # khoảng cách từ mép trái bbox đến rìa trái
            right_dist = max(0.0, w - bx2)        # khoảng cách từ mép phải bbox đến rìa phải
            top_dist   = max(0.0, by1)
            min_edge_dist = min(left_dist, right_dist, top_dist)
            edge_fade = float(np.clip(min_edge_dist / max(edge_margin, 1.0), 0.0, 1.0))
            slim_factor = base_factor * edge_fade
            
            if slim_factor < 0.001:
                continue  # người quá sát rìa, bỏ qua hoàn toàn
            
            dx = x_coords - cx
            dy = y_coords - cy
            dist = np.sqrt(dx**2 + dy**2)
            
            roi_mask = dist < radius
            factor = np.zeros_like(dist, dtype=np.float32)
            factor[roi_mask] = (1 - (dist[roi_mask] / radius)) ** 2
            
            disp_x += dx * slim_factor * factor
        
        # ✅ Key fix: nhân displacement với person_mask TRƯỚC khi tạo warp map
        # → Background pixels có mask=0 → displacement=0 → không bị dịch chuyển
        person_mask = self._get_person_mask(img)
        
        # Erode mask ~15px để co vào trong cơ thể người TRƯỚC khi blur
        # → Vùng chuyển tiếp (feather) nằm BÊN TRONG người, không lan ra nền
        erode_px = max(3, int(min(img.shape[:2]) * 0.008))  # ~0.8% min(h,w), ~15px với ảnh 2K
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        person_mask_eroded = cv2.erode(person_mask, kernel, iterations=1)
        person_mask_blurred = cv2.GaussianBlur(person_mask_eroded, (31, 31), 0)
        
        # Áp displacement đã được mask vào warp map
        map_x = x_coords.astype(np.float32) + disp_x * person_mask_blurred
        map_y = y_coords.astype(np.float32)
        
        result = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result

    def apply_leg_stretch(self, img: np.ndarray, stretch_pct: float) -> np.ndarray:
        """
        Kéo dài chân bằng warp gradient.
        Chỉ dãn vùng từ hông xuống. Background giữ nguyên nhờ blend person mask.
        stretch_pct: 0-100 (100 = giãn 15%)
        """
        if stretch_pct <= 0 or YOLO is None:
            return img
            
        self._load_pose_model()
        if self.pose_model is None:
            return img
            
        results = self.pose_model(img, verbose=False, imgsz=1280, conf=0.15)
        hips_y = []
        
        if len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy.cpu().numpy()
            for p in kpts:
                if len(p) >= 13:
                    left_hip = p[11]
                    right_hip = p[12]
                    if left_hip[1] > 0 and right_hip[1] > 0:
                        hips_y.append((left_hip[1] + right_hip[1]) / 2.0)
                    elif left_hip[1] > 0:
                        hips_y.append(left_hip[1])
                    elif right_hip[1] > 0:
                        hips_y.append(right_hip[1])
                        
        h, w = img.shape[:2]
        hip_y = int(np.mean(hips_y)) if hips_y else int(h * 0.55)
        hip_y = max(int(h * 0.2), min(int(h * 0.80), hip_y))
        leg_len = h - hip_y
        
        max_stretch = 0.15  # Tối đa 15%
        stretch_factor = (stretch_pct / 100.0) * max_stretch
        max_disp = leg_len * stretch_factor
        
        # Warp map gradient: trên hip_y không đổi, dưới hip_y dãn dần
        y_coords_1d = np.arange(h, dtype=np.float32)
        map_y_1d = np.zeros(h, dtype=np.float32)
        
        for yi in range(h):
            if yi <= hip_y:
                map_y_1d[yi] = float(yi)
            else:
                t = min(1.0, (yi - hip_y) / leg_len)
                smooth_t = t * t * (3 - 2 * t)  # Smooth step
                map_y_1d[yi] = yi - smooth_t * max_disp
        
        map_y_2d = np.tile(map_y_1d[:, np.newaxis], (1, w)).astype(np.float32)
        map_x_2d = np.tile(np.arange(w, dtype=np.float32)[np.newaxis, :], (h, 1))
        
        warped = cv2.remap(img, map_x_2d, map_y_2d,
                           interpolation=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT)
        
        # ✅ Blend: chỉ áp dụng stretch lên vùng người (phần chân), background giữ nguyên
        person_mask = self._get_person_mask(img)
        leg_mask = person_mask.copy()
        leg_mask[:hip_y, :] = 0  # Phần trên hông không chỉnh
        
        # Mở rộng mask xuống dưới bàn chân: fill 1.0 trong bbox từ hông đến gầm giày
        # (anchor_vals = 0 tại by2 nên cách cũ không hiệu quả)
        if len(results) > 0 and results[0].boxes is not None:
            boxes_np = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes_np:
                bx1, by1, bx2, by2 = [int(v) for v in box]
                if by2 <= hip_y:
                    continue
                col_s = max(0, bx1)
                col_e = min(w, bx2)
                row_start = max(hip_y, by1)
                # Mở rộng xuống thêm 5% để phủ hết bàn chân
                row_end = min(h, by2 + int(h * 0.05))
                if col_e > col_s and row_end > row_start:
                    # Fill 1.0 toàn bbox: Gaussian blur sẽ tạo viền mịn tự nhiên
                    leg_mask[row_start:row_end, col_s:col_e] = np.maximum(
                        leg_mask[row_start:row_end, col_s:col_e], 1.0
                    )
        leg_mask = np.clip(leg_mask, 0, 1)
        
        leg_mask = cv2.GaussianBlur(leg_mask, (51, 51), 0)
        
        mask_3d = np.repeat(leg_mask[:, :, np.newaxis], 3, axis=2)
        result = (img.astype(np.float32) * (1.0 - mask_3d) + warped.astype(np.float32) * mask_3d).astype(np.uint8)
        return result

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
