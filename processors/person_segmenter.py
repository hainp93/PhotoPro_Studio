import cv2
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

class PersonSegmenter:
    """
    Sử dụng YOLOv8-seg để phân tách cơ thể người ra khỏi nền.
    Dùng Singleton pattern hoặc module level để chỉ load model 1 lần.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PersonSegmenter, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            # Load model phân đoạn nhẹ nhất của YOLOv8 (yolov8n-seg.pt)
            # Ultralytics sẽ tự động tải model về thư mục bộ nhớ cache nếu chưa có.
            logger.info("Đang tải model YOLOv8 Segmentation...")
            self._model = YOLO('yolov8n-seg.pt')
            logger.info("✅ YOLOv8-seg loaded thành công.")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo YOLOv8-seg: {e}")
            self._model = None

    def get_person_mask(self, image: np.ndarray, feather_amount: int = 21) -> np.ndarray:
        """
        Nhận vào ảnh RGB/BGR (np.ndarray).
        Trả về mask cơ thể người (float32, giá trị từ 0.0 đến 1.0).
        Kích thước mask bằng đúng kích thước ảnh gốc.
        """
        if self._model is None:
            logger.warning("YOLOv8-seg không khả dụng. Trả về mask toàn trắng (apply cho toàn ảnh).")
            return np.ones(image.shape[:2], dtype=np.float32)

        h, w = image.shape[:2]
        
        # Chạy inference trên ảnh, chỉ lấy class 0 (person), giữ nguyên kích thước
        # retina_masks=True để mask có độ phân giải cao, bám sát viền.
        results = self._model.predict(
            source=image,
            classes=[0],           # Chỉ lấy person
            retina_masks=True,     # Mask high resolution
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Tạo mask trống (màu đen)
        combined_mask = np.zeros((h, w), dtype=np.float32)

        # Trích xuất và gộp các mask của từng người
        result = results[0]
        if result.masks is not None:
            # result.masks.data có shape (N, H, W)
            masks_data = result.masks.data.cpu().numpy()
            
            # Nếu mask có kích thước khác ảnh gốc, resize lại (Dù retina_masks=True thường đã chuẩn)
            for i in range(masks_data.shape[0]):
                m = masks_data[i]
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                # Gộp mask (logical OR tương đương np.maximum cho float 0-1)
                combined_mask = np.maximum(combined_mask, m)
                
        # Làm mềm viền mask (feathering) để trộn ảnh tự nhiên hơn
        if feather_amount > 0:
            # Đảm bảo feather_amount là số lẻ
            ksize = feather_amount if feather_amount % 2 == 1 else feather_amount + 1
            combined_mask = cv2.GaussianBlur(combined_mask, (ksize, ksize), 0)
            
        return combined_mask
