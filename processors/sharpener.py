"""
Sharpener — Làm nét ảnh không dùng AI, giữ nguyên đường nét.
Phương pháp: Unsharp Mask + Wavelet Sharpening.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Sharpener:
    """
    Kết hợp 2 phương pháp làm nét:
    1. Unsharp Mask (nhanh, ổn định)
    2. High-frequency detail boost tùy chọn
    """

    def process(
        self,
        image: np.ndarray,
        amount: float = 1.0,
        radius: float = 1.0,
        threshold: int = 3,
    ) -> np.ndarray:
        """
        image: BGR uint8
        amount  : 0.0–3.0, mức độ làm nét
        radius  : sigma của Gaussian blur (pixels)
        threshold: ngưỡng pixel diff để apply sharpening (0–10)
        """
        if amount <= 0:
            return image

        # Làm việc trên float32 để tránh overflow
        img_f = image.astype(np.float32)

        # Unsharp Mask
        ksize = max(3, int(radius * 6) | 1)  # odd number
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), radius)
        detail = img_f - blurred

        # Threshold: chỉ sharpen pixel có detail đủ lớn
        if threshold > 0:
            mask = np.abs(detail) > threshold
            detail = np.where(mask, detail, 0)

        sharpened = img_f + amount * detail

        # Clip và convert
        result = np.clip(sharpened, 0, 255).astype(np.uint8)
        return result

    def process_wavelet(
        self,
        image: np.ndarray,
        amount: float = 1.0,
        levels: int = 2,
    ) -> np.ndarray:
        """
        Wavelet-based sharpening — tốt hơn cho chi tiết fine.
        Chỉ sharpen high-frequency band.
        """
        try:
            import pywt
        except ImportError:
            logger.warning("pywt not installed, falling back to Unsharp Mask")
            return self.process(image, amount=amount)

        img_f = image.astype(np.float32)
        channels = cv2.split(img_f)
        result_channels = []

        for ch in channels:
            # Multi-level wavelet decompose
            coeffs = pywt.wavedec2(ch, "db1", level=levels)
            # Boost detail coefficients
            boosted = [coeffs[0]]  # approximation unchanged
            for detail in coeffs[1:]:
                boosted.append(tuple(c * (1.0 + amount * 0.5) for c in detail))
            # Reconstruct
            reconstructed = pywt.waverec2(boosted, "db1")
            # Crop nếu shape thay đổi
            h, w = ch.shape
            reconstructed = reconstructed[:h, :w]
            result_channels.append(reconstructed)

        merged = cv2.merge(result_channels)
        return np.clip(merged, 0, 255).astype(np.uint8)

    def process_auto(
        self,
        image: np.ndarray,
        amount: float = 1.0,
        radius: float = 1.0,
        threshold: int = 3,
        use_wavelet: bool = False,
    ) -> np.ndarray:
        """Entry point từ pipeline."""
        if use_wavelet:
            return self.process_wavelet(image, amount=amount)
        return self.process(image, amount=amount, radius=radius, threshold=threshold)
