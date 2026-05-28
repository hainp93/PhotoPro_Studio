"""
Denoiser — Khử noise ảnh, hỗ trợ Luminance và Color noise riêng biệt.
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Denoiser:
    """
    Khử noise bằng OpenCV NLM (Non-Local Means) — nhanh, chất lượng tốt.
    Tách riêng luminance noise và color noise để giữ màu sắc tự nhiên.
    """

    def process(
        self,
        image: np.ndarray,
        luminance_strength: float = 5.0,
        color_strength: float = 5.0,
    ) -> np.ndarray:
        """
        image: BGR uint8
        luminance_strength: 0–20 (strength cho noise độ sáng)
        color_strength    : 0–20 (strength cho noise màu)
        """
        if luminance_strength <= 0 and color_strength <= 0:
            return image

        # Chuyển sang LAB để xử lý L và AB riêng
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Denoise luminance channel (L) — giữ detail tốt hơn
        if luminance_strength > 0:
            h = int(luminance_strength)
            l_denoised = cv2.fastNlMeansDenoising(
                l_channel,
                h=h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            l_denoised = l_channel

        # Denoise color channels (A, B) — giảm color noise
        if color_strength > 0:
            hColor = int(color_strength)
            ab_combined = np.stack([a_channel, b_channel], axis=2)
            # Dùng colorfastNlMeans trên BGR → convert trick
            bgr_color = cv2.cvtColor(image, cv2.COLOR_BGR2BGR)  # same
            ab_denoised_bgr = cv2.fastNlMeansDenoisingColored(
                image,
                h=1,          # lum minimal
                hColor=hColor,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            lab_ab = cv2.cvtColor(ab_denoised_bgr, cv2.COLOR_BGR2LAB)
            _, a_denoised, b_denoised = cv2.split(lab_ab)
        else:
            a_denoised, b_denoised = a_channel, b_channel

        # Merge lại
        lab_result = cv2.merge([l_denoised, a_denoised, b_denoised])
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        return result

    def process_fast(
        self,
        image: np.ndarray,
        strength: float = 5.0,
    ) -> np.ndarray:
        """Nhanh hơn nhưng ít control hơn — dùng cho preview."""
        if strength <= 0:
            return image
        h = int(strength)
        return cv2.fastNlMeansDenoisingColored(
            image, h=h, hColor=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def process_gaussian(
        self,
        image: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Gaussian blur nhẹ — cực nhanh cho preview realtime."""
        if strength <= 0:
            return image
        ksize = max(3, int(strength * 2) | 1)
        return cv2.GaussianBlur(image, (ksize, ksize), strength * 0.5)
