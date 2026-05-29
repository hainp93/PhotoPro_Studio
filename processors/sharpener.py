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
        Luminance-only Bilateral Sharpening — kỹ thuật chuyên nghiệp.

        1. Chuyển sang LAB color space
        2. Chỉ sharpen kênh L (Luminance = độ sáng/chi tiết)
        3. Giữ nguyên a, b (màu sắc) → không color shift, không color fringing
        4. Dùng bilateral filter nhẹ (conservative) — không plasticky

        Ưu điểm:
          - Không thay đổi màu sắc ảnh
          - Không tạo halo
          - Mịn, tự nhiên — đúng cách các tool chuyên nghiệp làm
        """
        if amount <= 0:
            return image

        # Chuyển sang LAB — sharpen trên L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        l_f = l_ch.astype(np.float32)

        # Bilateral filter trên L — conservative params
        sigma_color = 20        # chỉ blend pixel rất giống nhau → giữ cạnh sắc
        sigma_space = max(8, int(radius * 12))
        d = max(5, int(radius * 6) | 1)

        base = cv2.bilateralFilter(l_ch, d=d,
                                   sigmaColor=sigma_color,
                                   sigmaSpace=sigma_space).astype(np.float32)

        # Detail layer = high-frequency luminance
        detail = l_f - base

        # Threshold: bỏ qua noise ISO nhỏ
        if threshold > 0:
            detail = np.where(np.abs(detail) > threshold, detail, 0)

        # Boost (gentle — bilateral đã hiệu quả hơn USM nên amount nhỏ hơn)
        l_sharp = np.clip(l_f + amount * detail, 0, 255).astype(np.uint8)

        # Ghép lại — màu không đổi
        result_lab = cv2.merge([l_sharp, a_ch, b_ch])
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def process_usm(
        self,
        image: np.ndarray,
        amount: float = 1.0,
        radius: float = 1.0,
        threshold: int = 3,
    ) -> np.ndarray:
        """
        Unsharp Mask — cũng chỉ sharpen kênh L trong LAB.
        Dùng BORDER_REFLECT để tránh artifact rìa ảnh.
        """
        if amount <= 0:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        l_f = l_ch.astype(np.float32)
        h, w = l_f.shape
        ksize = max(3, int(radius * 6) | 1)
        pad = ksize

        padded = cv2.copyMakeBorder(
            l_f, pad, pad, pad, pad, cv2.BORDER_REFLECT_101
        )
        blurred = cv2.GaussianBlur(padded, (ksize, ksize), radius)[pad:pad+h, pad:pad+w]

        detail = l_f - blurred
        if threshold > 0:
            detail = np.where(np.abs(detail) > threshold, detail, 0)

        l_sharp = np.clip(l_f + amount * detail, 0, 255).astype(np.uint8)
        result_lab = cv2.merge([l_sharp, a_ch, b_ch])
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

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
        """Entry point từ pipeline (classical)."""
        if use_wavelet:
            return self.process_wavelet(image, amount=amount)
        return self.process(image, amount=amount, radius=radius, threshold=threshold)

    def process_ai(
        self,
        image: np.ndarray,
        model_name: str = "realesrgan-x4plus",
        tile: int = 0,
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        AI làm nét bằng Real-ESRGAN:
          1. Upscale 4x bằng AI model
          2. Resize về kích thước gốc bằng LANCZOS
          → Kết quả sắc nét hơn, ít noise hơn, giữ nguyên kích thước.
        
        strength: 0.0–1.0 blend với ảnh gốc (1.0 = full AI, 0.5 = 50/50)
        """
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch
        except ImportError as e:
            logger.warning(f"AI sharpen requires realesrgan+basicsr: {e}. Falling back to classical.")
            return self.process(image, amount=2.0)

        orig_h, orig_w = image.shape[:2]
        logger.info(f"AI Sharpen: {orig_w}×{orig_h} → upscale 4x → resize về gốc | model={model_name}")

        try:
            # Tìm model weights
            from pathlib import Path
            import os

            model_dirs = [
                Path(__file__).parent.parent / "models",
                Path(__file__).parent.parent / "weights",
                Path(os.environ.get("REALESRGAN_WEIGHTS", "")),
            ]
            model_path = None
            model_filename = {
                "realesrgan-x4plus": "RealESRGAN_x4plus.pth",
                "realesrgan-x4plus-anime": "RealESRGAN_x4plus_anime_6B.pth",
                "realesrgan-x2plus": "RealESRGAN_x2plus.pth",
            }.get(model_name, "RealESRGAN_x4plus.pth")

            for d in model_dirs:
                p = d / model_filename
                if p.exists():
                    model_path = str(p)
                    break

            if not model_path:
                raise FileNotFoundError(
                    f"Model '{model_filename}' không tìm thấy. "
                    f"Tải về từ: https://github.com/xinntao/Real-ESRGAN/releases"
                )

            # Chọn model architecture
            is_anime = "anime" in model_name
            num_block = 6 if is_anime else 23
            scale = 2 if "x2" in model_name else 4

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=num_block, num_grow_ch=32, scale=scale,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
                device=device,
            )

            # Upscale 4x bằng AI
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output_rgb, _ = upsampler.enhance(rgb, outscale=scale)
            upscaled = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

            # Resize về kích thước gốc
            ai_result = cv2.resize(upscaled, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            # Blend với ảnh gốc theo strength
            if strength < 1.0:
                alpha = np.clip(strength, 0.0, 1.0)
                ai_result = cv2.addWeighted(
                    ai_result, alpha,
                    image.astype(ai_result.dtype), 1.0 - alpha,
                    0
                ).astype(np.uint8)

            logger.info("AI Sharpen hoàn thành")
            return ai_result

        except Exception as e:
            logger.error(f"AI Sharpen lỗi: {e}")
            raise
