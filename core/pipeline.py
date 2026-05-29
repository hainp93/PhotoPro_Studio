"""
Processing Pipeline — quản lý chuỗi xử lý ảnh.
Mỗi bước có thể bật/tắt độc lập.
"""
import numpy as np
import logging
import threading
from typing import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PipelineSettings:
    # --- Beauty & Body ---
    beauty_enabled: bool = False
    skin_smooth: float = 0.0               # 0-100
    skin_tone: float = 0.0                 # 0-100 (trắng hồng)
    body_slim: float = 0.0                 # 0-100
    leg_stretch: float = 0.0               # 0-100

    # --- Denoise ---
    denoise_enabled: bool = False          # Tắt mặc định — bật khi cần giảm noise
    denoise_strength: float = 5.0          # 0–20
    denoise_color_strength: float = 5.0

    # --- Upscale ---
    upscale_enabled: bool = False          # Tắt mặc định — cần cài basicsr+realesrgan
    upscale_factor: int = 2                # 2 or 4
    upscale_model: str = "realesrgan-x4plus"   # model key
    upscale_tile: int = 0                  # 0 = auto
    upscale_max_long_side: int = 0         # 0 = không giới hạn, > 0 = giới hạn cạnh dài (px)

    # --- Sharpen ---
    sharpen_enabled: bool = True
    sharpen_ai_enabled: bool = False       # AI sharpen dùng Real-ESRGAN
    sharpen_ai_strength: float = 0.85      # blend strength
    sharpen_ai_model: str = "realesrgan-x4plus"
    sharpen_method: str = "Bilateral"      # "Bilateral" | "USM"
    sharpen_amount: float = 1.0            # 0.0–2.0 (LAB bilateral hiệu quả hơn)
    sharpen_radius: float = 1.0            # pixels
    sharpen_threshold: int = 3             # 0–10
    sharpen_person_only: bool = True       # Chỉ làm nét cơ thể người bằng AI mask

    # --- Face Restore ---
    face_restore_enabled: bool = True      # Bật mặc định để cứu nét
    face_restore_fidelity: float = 0.8     # 0=AI, 1=original
    face_restore_model: str = "codeformer" # "codeformer" | "gfpgan"
    face_restore_upsample: bool = False    # Ảnh Fullframe đã đủ lớn, không cần upsample 2x
    face_restore_high_res: bool = True     # Quét toàn bộ ảnh (không thu nhỏ)

    # --- Export ---
    export_format: str = "PNG"             # PNG, JPG, WEBP, TIFF
    export_quality: int = 95              # JPEG/WEBP quality
    export_suffix: str = "_enhanced"


class Pipeline:
    """
    Chuỗi xử lý ảnh tuần tự.
    Mỗi processor được lazy-load để tránh tốn RAM khi không dùng.
    """

    def __init__(self):
        self._settings = PipelineSettings()
        self._processors: dict = {}
        self._lock = threading.Lock()

    @property
    def settings(self) -> PipelineSettings:
        return self._settings

    @settings.setter
    def settings(self, s: PipelineSettings):
        self._settings = s

    def _get_processor(self, name: str):
        """Lazy-load processor theo tên."""
        if name not in self._processors:
            if name == "denoiser":
                from processors.denoiser import Denoiser
                self._processors[name] = Denoiser()
            elif name == "upscaler":
                from processors.upscaler import Upscaler
                self._processors[name] = Upscaler()
            elif name == "sharpener":
                from processors.sharpener import Sharpener
                self._processors[name] = Sharpener()
            elif name == "face_restorer":
                from processors.face_restorer import FaceRestorer
                self._processors[name] = FaceRestorer()
            elif name == "person_segmenter":
                from processors.person_segmenter import PersonSegmenter
                self._processors[name] = PersonSegmenter()
            elif name == "beauty_processor":
                from processors.beauty_processor import BeautyProcessor
                self._processors[name] = BeautyProcessor()
        return self._processors[name]

    def process(
        self,
        image: np.ndarray,
        settings: PipelineSettings = None,
        progress_cb: Callable[[float, str], None] = None,
        cancel_flag: threading.Event = None,
        manual_bboxes: list = None,
    ) -> np.ndarray:
        """
        Xử lý ảnh qua toàn bộ pipeline.
        image: numpy array BGR uint8
        manual_bboxes: list of (x, y, w, h) to force face restoration
        Trả về: numpy array BGR uint8
        """
        s = settings or self._settings
        if progress_cb is None:
            progress_cb = lambda pct, msg: None
        if cancel_flag is None:
            cancel_flag = threading.Event()

        result = image.copy()
        steps_total = sum([
            s.beauty_enabled,
            s.denoise_enabled,
            s.upscale_enabled,
            s.sharpen_enabled,
            s.face_restore_enabled,
        ])
        step = 0
        warnings = []

        def _progress(msg):
            nonlocal step
            pct = (step / max(steps_total, 1)) * 100
            progress_cb(pct, msg)
            step += 1

        # ── Step 0: Beauty & Body Shaping ────────────────────────────
        if s.beauty_enabled and not cancel_flag.is_set():
            if s.skin_smooth > 0 or s.skin_tone > 0 or s.body_slim > 0 or s.leg_stretch > 0:
                _progress("Đang áp dụng Làm đẹp & Nắn dáng...")
                logger.debug("Pipeline: Beauty & Body Shaping")
                try:
                    proc = self._get_processor("beauty_processor")
                    
                    # 1. Thon gọn
                    if s.body_slim > 0:
                        result = proc.apply_body_slim(result, s.body_slim)
                    
                    # 2. Kéo dài chân
                    if s.leg_stretch > 0:
                        result = proc.apply_leg_stretch(result, s.leg_stretch)
                        
                    # 3. Mịn da / Trắng hồng
                    if s.skin_smooth > 0 or s.skin_tone > 0:
                        seg_proc = self._get_processor("person_segmenter")
                        mask = seg_proc.get_person_mask(result, feather_amount=21)
                        result = proc.apply_skin_retouch(result, mask, s.skin_smooth, s.skin_tone)
                except Exception as e:
                    msg = f"Làm đẹp thất bại: {e}"
                    logger.warning(msg)
                    warnings.append(msg)

        # ── Step 1: Denoise ──────────────────────────────────────────
        if s.denoise_enabled and not cancel_flag.is_set():
            _progress("Đang khử noise...")
            logger.debug("Pipeline: Denoise")
            try:
                proc = self._get_processor("denoiser")
                result = proc.process(
                    result,
                    luminance_strength=s.denoise_strength,
                    color_strength=s.denoise_color_strength,
                )
            except Exception as e:
                msg = f"Khử noise thất bại: {e}"
                logger.warning(msg)
                warnings.append(msg)

        # ── Step 2: Upscale ──────────────────────────────────────────
        if s.upscale_enabled and not cancel_flag.is_set():
            _progress(f"Đang upscale {s.upscale_factor}x...")
            logger.debug(f"Pipeline: Upscale {s.upscale_factor}x model={s.upscale_model}")
            try:
                proc = self._get_processor("upscaler")
                result = proc.process(
                    result,
                    scale=s.upscale_factor,
                    model_name=s.upscale_model,
                    tile=s.upscale_tile,
                )
                # Giới hạn cạnh dài sau upscale nếu được cấu hình
                max_ls = getattr(s, "upscale_max_long_side", 0)
                if max_ls > 0:
                    h_r, w_r = result.shape[:2]
                    long_side = max(h_r, w_r)
                    if long_side > max_ls:
                        ratio = max_ls / long_side
                        new_w = int(w_r * ratio)
                        new_h = int(h_r * ratio)
                        _progress(f"Resize về {max_ls}px...")
                        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                        logger.info(f"Upscale xong, resize: {w_r}x{h_r} → {new_w}x{new_h}")
            except Exception as e:
                msg = f"Upscale thất bại: {e}"
                logger.warning(msg)
                warnings.append(msg)

        # ── Step 3: Sharpen ──────────────────────────────────────────
        if s.sharpen_enabled and not cancel_flag.is_set():
            proc = self._get_processor("sharpener")
            sharpened_result = result.copy()
            
            if s.sharpen_ai_enabled:
                _progress("Đang làm nét AI (Real-ESRGAN)...")
                logger.debug(f"Pipeline: AI Sharpen model={s.sharpen_ai_model}")
                try:
                    sharpened_result = proc.process_ai(
                        sharpened_result,
                        model_name=s.sharpen_ai_model,
                        tile=s.upscale_tile,
                        strength=s.sharpen_ai_strength,
                    )
                except Exception as e:
                    logger.warning(f"AI Sharpen thất bại, fallback classical: {e}")
                    warnings.append(f"AI Sharpen fallback: {e}")
                    try:
                        sharpened_result = proc.process(
                            sharpened_result,
                            amount=s.sharpen_amount,
                            radius=s.sharpen_radius,
                            threshold=s.sharpen_threshold,
                        )
                    except Exception as e2:
                        warnings.append(f"Làm nét thất bại: {e2}")
            else:
                _progress("Đang làm nét...")
                logger.debug(f"Pipeline: {s.sharpen_method} Sharpen")
                try:
                    if s.sharpen_method == "USM":
                        sharpened_result = proc.process_usm(
                            sharpened_result,
                            amount=s.sharpen_amount,
                            radius=s.sharpen_radius,
                            threshold=s.sharpen_threshold,
                        )
                    else:  # Bilateral (default)
                        sharpened_result = proc.process(
                            sharpened_result,
                            amount=s.sharpen_amount,
                            radius=s.sharpen_radius,
                            threshold=s.sharpen_threshold,
                        )
                except Exception as e:
                    warnings.append(f"Làm nét thất bại: {e}")
            
            # Blend dựa trên Person Mask
            if s.sharpen_person_only:
                _progress("Đang bóc tách cơ thể (AI Person Mask)...")
                logger.debug("Pipeline: Áp dụng Person Masking cho Sharpening")
                try:
                    seg_proc = self._get_processor("person_segmenter")
                    # Lấy mask (numpy float32, từ 0 đến 1)
                    mask = seg_proc.get_person_mask(result, feather_amount=21)
                    # Mở rộng mask thành 3 kênh để nhân với ảnh BGR
                    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    
                    # Trộn ảnh: Giữ background gốc, lấy foreground đã làm nét
                    result = (result * (1.0 - mask_3d) + sharpened_result * mask_3d).astype(np.uint8)
                except Exception as e:
                    logger.warning(f"Person Masking thất bại, áp dụng toàn ảnh: {e}")
                    warnings.append(f"Masking lỗi: {e}")
                    result = sharpened_result
            else:
                result = sharpened_result

        # ── Step 4: Face Restore (optional) ─────────────────────────
        if s.face_restore_enabled and not cancel_flag.is_set():
            _progress("Đang phục hồi khuôn mặt (AI)...")
            logger.debug(f"Pipeline: Face Restore model={s.face_restore_model}")
            try:
                proc = self._get_processor("face_restorer")
                proc.setup_image(
                    result, 
                    upsample=s.face_restore_upsample, 
                    bg_upscale=False
                )
                
                # Quét AI
                proc.detect_faces(high_res=s.face_restore_high_res)
                
                # Cắt và lấy landmarks cho các mặt thủ công
                if manual_bboxes:
                    _progress("Đang xử lý khuôn mặt thủ công...")
                    for bbox in manual_bboxes:
                        proc.add_manual_face(bbox)
                
                result = proc.restore_faces(fidelity=s.face_restore_fidelity)
                proc.unload()
            except Exception as e:
                msg = f"Face Restore thất bại: {e}"
                logger.warning(msg)
                warnings.append(msg)

        if warnings:
            warn_summary = "\n".join(f"⚠ {w}" for w in warnings)
            progress_cb(100.0, f"Xong (có cảnh báo)")
            logger.warning(f"Pipeline hoàn thành với {len(warnings)} cảnh báo:\n{warn_summary}")
        else:
            progress_cb(100.0, "Hoàn thành!")
        return result

    def unload_models(self):
        """Giải phóng VRAM khi không dùng."""
        import torch
        for proc in self._processors.values():
            if hasattr(proc, "unload"):
                proc.unload()
        self._processors.clear()
        torch.cuda.empty_cache()
        logger.info("All models unloaded.")


# Singleton pipeline
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
