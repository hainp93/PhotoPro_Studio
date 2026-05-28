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
    # --- Denoise ---
    denoise_enabled: bool = True
    denoise_strength: float = 5.0          # 0–20
    denoise_color_strength: float = 5.0

    # --- Upscale ---
    upscale_enabled: bool = True
    upscale_factor: int = 2                # 2 or 4
    upscale_model: str = "realesrgan-x4plus"   # model key
    upscale_tile: int = 0                  # 0 = auto

    # --- Sharpen ---
    sharpen_enabled: bool = True
    sharpen_amount: float = 1.0            # 0.0–3.0
    sharpen_radius: float = 1.0            # pixels
    sharpen_threshold: int = 3             # 0–10

    # --- Face Restore ---
    face_restore_enabled: bool = False
    face_restore_fidelity: float = 0.5     # 0=AI, 1=original
    face_restore_model: str = "codeformer" # "codeformer" | "gfpgan"
    face_restore_upsample: bool = True

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
        return self._processors[name]

    def process(
        self,
        image: np.ndarray,
        settings: PipelineSettings = None,
        progress_cb: Callable[[float, str], None] = None,
        cancel_flag: threading.Event = None,
    ) -> np.ndarray:
        """
        Xử lý ảnh qua toàn bộ pipeline.
        image: numpy array BGR uint8
        Trả về: numpy array BGR uint8
        """
        s = settings or self._settings
        if progress_cb is None:
            progress_cb = lambda pct, msg: None
        if cancel_flag is None:
            cancel_flag = threading.Event()

        result = image.copy()
        steps_total = sum([
            s.denoise_enabled,
            s.upscale_enabled,
            s.sharpen_enabled,
            s.face_restore_enabled,
        ])
        step = 0

        def _progress(msg):
            nonlocal step
            pct = (step / max(steps_total, 1)) * 100
            progress_cb(pct, msg)
            step += 1

        # ── Step 1: Denoise ──────────────────────────────────────────
        if s.denoise_enabled and not cancel_flag.is_set():
            _progress("Đang khử noise...")
            logger.debug("Pipeline: Denoise")
            proc = self._get_processor("denoiser")
            result = proc.process(
                result,
                luminance_strength=s.denoise_strength,
                color_strength=s.denoise_color_strength,
            )

        # ── Step 2: Upscale ──────────────────────────────────────────
        if s.upscale_enabled and not cancel_flag.is_set():
            _progress(f"Đang upscale {s.upscale_factor}x...")
            logger.debug(f"Pipeline: Upscale {s.upscale_factor}x model={s.upscale_model}")
            proc = self._get_processor("upscaler")
            result = proc.process(
                result,
                scale=s.upscale_factor,
                model_name=s.upscale_model,
                tile=s.upscale_tile,
            )

        # ── Step 3: Sharpen ──────────────────────────────────────────
        if s.sharpen_enabled and not cancel_flag.is_set():
            _progress("Đang làm nét...")
            logger.debug("Pipeline: Sharpen")
            proc = self._get_processor("sharpener")
            result = proc.process(
                result,
                amount=s.sharpen_amount,
                radius=s.sharpen_radius,
                threshold=s.sharpen_threshold,
            )

        # ── Step 4: Face Restore (optional) ─────────────────────────
        if s.face_restore_enabled and not cancel_flag.is_set():
            _progress("Đang phục hồi khuôn mặt (AI)...")
            logger.debug(f"Pipeline: Face Restore model={s.face_restore_model}")
            proc = self._get_processor("face_restorer")
            result = proc.process(
                result,
                fidelity=s.face_restore_fidelity,
                model_name=s.face_restore_model,
                upsample=s.face_restore_upsample,
            )

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
