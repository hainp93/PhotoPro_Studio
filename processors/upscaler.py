"""
Upscaler — Real-ESRGAN based image super-resolution.
KHÔNG thay đổi khuôn mặt — giữ nguyên đường nét nhân vật.
"""
import cv2
import numpy as np
import os
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "realesrgan"

MODEL_URLS = {
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
    },
    "realesrgan-x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "scale": 2,
        "num_block": 23,
    },
    "realesrgan-x4plus-anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
    },
}


class Upscaler:
    """
    Real-ESRGAN upscaler.
    - Lazy-load model (chỉ load khi dùng)
    - Tự động FP16 trên GPU hỗ trợ
    - Tiling cho ảnh lớn, tránh OOM
    """

    def __init__(self):
        self._upsampler = None
        self._loaded_model: str = ""
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_model(self, model_name: str, tile: int = 0):
        if self._loaded_model == model_name:
            return  # already loaded

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
        except ImportError:
            raise ImportError(
                "Thư viện 'basicsr' chưa được cài đặt.\n"
                "Cài đặt bằng lệnh:\n"
                "  pip install basicsr realesrgan"
            )
        from core.gpu_detector import get_gpu_info

        gpu = get_gpu_info()
        device = gpu.get_torch_device()

        cfg = MODEL_URLS.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}. Choices: {list(MODEL_URLS)}")

        # Download weight nếu chưa có
        model_path = str(WEIGHTS_DIR / cfg["filename"])
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model_name}...")
            load_file_from_url(cfg["url"], model_dir=str(WEIGHTS_DIR), progress=True)

        # Build RRDBNet
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=cfg["num_block"],
            num_grow_ch=32,
            scale=cfg["scale"],
        )

        # Half precision
        half = gpu.supports_fp16 and gpu.has_cuda

        # Tile size: arg > gpu recommendation
        tile_size = tile if tile != 0 else gpu.recommended_tile_size()

        try:
            from realesrgan import RealESRGANer
        except ImportError:
            try:
                from basicsr.utils.realesrgan_utils import RealESRGANer
            except ImportError:
                raise ImportError(
                    "Thư viện 'realesrgan' chưa được cài đặt.\n"
                    "Cài đặt bằng lệnh:\n"
                    "  pip install realesrgan"
                )

        self._upsampler = RealESRGANer(
            scale=cfg["scale"],
            model_path=model_path,
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device,
        )
        self._loaded_model = model_name
        logger.info(f"Upscaler loaded: {model_name} | tile={tile_size} | half={half}")

    def process(
        self,
        image: np.ndarray,
        scale: int = 2,
        model_name: str = "realesrgan-x4plus",
        tile: int = 0,
        progress_cb=None,
    ) -> np.ndarray:
        """
        image: BGR uint8 numpy array
        scale: 2 or 4 (nếu model là x4 nhưng chọn 2, sẽ resize về sau)
        Trả về: BGR uint8 numpy array đã upscale
        """
        # Chọn model phù hợp với scale
        if scale == 2 and "x4" in model_name and "anime" not in model_name:
            model_name = "realesrgan-x2plus"

        self._load_model(model_name, tile)

        try:
            # enhance() trả về (output, img_mode)
            output, _ = self._upsampler.enhance(image, outscale=scale)
            return output
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM! Retrying with tiling=256...")
                torch.cuda.empty_cache()
                self._loaded_model = ""  # force reload with tile
                self._load_model(model_name, tile=256)
                output, _ = self._upsampler.enhance(image, outscale=scale)
                return output
            raise

    def unload(self):
        self._upsampler = None
        self._loaded_model = ""
        torch.cuda.empty_cache()
        logger.info("Upscaler unloaded.")
