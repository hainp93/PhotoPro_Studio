"""
RAW Image Reader — đọc file RAW từ máy ảnh (NEF, CR2, ARW, DNG...).
Output: BGR uint8 numpy array sẵn sàng cho pipeline.
"""
import numpy as np
import cv2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf",
    ".rw2", ".orf", ".pef", ".srw", ".x3f", ".3fr",
    ".erf", ".mef", ".mrw", ".nrw", ".rwl",
}


def is_raw_file(path: str) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


class RawReader:
    """
    Đọc file RAW và convert sang BGR uint8.
    Hỗ trợ demosaic, white balance tự động, highlight recovery.
    rawpy được import lazy — chỉ khi thực sự đọc file RAW.
    """

    def read(
        self,
        path: str,
        use_camera_wb: bool = True,
        highlight_mode: int = 0,
        output_bps: int = 8,
    ) -> np.ndarray:
        """
        path           : đường dẫn file RAW
        use_camera_wb  : dùng white balance từ metadata máy ảnh
        highlight_mode : 0=clip, 1=unclip, 2=blend, 3-9=rebuild
        output_bps     : 8 hoặc 16 bit
        Trả về: BGR uint8 (hoặc uint16 nếu output_bps=16)
        """
        try:
            import rawpy  # lazy import — tránh crash khi rawpy chưa cài
        except ImportError:
            raise ImportError(
                "Thư viện 'rawpy' chưa được cài đặt.\n"
                "Cài đặt bằng lệnh: pip install rawpy"
            )

        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=use_camera_wb,
                use_auto_wb=not use_camera_wb,
                highlight_mode=rawpy.HighlightMode(highlight_mode),
                output_bps=output_bps,
                no_auto_bright=False,
                gamma=(2.222, 4.5),   # sRGB gamma
            )

        # rawpy trả về RGB, convert sang BGR cho OpenCV pipeline
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 16-bit → 8-bit nếu cần
        if output_bps == 16:
            bgr = (bgr / 256).astype(np.uint8)

        logger.info(f"RAW read: {path} → {bgr.shape} {bgr.dtype}")
        return bgr

    def get_metadata(self, path: str) -> dict:
        """Đọc metadata cơ bản của file RAW."""
        try:
            import rawpy  # lazy import
        except ImportError:
            logger.warning("rawpy not installed, cannot read RAW metadata")
            return {}

        try:
            with rawpy.imread(path) as raw:
                return {
                    "camera": raw.camera_whitebalance,
                    "daylight_wb": raw.daylight_whitebalance,
                    "color_matrix": raw.color_matrix.tolist() if raw.color_matrix is not None else None,
                    "num_colors": raw.num_colors,
                    "sizes": {
                        "raw_width": raw.sizes.raw_width,
                        "raw_height": raw.sizes.raw_height,
                        "iwidth": raw.sizes.iwidth,
                        "iheight": raw.sizes.iheight,
                    },
                }
        except Exception as e:
            logger.warning(f"Could not read RAW metadata: {e}")
            return {}
