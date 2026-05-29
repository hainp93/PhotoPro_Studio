"""
Image I/O — đọc và ghi ảnh đa định dạng.
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_READ = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
    ".bmp", ".gif",
    # RAW (handled separately via raw_reader)
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf",
    ".rw2", ".orf", ".pef", ".srw",
}

EXPORT_FORMATS = ["PNG", "JPEG", "WEBP", "TIFF", "BMP"]


def read_image(path: str) -> np.ndarray:
    """
    Đọc ảnh bất kỳ định dạng, trả về BGR uint8.
    Tự động xử lý RAW nếu cần.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    from processors.raw_reader import is_raw_file, RawReader
    if is_raw_file(path):
        return RawReader().read(path)

    # Dùng PIL để đảm bảo đọc được nhiều format (TIFF 16bit, etc.)
    try:
        pil_img = Image.open(path)
        
        # ✅ Tự động xoay ảnh theo metadata EXIF Orientation
        # (ảnh chụp từ điện thoại thường bị lưu ngang nhưng EXIF ghi "xoay 90°")
        from PIL import ImageOps
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # Convert sang RGB nếu cần
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode == "L":
            pil_img = pil_img.convert("RGB")
        elif pil_img.mode == "I;16":
            pil_img = pil_img.convert("I")
            arr = np.array(pil_img, dtype=np.uint16)
            arr = (arr / 256).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        arr = np.array(pil_img, dtype=np.uint8)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"PIL failed ({e}), trying cv2...")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img


def save_image(
    image: np.ndarray,
    path: str,
    fmt: str = "PNG",
    quality: int = 95,
) -> str:
    """
    Lưu ảnh BGR uint8.
    fmt: PNG | JPEG | WEBP | TIFF | BMP
    quality: cho JPEG/WEBP (1-100)
    Trả về: đường dẫn file đã lưu
    """
    p = Path(path)
    fmt = fmt.upper()

    # Đảm bảo extension đúng
    ext_map = {
        "PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg",
        "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp",
    }
    ext = ext_map.get(fmt, p.suffix)
    if p.suffix.lower() != ext:
        p = p.with_suffix(ext)

    p.parent.mkdir(parents=True, exist_ok=True)

    # Convert BGR → RGB cho PIL
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    save_kwargs = {}
    if fmt in ("JPEG", "JPG"):
        save_kwargs = {"quality": quality, "optimize": True, "progressive": True}
        pil_img.save(str(p), format="JPEG", **save_kwargs)
    elif fmt == "WEBP":
        save_kwargs = {"quality": quality, "method": 4}
        pil_img.save(str(p), format="WEBP", **save_kwargs)
    elif fmt == "TIFF":
        pil_img.save(str(p), format="TIFF", compression="lzw")
    elif fmt == "PNG":
        pil_img.save(str(p), format="PNG", optimize=True)
    else:
        pil_img.save(str(p))

    logger.info(f"Saved: {p} ({fmt})")
    return str(p)


def build_output_path(input_path: str, output_dir: str, suffix: str, fmt: str) -> str:
    """Tạo đường dẫn output từ input + suffix + format."""
    ext_map = {
        "PNG": ".png", "JPEG": ".jpg", "JPG": ".jpg",
        "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp",
    }
    stem = Path(input_path).stem
    ext = ext_map.get(fmt.upper(), ".png")
    return str(Path(output_dir) / f"{stem}{suffix}{ext}")


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Chuyển BGR numpy → PIL Image (RGB)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Chuyển PIL Image → BGR numpy."""
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
