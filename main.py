"""
PhotoPro Studio — Entry Point
"""
import sys
import os
import logging
from pathlib import Path

# ── Fix path khi chạy từ PyInstaller ────────────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
    APP_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
    APP_DIR = BASE_DIR

# Thêm thư mục gốc vào path
sys.path.insert(0, str(BASE_DIR))

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = APP_DIR / "event_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("PhotoPro Studio starting...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"BASE_DIR: {BASE_DIR}")

    # GPU detection log
    try:
        from core.gpu_detector import get_gpu_info
        gpu = get_gpu_info()
        logger.info(f"GPU: {gpu.summary()}")
    except Exception as e:
        logger.warning(f"GPU detection error: {e}")

    # Launch UI
    from ui.app import PhotoProApp
    app = PhotoProApp()
    app.mainloop()

    logger.info("PhotoPro Studio closed.")


if __name__ == "__main__":
    main()
