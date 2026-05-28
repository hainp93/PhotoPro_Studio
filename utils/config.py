"""
Config Manager — lưu/load cài đặt người dùng bằng JSON.
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "user_config.json"
PRESETS_DIR = Path(__file__).parent.parent / "presets"


@dataclass
class AppConfig:
    # Window
    window_width: int = 1400
    window_height: int = 900
    theme: str = "dark"          # "dark" | "light"
    accent_color: str = "#2196F3"

    # Output
    output_dir: str = ""         # empty = same folder as input
    output_suffix: str = "_enhanced"
    output_format: str = "PNG"
    output_quality: int = 95

    # Last used
    last_input_dir: str = ""
    last_output_dir: str = ""

    # Processing defaults (sync với PipelineSettings)
    denoise_enabled: bool = True
    denoise_strength: float = 5.0
    denoise_color_strength: float = 5.0

    upscale_enabled: bool = True
    upscale_factor: int = 2
    upscale_model: str = "realesrgan-x4plus"

    sharpen_enabled: bool = True
    sharpen_amount: float = 1.0
    sharpen_radius: float = 1.0
    sharpen_threshold: int = 3

    face_restore_enabled: bool = False
    face_restore_fidelity: float = 0.5
    face_restore_model: str = "codeformer"

    # Batch
    batch_create_subfolder: bool = True
    batch_subfolder_name: str = "enhanced"

    # UI state
    sidebar_collapsed: bool = False
    preview_split_pos: float = 0.5


class ConfigManager:
    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        self._config = AppConfig()
        self.load()

    @property
    def config(self) -> AppConfig:
        return self._config

    def load(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data: dict = json.load(f)
                for key, val in data.items():
                    if hasattr(self._config, key):
                        setattr(self._config, key, val)
                logger.debug("Config loaded.")
            except Exception as e:
                logger.warning(f"Config load failed: {e}, using defaults.")

    def save(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self._config), f, indent=2, ensure_ascii=False)
            logger.debug("Config saved.")
        except Exception as e:
            logger.error(f"Config save failed: {e}")

    # ── Presets ────────────────────────────────────────────────────────
    def save_preset(self, name: str, settings: dict):
        path = PRESETS_DIR / f"{name}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"name": name, "settings": settings}, f,
                          indent=2, ensure_ascii=False)
            logger.info(f"Preset saved: {name}")
        except Exception as e:
            logger.error(f"Preset save failed: {e}")

    def load_preset(self, name: str) -> dict | None:
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("settings", {})
        except Exception as e:
            logger.error(f"Preset load failed: {e}")
            return None

    def list_presets(self) -> list[str]:
        return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))

    def delete_preset(self, name: str):
        path = PRESETS_DIR / f"{name}.json"
        if path.exists():
            path.unlink()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._config, item)


# Singleton
_config_manager: ConfigManager | None = None


def get_config() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
