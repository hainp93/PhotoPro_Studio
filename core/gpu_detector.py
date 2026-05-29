"""
GPU Detection & Device Configuration
Tự động phát hiện GPU và cấu hình device phù hợp.
torch được import lazy — không crash khi chưa cài.
"""
import platform
import logging

logger = logging.getLogger(__name__)


class GPUInfo:
    def __init__(self):
        self.has_cuda = False
        self.device_name = "CPU"
        self.vram_gb = 0.0
        self.cuda_version = ""
        self.compute_capability = (0, 0)
        self.device_str = "cpu"
        self.supports_fp16 = False
        self.gpu_tier = "cpu"   # "cpu" | "low" | "mid" | "high"

        try:
            import torch  # lazy import — tránh crash khi torch chưa cài
            self.has_cuda = torch.cuda.is_available()
            if self.has_cuda:
                self._detect_cuda(torch)
        except ImportError:
            logger.warning("torch chưa được cài đặt — chạy ở chế độ CPU.")

    def _detect_cuda(self, torch):
        try:
            idx = torch.cuda.current_device()
            self.device_name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            self.vram_gb = props.total_memory / (1024 ** 3)
            self.cuda_version = torch.version.cuda or ""
            self.compute_capability = (props.major, props.minor)
            self.device_str = f"cuda:{idx}"

            # FP16 không ổn định trên GTX 1650/1660
            no_fp16_gpus = ["1650", "1660"]
            self.supports_fp16 = not any(g in self.device_name for g in no_fp16_gpus)

            # Phân loại tier
            name_upper = self.device_name.upper()
            if any(k in name_upper for k in ["RTX 50", "RTX 40", "RTX 30", "RTX 20",
                                              "TITAN", "QUADRO A",
                                              "GTX 1080", "GTX 1070", "GTX 1060", "GTX 1050"]):
                self.gpu_tier = "high"
            elif any(k in name_upper for k in ["GTX 1650", "GTX 1660", "QUADRO RTX",
                                                "QUADRO T", "QUADRO P"]):
                self.gpu_tier = "mid"
            else:
                self.gpu_tier = "low"

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self.has_cuda = False
            self.device_str = "cpu"

    def get_torch_device(self):
        try:
            import torch
            return torch.device(self.device_str)
        except ImportError:
            raise ImportError(
                "Thư viện 'torch' chưa được cài đặt.\n"
                "Cài đặt theo hướng dẫn: https://pytorch.org/get-started/locally/"
            )

    def recommended_tile_size(self) -> int:
        """Tile size cho Real-ESRGAN dựa trên VRAM."""
        if not self.has_cuda:
            return 256
        if self.vram_gb >= 12:
            return 0       # 0 = no tiling (full image)
        elif self.vram_gb >= 8:
            return 512
        elif self.vram_gb >= 6:
            return 400
        else:
            return 256

    def summary(self) -> str:
        if not self.has_cuda:
            try:
                import psutil
                ram = psutil.virtual_memory().total / (1024 ** 3)
                ram_str = f" | RAM: {ram:.1f}GB"
            except ImportError:
                ram_str = ""
            return f"CPU Mode{ram_str} | OS: {platform.system()}"
        return (
            f"GPU: {self.device_name} | "
            f"VRAM: {self.vram_gb:.1f}GB | "
            f"CUDA: {self.cuda_version} | "
            f"FP16: {'✓' if self.supports_fp16 else '✗'} | "
            f"Tier: {self.gpu_tier.upper()}"
        )

    def __repr__(self):
        return f"<GPUInfo device='{self.device_str}' tier='{self.gpu_tier}'>"


# Singleton — chỉ detect 1 lần khi import
_gpu_info: GPUInfo | None = None


def get_gpu_info() -> GPUInfo:
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = GPUInfo()
        logger.info(f"GPU detected: {_gpu_info.summary()}")
    return _gpu_info


def get_device():
    return get_gpu_info().get_torch_device()
