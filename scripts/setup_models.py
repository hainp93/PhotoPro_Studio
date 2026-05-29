"""
setup_models.py — Thiết lập model weights cho PhotoPro Studio.

Ưu tiên:
  1. Copy từ Wedding Beauty Studio nếu có trên máy
  2. Download từ GitHub (tất cả đều là open-source models)

Chạy: python scripts/setup_models.py
"""
import sys
import shutil
import urllib.request
import hashlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"

# Wedding Beauty Studio path (chỉ có trên máy dev)
WEDDING_WEIGHTS = Path(
    r"f:\Setup\Wedding Beauty Studio V9.5\Wedding Beauty Studio V9.5\VGA\weights"
)

# Tất cả models đều open-source, download từ GitHub
MODELS = {
    ("CodeFormer", "codeformer.pth"): {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "size_mb": 376,
        "wedding_src": "CodeFormer/codeformer.pth",
    },
    ("realesrgan", "RealESRGAN_x2plus.pth"): {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        "size_mb": 67,
        "wedding_src": "realesrgan/RealESRGAN_x2plus.pth",
    },
    ("facelib", "detection_Resnet50_Final.pth"): {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "size_mb": 109,
        "wedding_src": "detection/detection_Resnet50_Final.pth",
    },
    ("facelib", "parsing_parsenet.pth"): {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "size_mb": 85,
        "wedding_src": "facelib/parsing_parsenet.pth",
    },
}


class ProgressBar:
    def __init__(self, total_mb):
        self.total = total_mb * 1024 * 1024
        self.downloaded = 0

    def __call__(self, count, block_size, total_size):
        if total_size > 0:
            self.total = total_size
        self.downloaded += block_size
        pct = min(100, self.downloaded * 100 // self.total)
        bar = "=" * (pct // 5) + " " * (20 - pct // 5)
        mb = self.downloaded / 1024 / 1024
        total_mb = self.total / 1024 / 1024
        print(f"\r  [{bar}] {pct}% {mb:.1f}/{total_mb:.1f}MB", end="", flush=True)
        if self.downloaded >= self.total:
            print()


def try_copy_from_wedding(dst_dir, dst_name, wedding_src) -> bool:
    if not WEDDING_WEIGHTS.exists():
        return False
    src = WEDDING_WEIGHTS / wedding_src
    if not src.exists():
        return False
    dst = WEIGHTS / dst_dir / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [Wedding] Copy {dst_name} ({src.stat().st_size // 1024 // 1024}MB)...", end="")
    shutil.copy2(src, dst)
    print(" OK")
    return True


def download_model(dst_dir, dst_name, url, size_mb) -> bool:
    dst = WEIGHTS / dst_dir / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")

    print(f"  [Download] {dst_name} (~{size_mb}MB)")
    print(f"  URL: {url}")
    try:
        progress = ProgressBar(size_mb)
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        tmp.rename(dst)
        print(f"  OK -> {dst}")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def patch_codeformer_basicsr():
    """Tạo basicsr/version.py nếu thiếu — cần cho import basicsr.__init__."""
    cf_basicsr = ROOT / "CodeFormer_repo" / "basicsr"
    if not cf_basicsr.exists():
        return  # CodeFormer_repo chưa clone
    ver_file = cf_basicsr / "version.py"
    if not ver_file.exists():
        ver_file.write_text("__version__ = '1.4.2-codeformer'\n")
        print(f"  [PATCH] Created {ver_file}")
    else:
        print(f"  [SKIP]  basicsr/version.py already exists")


def setup_all():
    print("=" * 60)
    print("PhotoPro Studio - Model Setup")
    print("=" * 60)

    # Patch CodeFormer basicsr trước
    print("\nPatching CodeFormer basicsr...")
    patch_codeformer_basicsr()

    wedding_available = WEDDING_WEIGHTS.exists()
    if wedding_available:
        print(f"Wedding Beauty Studio found: {WEDDING_WEIGHTS}")
    else:
        print("Wedding Beauty Studio not found - will download from GitHub")

    print()
    all_ok = True

    for (dst_dir, dst_name), info in MODELS.items():
        dst = WEIGHTS / dst_dir / dst_name
        if dst.exists():
            mb = dst.stat().st_size // 1024 // 1024
            print(f"  [SKIP] {dst_name} already exists ({mb}MB)")
            continue

        print(f"\n  --> {dst_name}")

        # Try Wedding app first
        if wedding_available and try_copy_from_wedding(
            dst_dir, dst_name, info["wedding_src"]
        ):
            continue

        # Download from GitHub
        ok = download_model(dst_dir, dst_name, info["url"], info["size_mb"])
        if not ok:
            all_ok = False

    print()
    print("=" * 60)
    print("Verification:")
    for (dst_dir, dst_name) in MODELS.keys():
        p = WEIGHTS / dst_dir / dst_name
        if p.exists():
            mb = p.stat().st_size // 1024 // 1024
            print(f"  OK  {mb:>4}MB  {dst_dir}/{dst_name}")
        else:
            print(f"  MISSING       {dst_dir}/{dst_name}")
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("All models ready! Run: python main.py")
    else:
        print("Some models missing. Check internet connection and try again.")

    return all_ok


if __name__ == "__main__":
    ok = setup_all()
    sys.exit(0 if ok else 1)
