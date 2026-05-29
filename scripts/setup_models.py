"""
setup_models.py — Thiết lập model weights cho PhotoPro Studio.

2 cách:
  1. Copy từ Wedding Beauty Studio (nếu đã cài trên máy)
  2. Download từ internet

Chạy: python scripts/setup_models.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"

# Đường dẫn Wedding Beauty Studio trên máy này
WEDDING_WEIGHTS = Path(
    r"f:\Setup\Wedding Beauty Studio V9.5\Wedding Beauty Studio V9.5\VGA\weights"
)

# Mapping: (nguồn từ Wedding, đích trong PhotoPro)
COPY_MAP = {
    "CodeFormer/codeformer.pth":                ("CodeFormer", "codeformer.pth"),
    "realesrgan/RealESRGAN_x2plus.pth":         ("realesrgan", "RealESRGAN_x2plus.pth"),
    "detection/detection_Resnet50_Final.pth":   ("facelib", "detection_Resnet50_Final.pth"),
    "facelib/parsing_parsenet.pth":             ("facelib", "parsing_parsenet.pth"),
}

# URLs để download nếu không có Wedding app
DOWNLOAD_URLS = {
    ("CodeFormer", "codeformer.pth"):
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    ("realesrgan", "RealESRGAN_x2plus.pth"):
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
    ("facelib", "detection_Resnet50_Final.pth"):
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    ("facelib", "parsing_parsenet.pth"):
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
}


def copy_from_wedding():
    if not WEDDING_WEIGHTS.exists():
        print(f"❌ Không tìm thấy Wedding Beauty Studio tại: {WEDDING_WEIGHTS}")
        return False

    print(f"✅ Tìm thấy Wedding Beauty Studio: {WEDDING_WEIGHTS}")
    all_ok = True

    for src_rel, (dst_dir, dst_name) in COPY_MAP.items():
        src = WEDDING_WEIGHTS / src_rel
        dst_folder = WEIGHTS / dst_dir
        dst = dst_folder / dst_name

        if dst.exists():
            print(f"  ⏭  {dst_name} đã có ({dst.stat().st_size // 1024 // 1024}MB)")
            continue

        if not src.exists():
            print(f"  ❌ Không tìm thấy nguồn: {src}")
            all_ok = False
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)
        print(f"  📋 Copy {src.name} ({src.stat().st_size // 1024 // 1024}MB)...", end="", flush=True)
        shutil.copy2(src, dst)
        print(" ✅")

    return all_ok


def download_models():
    print("\n📥 Download model từ internet...")
    try:
        from basicsr.utils.download_util import load_file_from_url
    except ImportError:
        print("❌ basicsr chưa cài. Chạy: pip install basicsr")
        return False

    all_ok = True
    for (dst_dir, dst_name), url in DOWNLOAD_URLS.items():
        dst_folder = WEIGHTS / dst_dir
        dst = dst_folder / dst_name

        if dst.exists():
            print(f"  ⏭  {dst_name} đã có ({dst.stat().st_size // 1024 // 1024}MB)")
            continue

        dst_folder.mkdir(parents=True, exist_ok=True)
        print(f"  ⬇  Download {dst_name}...")
        try:
            load_file_from_url(url, model_dir=str(dst_folder), progress=True)
            print(f"  ✅ {dst_name}")
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
            all_ok = False

    return all_ok


def verify():
    print("\n📋 Kiểm tra model:")
    all_ok = True
    for (dst_dir, dst_name) in DOWNLOAD_URLS.keys():
        p = WEIGHTS / dst_dir / dst_name
        exists = p.exists()
        size_mb = p.stat().st_size // 1024 // 1024 if exists else 0
        status = f"✅ {size_mb}MB" if exists else "❌ THIẾU"
        print(f"  {status}  {dst_dir}/{dst_name}")
        if not exists:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("PhotoPro Studio — Model Setup")
    print("=" * 60)

    # Ưu tiên copy từ Wedding app
    copied = copy_from_wedding()

    if not copied:
        print("\n⬇  Thử download từ internet...")
        download_models()

    all_ready = verify()

    if all_ready:
        print("\n🎉 Tất cả model đã sẵn sàng! Khởi động app: python main.py")
    else:
        print("\n⚠  Một số model còn thiếu. App vẫn chạy nhưng Face Restore và Upscale sẽ bị tắt.")

    sys.exit(0 if all_ready else 1)
