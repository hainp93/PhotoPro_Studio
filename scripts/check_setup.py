"""
check_setup.py — Kiểm tra nhanh môi trường PhotoPro Studio trên VPS.
Chạy: python scripts/check_setup.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(label, fn):
    try:
        msg = fn()
        results.append((OK, label, msg or ""))
        return True
    except Exception as e:
        results.append((FAIL, label, str(e)[:80]))
        return False


print("=" * 60)
print("PhotoPro Studio - Environment Check")
print("=" * 60)

# Python
check("Python", lambda: f"{sys.version.split()[0]}")

# PyTorch
def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU"
    return f"torch {torch.__version__} | CUDA={cuda} | {device}"
check("PyTorch + CUDA", check_torch)

# FP16
def check_fp16():
    import torch
    if not torch.cuda.is_available():
        return "N/A (CPU)"
    cap = torch.cuda.get_device_capability(0)
    fp16_ok = cap[0] >= 7
    return f"Compute {cap[0]}.{cap[1]} | FP16={'YES' if fp16_ok else 'NO'}"
check("GPU FP16", check_fp16)

# OpenCV
check("OpenCV", lambda: __import__("cv2").__version__)

# basicsr
check("basicsr", lambda: __import__("basicsr").__version__ if hasattr(__import__("basicsr"), "__version__") else "installed")

# realesrgan_utils (từ basicsr)
def check_esrgan():
    from basicsr.utils.realesrgan_utils import RealESRGANer
    return "basicsr.utils.realesrgan_utils OK"
check("RealESRGAN utils", check_esrgan)

# facelib
check("facelib", lambda: __import__("facelib") and "installed")

# CodeFormer arch
def check_codeformer():
    from basicsr.utils.registry import ARCH_REGISTRY
    cf = ARCH_REGISTRY.get("CodeFormer")
    if cf is None:
        raise ImportError("CodeFormer not in ARCH_REGISTRY — clone & install CodeFormer repo")
    return "ARCH_REGISTRY['CodeFormer'] found"
check("CodeFormer arch", check_codeformer)

# Model weights
print()
print("Model Weights:")
MODELS = [
    ("CodeFormer", "codeformer.pth", 360),
    ("realesrgan", "RealESRGAN_x2plus.pth", 60),
    ("facelib", "detection_Resnet50_Final.pth", 100),
    ("facelib", "parsing_parsenet.pth", 80),
]
weights_ok = True
for d, f, min_mb in MODELS:
    p = WEIGHTS / d / f
    if p.exists():
        mb = p.stat().st_size // 1024 // 1024
        if mb < min_mb // 2:
            print(f"  {WARN} {d}/{f} ({mb}MB, too small?)")
        else:
            print(f"  {OK}   {mb:>4}MB  {d}/{f}")
    else:
        print(f"  {FAIL} MISSING: {d}/{f}")
        weights_ok = False

# Summary
print()
print("=" * 60)
print("Summary:")
for status, label, msg in results:
    print(f"  {status} {label}" + (f": {msg}" if msg else ""))

print()
all_ok = all(s == OK for s, _, _ in results) and weights_ok
if all_ok:
    print("All OK! Face Restore + AI Sharpen ready.")
    print("Make sure 'Bat Face Restore' is ON in the app UI.")
else:
    print("Some issues found. Fix then re-run.")
    fails = [l for s,l,_ in results if s == FAIL]
    if "CodeFormer arch" in fails:
        print()
        print("  To fix CodeFormer:")
        print("  git clone https://github.com/sczhou/CodeFormer.git")
        print("  cd CodeFormer && pip install -e . --no-deps && cd ..")
    if not weights_ok:
        print()
        print("  To fix weights:")
        print("  python scripts/setup_models.py")
print("=" * 60)
