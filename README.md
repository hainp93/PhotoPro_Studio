# PhotoPro Studio 🖼️

Phần mềm AI chuyên làm nét, upscale và làm đẹp ảnh. Kết hợp tính năng của Photo-SOS và Wedding Beauty Studio.

## Tính năng
- ✅ **Upscale** ảnh 2x/4x bằng Real-ESRGAN (không thay đổi khuôn mặt)
- ✅ **Làm nét** bằng Unsharp Mask + Wavelet Sharpening
- ✅ **Khử noise** thông minh (Luminance & Color)
- ✅ **Face Restore** (CodeFormer) — tùy chọn bật/tắt
- ✅ **Batch processing** xử lý hàng loạt
- ✅ **Hỗ trợ RAW** (NEF, CR2, ARW, DNG...)
- ✅ **Export** JPG / PNG / WEBP / TIFF
- ✅ **Preset** lưu/load cài đặt
- ✅ **Before/After** preview realtime

## Yêu cầu hệ thống (Production VM)
- GPU: RTX 5090 16GB (CUDA 12.8)
- RAM: 32GB+
- Python: 3.10

## Cài đặt (trên VM)

```bash
# 1. Clone repo
git clone <repo-url>
cd PhotoPro_Studio

# 2. Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Cài PyTorch CUDA 12.8 trước
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Cài các thư viện còn lại
pip install -r requirements.txt

# 5. Chạy
python main.py
```

## Build thành .exe

```bash
BUILD.bat
```

Output: `dist/PhotoPro_Studio/`

## Cấu trúc thư mục

```
PhotoPro_Studio/
├── main.py
├── core/           # GPU detector, pipeline, worker
├── processors/     # Upscaler, Sharpener, Denoiser, FaceRestorer, RAW
├── ui/             # CustomTkinter UI
├── utils/          # Image I/O, Config, Preview
├── presets/        # JSON preset files
├── sample_images/  # Ảnh test (tự thêm vào, không commit)
└── weights/        # AI model weights (tự download, không commit)
```

## Tác giả
Kenny Phạm — Internal Tool
