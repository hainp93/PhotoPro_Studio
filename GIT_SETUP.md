# PhotoPro Studio — Git & VM Setup Guide

## Bước 1: Khởi tạo Git trên máy này (local)

```powershell
cd E:\Tool\PhotoPro_Studio
git init
git add .
git commit -m "feat: initial project - PhotoPro Studio v1.0.0"
```

## Bước 2: Tạo repo trên GitHub

- Vào GitHub → New repository
- Đặt tên: `PhotoPro_Studio`
- Private ✅

```powershell
git remote add origin https://github.com/<YOUR_USERNAME>/PhotoPro_Studio.git
git branch -M main
git push -u origin main
```

## Bước 3: Cài đặt trên VM (RTX 5090)

```bash
# Clone
git clone https://github.com/<YOUR_USERNAME>/PhotoPro_Studio.git
cd PhotoPro_Studio

# Tạo venv
python -m venv venv
.\venv\Scripts\activate   # Windows

# 1. Cài PyTorch CUDA 12.8 TRƯỚC
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Cài các thư viện còn lại
pip install -r requirements.txt

# 3. Chạy thử
python main.py
```

## Workflow hàng ngày

**Máy local (bạn code/chỉnh):**
```powershell
git add .
git commit -m "feat: ..."
git push
```

**VM (chạy/test):**
```bash
git pull
python main.py
```

## Lưu ý

- File `weights/` **KHÔNG** được commit (quá nặng, tự download khi chạy)
- File `sample_images/*.jpg` KHÔNG commit
- File `config/user_config.json` KHÔNG commit (settings cá nhân)
