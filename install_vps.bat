@echo off
REM ============================================================
REM install_vps.bat — Cài đặt PhotoPro Studio trên VPS RTX 5090
REM Chạy 1 lần sau khi git clone/pull
REM ============================================================

echo.
echo ============================================================
echo PhotoPro Studio - VPS Setup (RTX 5090 + CUDA)
echo ============================================================
echo.

REM Check Python
python --version || (echo Python not found! && exit /b 1)

REM 1. Core scientific packages
echo [1/5] Installing core packages...
pip install numpy opencv-python pillow tqdm requests

REM 2. Torch CUDA (cu128 cho RTX 5090 - Blackwell)
echo [2/5] Installing PyTorch CUDA 12.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

REM 3. UI packages
echo [3/5] Installing UI packages...
pip install customtkinter

REM 4. BasicSR (cần cho CodeFormer + RealESRGAN utils)
echo [4/5] Installing basicsr + realesrgan...
pip install basicsr==1.4.2
pip install realesrgan
pip install facelib

REM 5. CodeFormer arch — Clone và install
echo [5/5] Installing CodeFormer (face restoration AI)...
if not exist "CodeFormer_repo" (
    git clone https://github.com/sczhou/CodeFormer.git CodeFormer_repo
)
cd CodeFormer_repo
pip install -e . --no-deps
cd ..

REM 6. Download model weights
echo.
echo [6/6] Downloading model weights...
python scripts/setup_models.py

echo.
echo ============================================================
echo Setup complete! Run: python main.py
echo ============================================================
pause
