@echo off
REM ============================================================
REM install_vps.bat — PhotoPro Studio VPS Setup
REM Python 3.12 + PyTorch 2.11 + CUDA 12.8 (RTX 5060Ti/5090)
REM ============================================================
chcp 65001 > nul
echo.
echo ============================================================
echo PhotoPro Studio - VPS Install
echo ============================================================
echo.

python --version || (echo [ERROR] Python not found! && pause && exit /b 1)

echo [1/6] Core packages...
pip install numpy opencv-python pillow tqdm requests rawpy

echo.
echo [2/6] PyTorch CUDA 12.8 (RTX 50xx Blackwell)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo.
echo [3/6] UI packages...
pip install customtkinter darkdetect

echo.
echo [4/6] facelib (face detection + parsing)...
pip install facelib

echo.
echo [5/6] CodeFormer (face restoration AI)...
REM Dung CodeFormer repo thay vi basicsr pip - tuong thich Python 3.12
if not exist "CodeFormer_repo" (
    echo Cloning CodeFormer from GitHub...
    git clone https://github.com/sczhou/CodeFormer.git CodeFormer_repo
    if errorlevel 1 (echo [ERROR] git clone failed! && pause && exit /b 1)
)
cd CodeFormer_repo
echo Installing CodeFormer...
pip install -e . --no-deps
pip install lpips einops tb-nightly
cd ..

echo.
echo [6/6] Downloading model weights (~640MB total)...
python scripts/setup_models.py

echo.
echo ============================================================
echo Checking setup...
python scripts/check_setup.py

echo.
echo ============================================================
echo Done! Run: python main.py
echo ============================================================
pause
