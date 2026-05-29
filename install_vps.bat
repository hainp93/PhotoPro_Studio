@echo off
REM ============================================================
REM install_vps.bat — PhotoPro Studio VPS Setup
REM Python 3.12 + PyTorch 2.11 + CUDA 12.8 (RTX 50xx)
REM ============================================================
chcp 65001 > nul
echo.
echo ============================================================
echo PhotoPro Studio - VPS Install
echo ============================================================

python --version || (echo [ERROR] Python not found! && pause && exit /b 1)

echo.
echo [1/5] Core packages...
pip install numpy opencv-python pillow tqdm requests rawpy

echo.
echo [2/5] PyTorch CUDA 12.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo.
echo [3/5] UI packages...
pip install customtkinter darkdetect

echo.
echo [4/5] CodeFormer repo (includes facelib + basicsr for Python 3.12)...
REM Xoa neu ton tai nhung bi loi
if exist "CodeFormer_repo\README.md" (
    echo CodeFormer_repo already exists, skipping clone.
) else (
    echo Removing broken CodeFormer_repo if exists...
    if exist "CodeFormer_repo" rmdir /s /q CodeFormer_repo
    echo Cloning CodeFormer...
    git clone https://github.com/sczhou/CodeFormer.git CodeFormer_repo
    if errorlevel 1 (echo [ERROR] git clone failed! Check internet. && pause && exit /b 1)
)

echo Installing CodeFormer dependencies...
pip install lpips einops scipy scikit-image matplotlib
pip install -r CodeFormer_repo/requirements.txt --ignore-requires-python

echo.
echo [5/5] Downloading model weights (~640MB)...
python scripts/setup_models.py

echo.
echo Checking setup...
python scripts/check_setup.py

echo.
echo ============================================================
echo Done! Run: python main.py
echo ============================================================
pause
