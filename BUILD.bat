@echo off
setlocal enabledelayedexpansion
title PhotoPro Studio — Build

echo.
echo =====================================================
echo    PhotoPro Studio Build Script
echo    Target: Windows x64, CUDA 12.8
echo =====================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Activate your venv first!
    pause & exit /b 1
)

REM Check PyInstaller
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyInstaller not installed. Run: pip install pyinstaller
    pause & exit /b 1
)

echo [INFO] Cleaning previous build...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo [INFO] Building...
python -m PyInstaller build.spec --clean --noconfirm

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed!
    pause & exit /b 1
)

echo.
echo =====================================================
echo    Build complete!
echo    Output: dist\PhotoPro_Studio\
echo =====================================================
echo.
pause
