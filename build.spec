# -*- mode: python ; coding: utf-8 -*-
# PhotoPro Studio — PyInstaller Spec
# Build: pyinstaller build.spec --clean

import sys
from pathlib import Path

ROOT = Path(SPECPATH)

block_cipher = None

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # Presets
        (str(ROOT / "presets"), "presets"),
        # Config defaults
        (str(ROOT / "config"), "config"),
        # Assets (icon etc.)
        # (str(ROOT / "assets"), "assets"),
    ],
    hiddenimports=[
        "customtkinter",
        "PIL._tkinter_finder",
        "cv2",
        "numpy",
        "torch",
        "torchvision",
        "basicsr",
        "basicsr.archs.rrdbnet_arch",
        "basicsr.utils.realesrgan_utils",
        "basicsr.utils.registry",
        "realesrgan",
        "facexlib",
        "facelib",
        "facelib.utils.face_restoration_helper",
        "rawpy",
        "psutil",
        "yaml",
        "scipy",
        "skimage",
        "pywt",
        "onnxruntime",
        "insightface",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "jupyter", "notebook", "IPython", "matplotlib.tests",
        "tkinter.test", "test", "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PhotoPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # Không hiện console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="assets/icon.ico",  # uncomment khi có icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PhotoPro_Studio",
)
