# -*- mode: python ; coding: utf-8 -*-
# claims_scanner.spec  -- PyInstaller spec for the main Claims Scanner app
# Build:  pyinstaller claims_scanner.spec

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[str(Path('.').resolve())],
    binaries=[],
    datas=[
        # Bundle version files next to the EXE
        ('version.py', '.'),
        ('version.json', '.'),
        # Tesseract bundled binaries (optional — comment out if Tesseract is pre-installed on client)
        # ('tesseract-5.5.2', 'tesseract-5.5.2'),
    ],
    hiddenimports=[
        'cv2',
        'PIL',
        'PIL.Image',
        'numpy',
        'pytesseract',
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='claims_scanner',        # → dist/claims_scanner.exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                     # compress with UPX if available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                # GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',     # Uncomment after adding an icon file
)
