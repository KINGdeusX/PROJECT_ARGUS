# -*- mode: python ; coding: utf-8 -*-
# argus_updater.spec  -- PyInstaller spec for the standalone updater
# Build:  pyinstaller argus_updater.spec

from pathlib import Path

block_cipher = None

a = Analysis(
    ['updater.py'],
    pathex=[str(Path('.').resolve())],
    binaries=[],
    datas=[
        ('version.py',   '.'),
        ('version.json', '.'),
    ],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['cv2', 'numpy', 'pytesseract', 'tkinter', 'matplotlib'],
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
    name='argus_updater',          # → dist/argus_updater.exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                 # GUI updater — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',      # Uncomment after adding an icon file
)
