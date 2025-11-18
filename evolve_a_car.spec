# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Evolve a Car application.
Supports building on Linux, Windows, and macOS.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all necessary data files
datas = [
    ('resources/*.ttf', 'resources'),
]

# Collect hidden imports for dependencies
hiddenimports = [
    'Box2D',
    'Box2D._Box2D',
    'pyglet',
    'pyglet.gl',
    'pyglet.window',
    'pyglet.window.xlib',  # Linux
    'pyglet.window.win32',  # Windows
    'pyglet.window.cocoa',  # macOS
    'imgui',
    'imgui.core',
    'imgui.integrations',
    'imgui.integrations.pyglet',
    'scipy',
    'scipy.spatial',
    'numpy',
]

# Add imgui data files
try:
    imgui_datas = collect_data_files('imgui')
    datas.extend(imgui_datas)
except Exception:
    pass

a = Analysis(
    ['src/app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='evolve_a_car',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for GUI-only mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon file path here if you have one
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='EvolveACar.app',
        icon=None,
        bundle_identifier='com.evolveacar.app',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
        },
    )
