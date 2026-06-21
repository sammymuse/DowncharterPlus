# -*- mode: python ; coding: utf-8 -*-
"""
downcharter.spec — PyInstaller (onedir) for Downcharter+

Bundles the GUI + all dependencies (numpy, soundfile/libsndfile, mido) into a
self-contained folder in dist/Downcharter+/. The user just unzips and runs the
.exe — no need to install Python or any libs.

    pyinstaller downcharter.spec --noconfirm

Onedir (not onefile): instant startup and a more robust libsndfile.dll.
"""
import os
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# soundfile brings libsndfile via collect_all (datas/binaries/submodules).
for pkg in ("soundfile",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += ["mido", "numpy"]

# soundfile is a single module (not a package) → collect_all does NOT pick up
# libsndfile. The DLL lives in <site-packages>/_soundfile_data/ and soundfile
# looks for it relative to itself, so we place it at the bundle root.
import soundfile as _sf
_sf_data = os.path.join(os.path.dirname(_sf.__file__), "_soundfile_data")
if os.path.isdir(_sf_data):
    datas.append((_sf_data, "_soundfile_data"))

# Optional icon (assets/downcharter.ico). Only used if it exists.
_icon = os.path.join("assets", "downcharter.ico")
icon = _icon if os.path.exists(_icon) else None

# Include the assets folder (icon) in the bundle if it exists.
if os.path.isdir("assets"):
    datas.append(("assets", "assets"))


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "pytest", "matplotlib",
              "PySide6", "shiboken6", "PyQt5", "PyQt6", "pymupdf", "fitz",
              "IPython", "scipy", "pandas"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Downcharter+",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # GUI app → no console
    disable_windowed_traceback=False,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Downcharter+",
)
