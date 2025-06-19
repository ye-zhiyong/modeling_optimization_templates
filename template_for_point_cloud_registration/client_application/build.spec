# -*- mode: python ; coding: utf-8 -*-

import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)    # 将递归限制提高到5倍

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/optosky_logo.png', 'assets'),
        ('ui_mainwindow.py', '.'),
        ('registration_pcd.py', '.'),
        ('base_gps.py', '.')
    ],
    hiddenimports=[
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'open3d',
        'numpy',
        'pyproj',
        'matplotlib',
        'os',
        'sys',
        'uuid',
        'pandas',
        'datetime',
        'chardet'
    ],
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
    a.datas,
    [],
    name='PointCloudRegistration',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 设置为False以隐藏控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)