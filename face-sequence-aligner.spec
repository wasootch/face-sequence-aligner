# PyInstaller spec for Face Sequence Aligner
# Build:  pyinstaller face-sequence-aligner.spec
# Output: dist/FaceSequenceAligner/FaceSequenceAligner.exe

from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

# ── Data files from third-party packages ──────────────────────────────────────
mediapipe_datas,    mediapipe_binaries,    mediapipe_hiddens    = collect_all('mediapipe')
customtkinter_datas = collect_data_files('customtkinter')

# Bundled app data (music README so the folder structure is present)
app_datas = [
    ('music/README.md', 'music'),
]

a = Analysis(
    ['main.py'],
    pathex=[SPECPATH],
    binaries=mediapipe_binaries,
    datas=[
        *mediapipe_datas,
        *customtkinter_datas,
        *app_datas,
    ],
    hiddenimports=[
        *mediapipe_hiddens,
        'customtkinter',
        'PIL._tkinter_finder',
        # Tkinter backend used by customtkinter
        '_tkinter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'scipy', 'IPython', 'jupyter',
        'tkinter.test', 'unittest',
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
    name='FaceSequenceAligner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',  # uncomment and add icon.ico to use a custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceSequenceAligner',
)
