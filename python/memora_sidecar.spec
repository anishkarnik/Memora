# PyInstaller spec for the Memora sidecar binary.
# Run:  pyinstaller memora_sidecar.spec
# Output: dist/memora-sidecar  (or .exe on Windows)

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ["sidecar_entry.py"],
    pathex=[str(Path(".").resolve())],
    binaries=[],
    datas=[],
    hiddenimports=[
        # FastAPI / Uvicorn internals
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        # SQLAlchemy dialects
        "sqlalchemy.dialects.sqlite",
        # HuggingFace transformers extras
        "transformers.models.blip",
        "transformers.models.clip",
        # InsightFace
        "insightface.app",
        "insightface.model_zoo",
        # sklearn
        "sklearn.cluster._dbscan_inner",
        "sklearn.neighbors._dist_metrics",
        # FAISS
        "faiss",
        # Hardware detection
        "psutil",
        "hardware",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "notebook", "IPython"],
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
    name="memora-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="memora-sidecar",
)
