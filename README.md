# Memora — Privacy-First AI Media Manager

A desktop application that brings smartphone-style AI photo intelligence to Windows and Linux.
**100% local — no data ever leaves your machine.**

## Features

- **Face Recognition** — Detects and clusters faces across your photo library. Name people once; they're recognised everywhere.
- **AI Captions** — Choose from Moondream2, Florence-2, or BLIP to generate natural-language descriptions.
- **Semantic Search** — CLIP lets you search with plain English: *"mountains at sunset"*, *"birthday party"*.
- **EXIF Metadata** — Date, GPS, camera model extracted and displayed.
- **Privacy-First** — All AI runs locally via ONNX / HuggingFace transformers. No cloud required.
- **Adaptive Performance** — Auto-detects hardware (CPU cores, RAM, GPU/VRAM) and scales workers, batch sizes, and precision (float16/int8) accordingly. Three profiles: lite, standard, performance.

## Architecture

```
Tauri (Rust)  ←IPC→  React/TypeScript frontend
     ↕ HTTP (localhost)
Python FastAPI sidecar
  InsightFace · BLIP · CLIP · Pillow · SQLite · FAISS
```

## Prerequisites

| Tool | Version |
|------|---------|
| Rust | ≥ 1.77 |
| Node.js | ≥ 20 |
| Python | ≥ 3.11 |
| pip | latest |

## Quick Start (development)

### 1. Python sidecar

```bash
cd python
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run standalone (for API testing)
uvicorn main:app --port 8765 --reload
```

### 2. Tauri + React frontend

```bash
npm install
npm run tauri dev
```

On first run, models (~1.4 GB total) are downloaded to `~/.memora/models/`.

### 3. API testing

```bash
# Start a scan
curl -X POST http://localhost:8765/scan/start \
  -H 'Content-Type: application/json' \
  -d '{"paths":["/home/user/Pictures"]}'

# Check progress
curl http://localhost:8765/scan/status

# Search
curl "http://localhost:8765/search?q=beach+sunset&limit=10"
```

## Production Build

### Package Python sidecar

```bash
cd python
pip install pyinstaller
pyinstaller memora_sidecar.spec
# Output: dist/memora-sidecar/memora-sidecar (or .exe)
```

Copy the binary to `src-tauri/binaries/memora-sidecar-{target-triple}`.

### Build Tauri app

```bash
npm run tauri build
# Windows: NSIS installer in src-tauri/target/release/bundle/nsis/
# Linux:   AppImage + .deb in src-tauri/target/release/bundle/
```

## Data Storage

All data lives in `~/.memora/`:
- `memora.db` — SQLite database (images, faces, people, scan jobs)
- `clip_index.faiss` — FAISS vector index for semantic search
- `clip_index_map.npy` — Maps FAISS row index → media_files.id
- `models/` — Downloaded AI model weights
- `settings.json` — App settings (scan paths, auto-scan, performance profile, model selection)

## Adding Video Support (future)

1. Add `media_type` column handling in `scanner.py`
2. Create `python/video_engine.py` (keyframe extraction via cv2)
3. No schema, API, or frontend changes needed
