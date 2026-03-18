# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow

### Step 1 — Python sidecar (must start first)
```bash
cd python
source .venv/bin/activate
uvicorn main:app --port 8765 --reload
```

### Step 2 — Tauri + React app
```bash
npm run tauri dev
```

In dev mode, Tauri does **not** spawn a Python process — `sidecar.rs` attaches to the already-running uvicorn on port 8765 (overridable via `MEMORA_PORT` env var). The Python sidecar must be running before the Tauri app starts or it will error out.

### Frontend only (no Tauri shell)
```bash
npm run dev   # Vite on localhost:1420 — useful for pure UI work
```

### Test API endpoints directly
```bash
curl -X POST http://localhost:8765/scan/start \
  -H 'Content-Type: application/json' \
  -d '{"paths":["/path/to/photos"]}'

curl http://localhost:8765/scan/status
curl "http://localhost:8765/search?q=beach+sunset"
```

### Production build
```bash
# 1. Bundle Python sidecar
cd python && pyinstaller memora_sidecar.spec
# Copy dist/memora-sidecar binary to src-tauri/binaries/memora-sidecar-{target-triple}

# 2. Build Tauri app
npm run tauri build
```

---

## Architecture

```
React/TypeScript frontend
        ↕  invoke("api_request" | "start_sidecar" | "get_sidecar_port")
Rust (Tauri) — src-tauri/src/lib.rs + sidecar.rs
        ↕  HTTP REST on localhost:{random port}
Python FastAPI sidecar — python/main.py
        ↕
SQLite (~/. memora/memora.db) + FAISS (~/.memora/clip_index.faiss)
```

**The frontend never talks to Python directly.** All HTTP goes through the `api_request` Tauri command in `lib.rs`, which proxies to the sidecar port managed by `sidecar.rs`. This avoids CORS and keeps the port opaque to the webview.

**Dev vs release sidecar:** `sidecar.rs` uses `#[cfg(dev)]` / `#[cfg(not(dev))]` to branch. Dev attaches to an existing uvicorn; release spawns the PyInstaller binary from `src-tauri/binaries/`.

---

## Python Backend Key Design Decisions

### SQLite write contention
`scanner.py` deliberately keeps **one thread as the sole DB writer**. Worker threads (`ThreadPoolExecutor`, 2 workers) run `_infer_image()` which does all AI inference and returns a plain dict. The scan orchestrator thread then calls `_persist_result()` serially. Never add DB writes inside `_infer_image()`.

### Face clustering
`cluster_engine.py` uses `AgglomerativeClustering(linkage="average", distance_threshold=0.45)`, not DBSCAN. DBSCAN with `min_samples=1` degenerates to single-linkage chaining and merges different people. Tuning notes:
- `DISTANCE_THRESHOLD = 0.45` — lower = stricter (more splits), higher = looser (more merges). InsightFace buffalo_sc: same-person ~0.2–0.4, different-person ~0.5+.
- `NAMED_MERGE_THRESHOLD = 0.60` — centroid similarity required to match a new cluster to an existing **named** person during re-cluster.
- `UNNAMED_MERGE_THRESHOLD = 0.55` — same for unnamed persons.

Re-clustering (`POST /cluster`) preserves named `Person` rows, deletes all unnamed ones, clears the face thumbnail cache (`~/.memora/face_thumbnails/`), then rebuilds. The cache must be cleared because SQLite reuses deleted person IDs — stale cached thumbnails would show the wrong face.

### Incremental scan
Files are skipped if their SHA-256 hash already exists in `media_files.file_hash`. Adding/modifying files is handled; renames are treated as new files.

### Thumbnail caching
- Media thumbnails: `~/.memora/thumbnails/{media_id}.jpg` — 400×400px center-crop JPEG
- Face thumbnails: `~/.memora/face_thumbnails/{person_id}.jpg` — 200×200px crop around face bbox with 40% padding

Both are generated lazily on first request and served by FastAPI via `FileResponse`.

---

## Data Directory (`~/.memora/`)

| Path | Contents |
|------|----------|
| `memora.db` | SQLite — media_files, faces, people, scan_jobs |
| `clip_index.faiss` | FAISS flat index for semantic search |
| `clip_index_map.npy` | Maps FAISS row → media_files.id |
| `models/` | InsightFace buffalo_sc weights (~80 MB), downloaded on first run |
| `thumbnails/` | Cached 400px media thumbnails |
| `face_thumbnails/` | Cached 200px face-crop thumbnails |
| `settings.json` | scan_paths, auto_scan_enabled |

`MEMORA_DATA_DIR` env var overrides the data directory location.

---

## AI Models

| Model | Purpose | Lazy-loaded in |
|-------|---------|----------------|
| InsightFace `buffalo_sc` (ONNX) | Face detection + 512-d embeddings | `face_engine.py` |
| `vikhyatk/moondream2` (~1.6B) | Image captioning | `caption_engine.py` |
| `openai/clip-vit-base-patch32` | Semantic search embeddings | `clip_engine.py` |

All models are CPU-compatible; CUDA is used automatically if available. Models are downloaded to `~/.memora/models/` on first use — not bundled in the repo.

---

## Frontend API Contract

`src/api/client.ts` is the single source of truth for the frontend↔sidecar API. All calls go via `invoke("api_request", { method, path, body })`. Image URLs are never hardcoded — `ensureSidecarPort()` resolves the dynamic port at runtime before constructing `http://127.0.0.1:{port}/...` URLs.

`PersonSummary.face_thumbnail_url` returns a relative path (e.g. `/people/3/face-thumbnail`), not an absolute URL. Components call `ensureSidecarPort()` to prepend the host.
