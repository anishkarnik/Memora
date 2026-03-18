"""FastAPI sidecar — all REST endpoints for Memora."""
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import database
import scanner
from models import Face, MediaFile, Person, ScanJob

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    # On startup, any job stuck in "running"/"queued" from a previous crashed
    # process will never finish — mark them as errored so the UI doesn't spin forever.
    _reset_stuck_jobs()
    yield


def _reset_stuck_jobs():
    from database import SessionLocal
    db = SessionLocal()
    try:
        stuck = db.query(ScanJob).filter(ScanJob.status.in_(["running", "queued"])).all()
        for job in stuck:
            job.status = "error"
            job.error_message = "Process restarted — scan was interrupted"
        if stuck:
            db.commit()
    finally:
        db.close()


app = FastAPI(title="Memora API", lifespan=lifespan)

# Allow Tauri webview origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://localhost", "http://127.0.0.1"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ScanStartRequest(BaseModel):
    paths: list[str]
    mode: str = "manual"  # "manual" | "auto"


class NameRequest(BaseModel):
    name: str


class MergeRequest(BaseModel):
    source_id: int
    target_id: int


class SettingsRequest(BaseModel):
    scan_paths: list[str] = []
    auto_scan_enabled: bool = False
    caption_model: str = "moondream2"    # moondream2 | florence2 | blip
    embedding_model: str = "clip"         # clip | siglip2
    performance_profile: str = "auto"     # auto | lite | standard | performance
    skip_captioning: bool = False
    skip_face_detection: bool = False


# Simple in-memory settings (persisted to ~/.memora/settings.json)
SETTINGS_PATH = Path.home() / ".memora" / "settings.json"

_SETTINGS_DEFAULTS = {
    "scan_paths": [],
    "auto_scan_enabled": False,
    "caption_model": "moondream2",
    "embedding_model": "clip",
    "performance_profile": "auto",
    "skip_captioning": False,
    "skip_face_detection": False,
}


def _load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
            # Merge with defaults so new fields are always present
            return {**_SETTINGS_DEFAULTS, **saved}
        except Exception:
            pass
    return dict(_SETTINGS_DEFAULTS)


def _save_settings(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Always persist all known keys so downstream readers never hit KeyError
    merged = {**_SETTINGS_DEFAULTS, **data}
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


# ---------------------------------------------------------------------------
# System / Hardware
# ---------------------------------------------------------------------------

@app.get("/system/hardware")
def system_hardware():
    import hardware
    return hardware.get_hardware_info()


@app.post("/system/preload-models")
def preload_models():
    """Warm up AI models in a background thread so first scan is faster."""
    def _preload():
        try:
            import face_engine
            face_engine._get_app()
        except Exception:
            pass
        try:
            import caption_engine
            caption_engine._load()
        except Exception:
            pass
        try:
            import clip_engine
            clip_engine._load()
        except Exception:
            pass

    thread = threading.Thread(target=_preload, daemon=True)
    thread.start()
    return {"status": "preloading"}


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

@app.post("/scan/start")
def scan_start(req: ScanStartRequest, db: Session = Depends(database.get_db)):
    # Only one scan at a time
    active = scanner.get_active_job_id()
    if active is not None:
        raise HTTPException(status_code=409, detail="Scan already running")

    job = ScanJob(paths_json=json.dumps(req.paths), status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)

    thread = threading.Thread(
        target=scanner.start_scan, args=(req.paths, job.id), daemon=True
    )
    thread.start()

    return {"job_id": job.id, "status": "queued"}


@app.get("/scan/status")
def scan_status(db: Session = Depends(database.get_db)):
    active_id = scanner.get_active_job_id()
    if active_id is None:
        # Return last job
        job = (
            db.query(ScanJob)
            .order_by(ScanJob.id.desc())
            .first()
        )
    else:
        job = db.query(ScanJob).filter(ScanJob.id == active_id).first()

    if job is None:
        return {"status": "idle", "progress_pct": 0, "current_file": None}

    pct = 0
    if job.total_files and job.total_files > 0:
        pct = round(100 * job.processed_files / job.total_files, 1)

    return {
        "job_id": job.id,
        "status": job.status,
        "progress_pct": pct,
        "total_files": job.total_files,
        "processed_files": job.processed_files,
        "current_file": job.current_file,
        "error_message": job.error_message,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@app.post("/scan/cancel")
def scan_cancel():
    scanner.cancel_scan()  # no-op if nothing is running
    return {"status": "cancelling"}


@app.post("/cluster")
def recluster(db: Session = Depends(database.get_db)):
    """
    Re-run face clustering preserving named people.
    - Named Person rows are kept; new faces close to them are auto-assigned.
    - Only unnamed Person rows are dropped and rebuilt from scratch.
    """
    import cluster_engine
    # Unassign all faces first so clustering starts with a clean slate
    db.query(Face).update({"person_id": None})
    # Delete only unnamed people — named ones survive and act as anchors
    db.query(Person).filter(Person.name.is_(None)).delete()
    db.commit()
    # Clear ALL face thumbnail cache — person IDs get reused after delete,
    # so stale cached files would show wrong faces for new persons.
    if FACE_THUMB_DIR.exists():
        for f in FACE_THUMB_DIR.glob("*.jpg"):
            f.unlink(missing_ok=True)
    result = cluster_engine.run_clustering(db)
    return result


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

FACE_THUMB_DIR = Path.home() / ".memora" / "face_thumbnails"
FACE_THUMB_SIZE = 200  # square px


def _face_thumb_path(person_id: int) -> Path:
    return FACE_THUMB_DIR / f"{person_id}.jpg"


def _generate_face_thumbnail(image_path: str, bbox: list, dest: Path) -> None:
    """Crop image to face bbox with 40% padding, square-resize to FACE_THUMB_SIZE."""
    from PIL import Image, ImageOps
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        W, H = img.size
        x1, y1, x2, y2 = bbox
        fw, fh = x2 - x1, y2 - y1
        pad = max(fw, fh) * 0.4  # 40% padding so face isn't edge-to-edge
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(fw, fh) + pad * 2  # square region
        left  = max(0, int(cx - side / 2))
        top   = max(0, int(cy - side / 2))
        right = min(W, int(cx + side / 2))
        bottom = min(H, int(cy + side / 2))
        face_crop = img.crop((left, top, right, bottom))
        face_crop = ImageOps.fit(face_crop, (FACE_THUMB_SIZE, FACE_THUMB_SIZE), Image.LANCZOS)
        face_crop.save(dest, "JPEG", quality=85, optimize=True)


@app.get("/people")
def list_people(db: Session = Depends(database.get_db)):
    people = db.query(Person).all()
    result = []
    for p in people:
        face_count = db.query(Face).filter(Face.person_id == p.id).count()
        result.append({
            "id": p.id,
            "name": p.name,
            "face_count": face_count,
            "face_thumbnail_url": f"/people/{p.id}/face-thumbnail",
            "created_at": p.created_at.isoformat() if p.created_at else None,
        })
    # Named people first, then unnamed sorted by face count descending
    result.sort(key=lambda x: (x["name"] is None, -x["face_count"]))
    return result


@app.get("/people/{person_id}/face-thumbnail")
def get_face_thumbnail(person_id: int, db: Session = Depends(database.get_db)):
    """Serve a face-cropped square thumbnail for a person."""
    dest = _face_thumb_path(person_id)
    if not dest.exists():
        # Pick the face with the largest bbox area (most prominent / clearest shot)
        faces = db.query(Face).filter(Face.person_id == person_id).all()
        best_face = None
        best_area = 0
        for face in faces:
            bbox = json.loads(face.bbox_json)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > best_area:
                best_area = area
                best_face = face

        if not best_face:
            raise HTTPException(status_code=404, detail="No face found")

        media = db.query(MediaFile).filter(MediaFile.id == best_face.media_file_id).first()
        if not media or not Path(media.path).exists():
            raise HTTPException(status_code=404, detail="Source image not found")

        try:
            bbox = json.loads(best_face.bbox_json)
            _generate_face_thumbnail(media.path, bbox, dest)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(
        str(dest),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/people/{person_id}/images")
def person_images(person_id: int, db: Session = Depends(database.get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    faces = db.query(Face).filter(Face.person_id == person_id).all()
    media_ids = list({f.media_file_id for f in faces})
    media_files = db.query(MediaFile).filter(MediaFile.id.in_(media_ids)).all()

    return [_media_summary(m) for m in media_files]


@app.post("/people/{person_id}/name")
def set_person_name(person_id: int, req: NameRequest, db: Session = Depends(database.get_db)):
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    person.name = req.name
    db.commit()
    return {"id": person_id, "name": req.name}


@app.post("/people/merge")
def merge_people(req: MergeRequest, db: Session = Depends(database.get_db)):
    source = db.query(Person).filter(Person.id == req.source_id).first()
    target = db.query(Person).filter(Person.id == req.target_id).first()
    if not source or not target:
        raise HTTPException(status_code=404, detail="Person not found")

    # Reassign all faces from source → target
    db.query(Face).filter(Face.person_id == req.source_id).update(
        {"person_id": req.target_id}
    )
    db.delete(source)
    db.commit()
    return {"merged_into": req.target_id}


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

@app.get("/gallery")
def gallery(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(database.get_db),
):
    offset = (page - 1) * page_size
    total = db.query(MediaFile).count()
    items = (
        db.query(MediaFile)
        .order_by(MediaFile.date_taken.desc().nullslast(), MediaFile.id.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_media_summary(m) for m in items],
    }


@app.get("/media/{media_id}")
def get_media(media_id: int, db: Session = Depends(database.get_db)):
    media = db.query(MediaFile).filter(MediaFile.id == media_id).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    faces = db.query(Face).filter(Face.media_file_id == media_id).all()
    face_data = []
    for f in faces:
        face_data.append({
            "id": f.id,
            "bbox": json.loads(f.bbox_json),
            "person_id": f.person_id,
        })

    return {
        **_media_detail(media),
        "faces": face_data,
    }


THUMB_DIR = Path.home() / ".memora" / "thumbnails"
THUMB_SIZE = 400  # square side length


def _thumb_path(media_id: int) -> Path:
    return THUMB_DIR / f"{media_id}.jpg"


def _generate_thumbnail(src: str, dest: Path) -> None:
    from PIL import Image, ImageOps
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)   # correct rotation from EXIF
        img = img.convert("RGB")
        # Center-crop to square, then resize — every thumbnail is exactly 400×400
        img = ImageOps.fit(img, (THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        img.save(dest, "JPEG", quality=75, optimize=True)


@app.get("/media/{media_id}/thumbnail")
def get_thumbnail(media_id: int, db: Session = Depends(database.get_db)):
    """Serve a 400px cached thumbnail; generate on first request."""
    thumb = _thumb_path(media_id)
    if not thumb.exists():
        media = db.query(MediaFile).filter(MediaFile.id == media_id).first()
        if not media or not Path(media.path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        try:
            _generate_thumbnail(media.path, thumb)
        except Exception:
            # Fall back to original if thumbnail generation fails
            return FileResponse(media.path)
    return FileResponse(str(thumb), media_type="image/jpeg")


@app.get("/media/{media_id}/file")
def get_file(media_id: int, db: Session = Depends(database.get_db)):
    media = db.query(MediaFile).filter(MediaFile.id == media_id).first()
    if not media or not Path(media.path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(media.path)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/search")
def search(
    q: str = Query(""),
    limit: int = Query(20, ge=1, le=100),
    person_id: Optional[int] = Query(None),
    db: Session = Depends(database.get_db),
):
    import clip_engine
    import vector_store

    q = q.strip()

    # Person-only mode: no text query, just return all photos for that person
    if not q and person_id is not None:
        person_media_ids = [
            f.media_file_id
            for f in db.query(Face).filter(Face.person_id == person_id).all()
        ]
        media_rows = (
            db.query(MediaFile)
            .filter(MediaFile.id.in_(person_media_ids))
            .order_by(MediaFile.date_taken.desc().nullslast(), MediaFile.id.desc())
            .limit(limit)
            .all()
        )
        results = []
        for m in media_rows:
            item = _media_summary(m)
            item["score"] = 1.0
            results.append(item)
        return {"query": "", "results": results}

    if not q:
        return {"query": "", "results": []}

    text_emb = clip_engine.embed_text(q)
    clip_results: list[tuple[int, float]] = []
    if text_emb is not None:
        clip_results = vector_store.search(text_emb, k=limit * 2)

    # Also do full-text search on captions
    caption_matches = (
        db.query(MediaFile)
        .filter(MediaFile.caption.ilike(f"%{q}%"))
        .limit(limit)
        .all()
    )
    fts_ids: set[int] = {m.id for m in caption_matches}

    # Merge results: CLIP first, then FTS
    seen: set[int] = set()
    merged: list[tuple[int, float]] = []

    for media_id, dist in clip_results:
        if media_id not in seen:
            merged.append((media_id, dist))
            seen.add(media_id)

    for mid in fts_ids:
        if mid not in seen:
            merged.append((mid, 9999.0))  # Low rank
            seen.add(mid)

    # Filter by person if requested
    if person_id is not None:
        person_media_ids = {
            f.media_file_id
            for f in db.query(Face).filter(Face.person_id == person_id).all()
        }
        merged = [(mid, d) for mid, d in merged if mid in person_media_ids]

    merged = merged[:limit]

    # Fetch media records
    id_to_dist = {mid: d for mid, d in merged}
    media_rows = db.query(MediaFile).filter(MediaFile.id.in_(list(id_to_dist.keys()))).all()
    id_to_media = {m.id: m for m in media_rows}

    results = []
    for mid, dist in merged:
        m = id_to_media.get(mid)
        if m:
            item = _media_summary(m)
            item["score"] = round(1 / (1 + dist), 4)  # normalized score 0-1
            results.append(item)

    return {"query": q, "results": results}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@app.get("/settings")
def get_settings():
    return _load_settings()


@app.post("/settings")
def update_settings(req: SettingsRequest):
    data = {
        "scan_paths": req.scan_paths,
        "auto_scan_enabled": req.auto_scan_enabled,
        "caption_model": req.caption_model,
        "embedding_model": req.embedding_model,
        "performance_profile": req.performance_profile,
        "skip_captioning": req.skip_captioning,
        "skip_face_detection": req.skip_face_detection,
    }
    _save_settings(data)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _media_summary(m: MediaFile) -> dict:
    return {
        "id": m.id,
        "path": m.path,
        "thumbnail_url": f"/media/{m.id}/thumbnail",
        "file_url": f"/media/{m.id}/file",
        "width": m.width,
        "height": m.height,
        "format": m.format,
        "date_taken": m.date_taken.isoformat() if m.date_taken else None,
        "caption": m.caption,
        "media_type": m.media_type,
    }


def _media_detail(m: MediaFile) -> dict:
    return {
        **_media_summary(m),
        "gps_lat": m.gps_lat,
        "gps_lon": m.gps_lon,
        "camera_model": m.camera_model,
        "processed_at": m.processed_at.isoformat() if m.processed_at else None,
    }
