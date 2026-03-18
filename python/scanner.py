"""Directory walker and scan orchestrator.

Architecture: workers do AI inference only (CPU-bound, no DB writes).
The main scan thread collects results and writes to DB serially,
eliminating all SQLite write contention.
"""
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif"}
MAX_WORKERS = 2  # Conservative for CPU/RAM

_active_job_id: Optional[int] = None
_cancel_event = threading.Event()
_lock = threading.Lock()


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_images(paths: list[str]) -> list[str]:
    found = []
    for root_path in paths:
        p = Path(root_path)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(str(p))
        elif p.is_dir():
            for item in p.rglob("*"):
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.append(str(item))
    return sorted(set(found))


def _infer_image(image_path: str) -> dict:
    """
    Pure AI inference — NO database access.
    Returns a result dict that the main thread will persist.
    """
    import exif_engine
    import face_engine
    import caption_engine
    import clip_engine

    result: dict = {
        "path": image_path,
        "file_hash": None,
        "status": "ok",
        "error": None,
        "meta": {},
        "faces": [],       # list of {bbox, embedding_bytes}
        "caption": None,
        "clip_embedding": None,
    }

    try:
        result["file_hash"] = _file_hash(image_path)
        result["meta"] = exif_engine.extract_metadata(image_path)
        result["faces"] = [
            {"bbox": f["bbox"], "embedding_bytes": face_engine.embedding_to_bytes(f["embedding"])}
            for f in face_engine.detect_faces(image_path)
        ]
        result["caption"] = caption_engine.generate_caption(image_path)
        result["clip_embedding"] = clip_engine.embed_image(image_path)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def _persist_result(result: dict, db) -> None:
    """Write one inference result to DB. Called from the single writer thread."""
    import vector_store
    from models import MediaFile, Face

    if result["status"] != "ok":
        return

    # Check for duplicates (hash-based incremental scan)
    existing = db.query(MediaFile).filter(MediaFile.file_hash == result["file_hash"]).first()
    if existing:
        return

    meta = result["meta"]
    media = MediaFile(
        path=result["path"],
        file_hash=result["file_hash"],
        width=meta.get("width"),
        height=meta.get("height"),
        format=meta.get("format"),
        date_taken=meta.get("date_taken"),
        gps_lat=meta.get("gps_lat"),
        gps_lon=meta.get("gps_lon"),
        camera_model=meta.get("camera_model"),
        caption=result["caption"],
        media_type="image",
        processed_at=datetime.utcnow(),
    )
    db.add(media)
    db.flush()  # assigns media.id without committing

    for face_data in result["faces"]:
        db.add(Face(
            media_file_id=media.id,
            bbox_json=json.dumps(face_data["bbox"]),
            embedding=face_data["embedding_bytes"],
        ))

    if result["clip_embedding"] is not None:
        faiss_id = vector_store.add_embedding(media.id, result["clip_embedding"])
        media.faiss_index_id = faiss_id

    db.commit()


def _update_job(job_id: int, **kwargs) -> None:
    """Single short-lived write to update the ScanJob row."""
    from database import SessionLocal
    from models import ScanJob

    db = SessionLocal()
    try:
        job = db.query(ScanJob).filter(ScanJob.id == job_id).first()
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def start_scan(paths: list[str], job_id: int) -> None:
    """
    Scan orchestrator — runs on a dedicated background thread.

    Workers (ThreadPoolExecutor) do AI inference only.
    This thread is the sole DB writer, eliminating SQLite lock contention.
    """
    global _active_job_id

    from database import SessionLocal
    import cluster_engine

    _cancel_event.clear()
    with _lock:
        _active_job_id = job_id

    db = SessionLocal()
    try:
        _update_job(job_id, status="running", started_at=datetime.utcnow())

        images = _collect_images(paths)
        _update_job(job_id, total_files=len(images))

        processed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_infer_image, img): img for img in images}
            for future in as_completed(futures):
                if _cancel_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                img_path = futures[future]
                processed += 1

                try:
                    result = future.result()
                    _persist_result(result, db)
                except Exception as exc:
                    # Non-fatal: log and continue
                    pass

                # Single writer thread — no contention here
                _update_job(job_id, processed_files=processed, current_file=img_path)

        if _cancel_event.is_set():
            _update_job(job_id, status="cancelled", completed_at=datetime.utcnow())
        else:
            cluster_engine.run_clustering(db)
            db.close()
            db = None
            _update_job(job_id, status="done", completed_at=datetime.utcnow())

    except Exception as exc:
        if db:
            db.rollback()
        try:
            _update_job(
                job_id,
                status="error",
                error_message=str(exc),
                completed_at=datetime.utcnow(),
            )
        except Exception:
            pass
    finally:
        if db:
            db.close()
        with _lock:
            _active_job_id = None


def cancel_scan() -> bool:
    with _lock:
        if _active_job_id is None:
            return False
    _cancel_event.set()
    return True


def get_active_job_id() -> Optional[int]:
    with _lock:
        return _active_job_id
