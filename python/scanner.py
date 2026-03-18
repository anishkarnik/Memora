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

import hardware

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif"}

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

        if not hardware.skip_face_detection():
            result["faces"] = [
                {"bbox": f["bbox"], "embedding_bytes": face_engine.embedding_to_bytes(f["embedding"])}
                for f in face_engine.detect_faces(image_path)
            ]

        if not hardware.skip_captioning():
            result["caption"] = caption_engine.generate_caption(image_path)

        # CLIP embedding always runs (core search feature, lightest model)
        result["clip_embedding"] = clip_engine.embed_image(image_path)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def _infer_batch(image_paths: list[str]) -> list[dict]:
    """
    Batch inference for multiple images.
    Per-image: hash, EXIF, face detection (via thread pool — InsightFace is single-image API).
    Batch: CLIP embeddings, captions.
    """
    import exif_engine
    import face_engine
    import caption_engine
    import clip_engine

    results = []
    for path in image_paths:
        results.append({
            "path": path,
            "file_hash": None,
            "status": "ok",
            "error": None,
            "meta": {},
            "faces": [],
            "caption": None,
            "clip_embedding": None,
        })

    # Per-image work: hash + EXIF + faces (use thread pool for face detection)
    ok_indices = []  # indices of images that didn't error on hash/exif
    for i, path in enumerate(image_paths):
        try:
            results[i]["file_hash"] = _file_hash(path)
            results[i]["meta"] = exif_engine.extract_metadata(path)
            ok_indices.append(i)
        except Exception as exc:
            results[i]["status"] = "error"
            results[i]["error"] = str(exc)

    # Face detection — sequential per image (InsightFace is single-image)
    if not hardware.skip_face_detection():
        for i in ok_indices:
            try:
                results[i]["faces"] = [
                    {"bbox": f["bbox"], "embedding_bytes": face_engine.embedding_to_bytes(f["embedding"])}
                    for f in face_engine.detect_faces(image_paths[i])
                ]
            except Exception:
                pass

    # Batch CLIP embeddings
    ok_paths = [image_paths[i] for i in ok_indices]
    if ok_paths:
        try:
            embeddings = clip_engine.embed_images_batch(ok_paths)
            for j, i in enumerate(ok_indices):
                results[i]["clip_embedding"] = embeddings[j]
        except Exception:
            # Fallback to sequential
            for i in ok_indices:
                try:
                    results[i]["clip_embedding"] = clip_engine.embed_image(image_paths[i])
                except Exception:
                    pass

    # Batch captions
    if not hardware.skip_captioning() and ok_paths:
        try:
            captions = caption_engine.generate_captions_batch(ok_paths)
            for j, i in enumerate(ok_indices):
                results[i]["caption"] = captions[j]
        except Exception:
            for i in ok_indices:
                try:
                    results[i]["caption"] = caption_engine.generate_caption(image_paths[i])
                except Exception:
                    pass

    return results


def _persist_result(result: dict, db) -> Optional[int]:
    """Write one inference result to DB. Called from the single writer thread.
    Returns the media_id if persisted, None otherwise."""
    import vector_store
    from models import MediaFile, Face

    if result["status"] != "ok":
        return None

    # Check for duplicates (hash-based incremental scan)
    existing = db.query(MediaFile).filter(MediaFile.file_hash == result["file_hash"]).first()
    if existing:
        return None

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

    return media.id


def _try_generate_thumbnail(image_path: str, media_id: int) -> None:
    """Pre-generate thumbnail during scan (non-fatal)."""
    try:
        from PIL import Image, ImageOps

        thumb_dir = Path.home() / ".memora" / "thumbnails"
        dest = thumb_dir / f"{media_id}.jpg"
        if dest.exists():
            return
        thumb_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = ImageOps.fit(img, (400, 400), Image.LANCZOS)
            img.save(dest, "JPEG", quality=75, optimize=True)
    except Exception:
        pass


def _update_job(job_id: int, **kwargs) -> None:
    """Update ScanJob using its own short-lived session.
    Use only when no other session holds a write transaction."""
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


def _update_job_inline(db, job_id: int, **kwargs) -> None:
    """Update ScanJob on the main scan session — avoids a second writer."""
    from models import ScanJob

    job = db.query(ScanJob).filter(ScanJob.id == job_id).first()
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
        # No separate commit — piggybacks on the next periodic commit


def start_scan(paths: list[str], job_id: int) -> None:
    """
    Scan orchestrator — runs on a dedicated background thread.

    Workers (ThreadPoolExecutor) do AI inference only.
    This thread is the sole DB writer, eliminating SQLite lock contention.
    """
    global _active_job_id

    from database import SessionLocal
    import cluster_engine
    import vector_store

    _cancel_event.clear()
    with _lock:
        _active_job_id = job_id

    # Use a separate session for status bookends (start/end) so they commit
    # immediately.  Progress updates inside the loop use the main session
    # via _update_job_inline to avoid a second concurrent writer on SQLite.
    _update_job(job_id, status="running", started_at=datetime.utcnow())

    db = SessionLocal()
    batch_size = hardware.get_clip_batch_size()
    commit_interval = hardware.get_db_commit_interval()

    try:
        images = _collect_images(paths)
        # total_files is safe via inline — nothing dirty on the session yet
        _update_job_inline(db, job_id, total_files=len(images))
        db.commit()

        processed = 0

        if batch_size <= 1:
            # Single-image mode (lite profile): use ThreadPoolExecutor as before
            with ThreadPoolExecutor(max_workers=hardware.get_max_workers()) as executor:
                futures = {executor.submit(_infer_image, img): img for img in images}
                for future in as_completed(futures):
                    if _cancel_event.is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    img_path = futures[future]
                    processed += 1

                    try:
                        result = future.result()
                        media_id = _persist_result(result, db)
                        if media_id is not None:
                            _try_generate_thumbnail(result["path"], media_id)
                    except Exception:
                        pass

                    # Update progress on the same session — committed together
                    _update_job_inline(db, job_id, processed_files=processed, current_file=img_path)

                    if processed % commit_interval == 0:
                        db.commit()
        else:
            # Batch mode (standard/performance profile)
            for i in range(0, len(images), batch_size):
                if _cancel_event.is_set():
                    break

                batch = images[i:i + batch_size]

                # Filter out already-hashed files
                from models import MediaFile
                to_process = []
                for img_path in batch:
                    try:
                        fh = _file_hash(img_path)
                        existing = db.query(MediaFile).filter(MediaFile.file_hash == fh).first()
                        if not existing:
                            to_process.append(img_path)
                    except Exception:
                        pass
                    processed += 1

                if to_process:
                    batch_results = _infer_batch(to_process)
                    for result in batch_results:
                        if _cancel_event.is_set():
                            break
                        try:
                            media_id = _persist_result(result, db)
                            if media_id is not None:
                                _try_generate_thumbnail(result["path"], media_id)
                        except Exception:
                            pass

                last_file = batch[-1] if batch else None
                _update_job_inline(db, job_id, processed_files=processed, current_file=last_file)

                if processed % commit_interval == 0:
                    db.commit()

        # Final commit for remaining unflushed writes
        db.commit()
        vector_store.flush()

        if _cancel_event.is_set():
            db.close()
            db = None
            _update_job(job_id, status="cancelled", completed_at=datetime.utcnow())
        else:
            # Clear face thumbnail cache before reclustering — person IDs
            # change during clustering, so stale thumbnails show wrong faces.
            face_thumb_dir = Path.home() / ".memora" / "face_thumbnails"
            if face_thumb_dir.exists():
                for thumb_file in face_thumb_dir.glob("*.jpg"):
                    thumb_file.unlink(missing_ok=True)
            cluster_engine.run_clustering(db)
            db.close()
            db = None
            _update_job(job_id, status="done", completed_at=datetime.utcnow())

    except Exception as exc:
        # Flush FAISS on error so we don't lose embeddings
        try:
            vector_store.flush()
        except Exception:
            pass
        if db:
            db.rollback()
            db.close()
            db = None
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
