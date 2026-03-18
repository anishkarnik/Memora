"""FAISS flat L2 index for image embeddings. Maps index row → media_files.id.

The index dimension is determined by the configured embedding_model in settings.json:
  clip    → 512-d
  siglip2 → 1152-d

If the on-disk index dimension does not match the configured model, the index is
automatically reset (empty). A full re-scan is needed to repopulate it.
"""
import json
import threading
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / ".memora"
INDEX_PATH = DATA_DIR / "clip_index.faiss"
MAP_PATH = DATA_DIR / "clip_index_map.npy"

_lock = threading.Lock()
_index = None
_id_map: list[int] = []
_adds_since_save: int = 0


def _get_dim() -> int:
    try:
        s_path = DATA_DIR / "settings.json"
        if s_path.exists():
            if json.loads(s_path.read_text()).get("embedding_model") == "siglip2":
                return 1152
    except Exception:
        pass
    return 512  # default: clip


def _get_index():
    global _index, _id_map
    if _index is None:
        import faiss

        dim = _get_dim()
        if INDEX_PATH.exists() and MAP_PATH.exists():
            loaded = faiss.read_index(str(INDEX_PATH))
            if loaded.d == dim:
                _index = loaded
                _id_map = np.load(str(MAP_PATH)).tolist()
            else:
                # Dimension mismatch — embedding model changed since last scan.
                # Reset to empty; user must re-scan to repopulate.
                _index = faiss.IndexFlatL2(dim)
                _id_map = []
                _save()
        else:
            _index = faiss.IndexFlatL2(dim)
            _id_map = []
    return _index


def add_embedding(media_file_id: int, embedding: np.ndarray) -> int:
    """Add embedding to index; returns the faiss_index_id (row index)."""
    global _adds_since_save
    import hardware

    with _lock:
        idx = _get_index()
        vec = embedding.astype(np.float32).reshape(1, idx.d)
        row = idx.ntotal
        idx.add(vec)
        _id_map.append(media_file_id)
        _adds_since_save += 1
        if _adds_since_save >= hardware.get_faiss_save_interval():
            _save()
            _adds_since_save = 0
        return row


def flush() -> None:
    """Force-save the FAISS index to disk. Call at end of scan or on error."""
    global _adds_since_save
    with _lock:
        if _index is not None and _adds_since_save > 0:
            _save()
            _adds_since_save = 0


def search(query_embedding: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
    """Returns list of (media_file_id, distance) sorted by distance ascending."""
    with _lock:
        idx = _get_index()
        if idx.ntotal == 0:
            return []
        k = min(k, idx.ntotal)
        vec = query_embedding.astype(np.float32).reshape(1, idx.d)
        distances, indices = idx.search(vec, k)
        results = []
        for dist, row in zip(distances[0], indices[0]):
            if row < 0 or row >= len(_id_map):
                continue
            results.append((_id_map[row], float(dist)))
        return results


def _save():
    """Must be called under _lock."""
    import faiss

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_index, str(INDEX_PATH))
    np.save(str(MAP_PATH), np.array(_id_map, dtype=np.int64))


def rebuild_from_db(session) -> None:
    """Reset the FAISS index to a consistent empty state."""
    global _index, _id_map
    import faiss

    with _lock:
        _index = faiss.IndexFlatL2(_get_dim())
        _id_map = []
        _save()


def total() -> int:
    with _lock:
        return _get_index().ntotal
