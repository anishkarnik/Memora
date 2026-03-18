"""Face detection and embedding using InsightFace buffalo_sc (ONNX, CPU-compatible)."""
import json
from pathlib import Path
from typing import Optional

import numpy as np

_app = None  # InsightFace FaceAnalysis, lazy-loaded


def _get_app():
    global _app
    if _app is None:
        import insightface
        from insightface.app import FaceAnalysis

        model_dir = Path.home() / ".memora" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        _app = FaceAnalysis(
            name="buffalo_sc",
            root=str(model_dir),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def detect_faces(image_path: str) -> list[dict]:
    """
    Returns list of dicts:
      { bbox: [x1,y1,x2,y2], embedding: np.ndarray shape (512,) }
    Returns [] on any error or no faces.
    """
    try:
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return []
        app = _get_app()
        faces = app.get(img)
        results = []
        for face in faces:
            # Skip low-confidence detections — they produce noisy embeddings
            if hasattr(face, 'det_score') and face.det_score < 0.6:
                continue
            bbox = face.bbox.astype(int).tolist()  # [x1,y1,x2,y2]
            # Skip faces too small to be reliable (< 40×40 px)
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 1600:
                continue
            emb = face.normed_embedding.astype(np.float32)
            results.append({"bbox": bbox, "embedding": emb})
        return results
    except Exception:
        return []


def embedding_to_bytes(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
