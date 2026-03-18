"""Image/text embeddings for semantic search — model selected via settings.json.

Supported embedding_model values:
  clip     — openai/clip-vit-base-patch32 (~350 MB, 512-d, fast CPU)  [default]
  siglip2  — google/siglip2-so400m-patch14-384 (~4.5 GB, 1152-d, GPU recommended)

The model is lazy-loaded on first use and cached for the process lifetime.
Changing embedding_model requires a sidecar restart AND a full re-scan
(the FAISS index dimension changes: 512 → 1152).
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

_processor = None
_model = None
_tokenizer = None   # siglip2 uses a separate tokenizer
_device = None
_loaded_model_name: Optional[str] = None


def _get_embedding_model() -> str:
    try:
        p = Path.home() / ".memora" / "settings.json"
        if p.exists():
            return json.loads(p.read_text()).get("embedding_model", "clip")
    except Exception:
        pass
    return "clip"


def get_embedding_dim() -> int:
    """Return the embedding dimension for the currently configured model."""
    return 1152 if _get_embedding_model() == "siglip2" else 512


def _load():
    global _processor, _model, _tokenizer, _device, _loaded_model_name
    if _model is not None:
        return

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = str(Path.home() / ".memora" / "models")
    model_name = _get_embedding_model()

    if model_name == "siglip2":
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        _processor = AutoProcessor.from_pretrained(
            "google/siglip2-so400m-patch14-384",
            cache_dir=cache_dir,
        )
        _tokenizer = AutoTokenizer.from_pretrained(
            "google/siglip2-so400m-patch14-384",
            cache_dir=cache_dir,
        )
        _model = AutoModel.from_pretrained(
            "google/siglip2-so400m-patch14-384",
            cache_dir=cache_dir,
        ).to(_device)
    else:  # clip (default)
        from transformers import CLIPProcessor, CLIPModel
        _processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
        )
        _model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
        ).to(_device)

    _model.eval()
    _loaded_model_name = model_name


def _normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def embed_image(image_path: str) -> Optional[np.ndarray]:
    """Returns a normalized embedding for the image, or None on error."""
    try:
        _load()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        if _loaded_model_name == "siglip2":
            inputs = _processor(images=[image], return_tensors="pt").to(_device)
            with torch.no_grad():
                features = _model.get_image_features(**inputs)
            emb = features[0].cpu().numpy().astype(np.float32)
        else:  # clip
            inputs = _processor(images=image, return_tensors="pt").to(_device)
            with torch.no_grad():
                features = _model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                emb = features.pooler_output[0].cpu().numpy().astype(np.float32)
            else:
                emb = features[0].cpu().numpy().astype(np.float32)

        return _normalize(emb)
    except Exception:
        return None


def embed_text(text: str) -> Optional[np.ndarray]:
    """Returns a normalized text embedding, or None on error."""
    try:
        _load()
        import torch

        if _loaded_model_name == "siglip2":
            # SigLIP requires padding="max_length" (trained that way)
            inputs = _tokenizer(
                [text], padding="max_length", return_tensors="pt"
            ).to(_device)
            with torch.no_grad():
                features = _model.get_text_features(**inputs)
            emb = features[0].cpu().numpy().astype(np.float32)
        else:  # clip
            inputs = _processor(
                text=[text], return_tensors="pt", padding=True
            ).to(_device)
            with torch.no_grad():
                features = _model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                emb = features.pooler_output[0].cpu().numpy().astype(np.float32)
            else:
                emb = features[0].cpu().numpy().astype(np.float32)

        return _normalize(emb)
    except Exception:
        return None


def embedding_to_bytes(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)
