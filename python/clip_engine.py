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

    # GPU: use float16 for ~2x throughput and lower VRAM
    if _device == "cuda":
        _model = _model.half()
    # CPU lite mode: int8 dynamic quantization
    elif _device == "cpu":
        import hardware
        if hardware.get_profile() == "lite":
            _model = torch.quantization.quantize_dynamic(
                _model, {torch.nn.Linear}, dtype=torch.qint8
            )

    _loaded_model_name = model_name


def _normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def _inference_context():
    """Return autocast context manager for CUDA float16, or nullcontext."""
    import torch
    if _device == "cuda":
        return torch.autocast("cuda")
    from contextlib import nullcontext
    return nullcontext()


def embed_image(image_path: str) -> Optional[np.ndarray]:
    """Returns a normalized embedding for the image, or None on error."""
    try:
        _load()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        with _inference_context():
            if _loaded_model_name == "siglip2":
                inputs = _processor(images=[image], return_tensors="pt").to(_device)
                with torch.no_grad():
                    features = _model.get_image_features(**inputs)
                emb = features[0].cpu().float().numpy().astype(np.float32)
            else:  # clip
                inputs = _processor(images=image, return_tensors="pt").to(_device)
                with torch.no_grad():
                    features = _model.get_image_features(**inputs)
                if hasattr(features, "pooler_output"):
                    emb = features.pooler_output[0].cpu().float().numpy().astype(np.float32)
                else:
                    emb = features[0].cpu().float().numpy().astype(np.float32)

        return _normalize(emb)
    except Exception:
        return None


def embed_images_batch(image_paths: list[str]) -> list[Optional[np.ndarray]]:
    """Batch-embed multiple images. Returns list of normalized embeddings (None for failures)."""
    if not image_paths:
        return []
    # Fallback to sequential for single image
    if len(image_paths) == 1:
        return [embed_image(image_paths[0])]

    try:
        _load()
        import torch
        from PIL import Image

        images = []
        indices = []  # tracks which original positions succeeded
        for i, path in enumerate(image_paths):
            try:
                images.append(Image.open(path).convert("RGB"))
                indices.append(i)
            except Exception:
                pass

        if not images:
            return [None] * len(image_paths)

        results: list[Optional[np.ndarray]] = [None] * len(image_paths)

        with _inference_context():
            if _loaded_model_name == "siglip2":
                inputs = _processor(images=images, return_tensors="pt", padding=True).to(_device)
            else:
                inputs = _processor(images=images, return_tensors="pt", padding=True).to(_device)

            with torch.no_grad():
                features = _model.get_image_features(**inputs)

            for j, orig_idx in enumerate(indices):
                emb = features[j].cpu().float().numpy().astype(np.float32)
                results[orig_idx] = _normalize(emb)

        return results
    except Exception:
        # Fallback to sequential on any batch error
        return [embed_image(p) for p in image_paths]


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
