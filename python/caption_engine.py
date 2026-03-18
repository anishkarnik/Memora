"""Image captioning — model selected via ~/.memora/settings.json (caption_model).

Supported values:
  moondream2  — vikhyatk/moondream2 (~3.8 GB, best quality)
  florence2   — microsoft/Florence-2-base (~270 MB, fast CPU)
  blip        — Salesforce/blip-image-captioning-base (~990 MB, legacy)

The model is lazy-loaded on first use and cached for the process lifetime.
Changing caption_model takes effect after restarting the sidecar.
"""
import json
from pathlib import Path
from typing import Optional

_model = None
_tokenizer = None   # moondream2
_processor = None   # florence2 / blip
_device = None
_loaded_model_name: Optional[str] = None


def _get_caption_model() -> str:
    try:
        p = Path.home() / ".memora" / "settings.json"
        if p.exists():
            return json.loads(p.read_text()).get("caption_model", "moondream2")
    except Exception:
        pass
    return "moondream2"


def _load():
    global _model, _tokenizer, _processor, _device, _loaded_model_name
    if _model is not None:
        return

    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = str(Path.home() / ".memora" / "models")
    model_name = _get_caption_model()

    if model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM
        _processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(_device)

    elif model_name == "blip":
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir,
        )
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir=cache_dir,
        ).to(_device)

    else:  # moondream2 (default)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(_device)

    _model.eval()

    # GPU: use float16 for ~2x throughput and lower VRAM
    if _device == "cuda":
        _model = _model.half()
    # CPU lite mode: int8 dynamic quantization (florence2/blip only — moondream2 has
    # custom architecture that may not quantize cleanly)
    elif _device == "cpu" and model_name in ("florence2", "blip"):
        import hardware
        if hardware.get_profile() == "lite":
            _model = torch.quantization.quantize_dynamic(
                _model, {torch.nn.Linear}, dtype=torch.qint8
            )

    _loaded_model_name = model_name


def _inference_context():
    """Return autocast context manager for CUDA, or nullcontext."""
    import torch
    if _device == "cuda":
        return torch.autocast("cuda")
    from contextlib import nullcontext
    return nullcontext()


def generate_caption(image_path: str) -> Optional[str]:
    """Generate a natural language caption for the image. Returns None on error."""
    try:
        _load()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        with _inference_context():
            if _loaded_model_name == "florence2":
                inputs = _processor(
                    text="<CAPTION>", images=image, return_tensors="pt"
                ).to(_device)
                with torch.no_grad():
                    generated_ids = _model.generate(
                        **inputs, max_new_tokens=100, num_beams=3
                    )
                generated_text = _processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                parsed = _processor.post_process_generation(
                    generated_text,
                    task="<CAPTION>",
                    image_size=(image.width, image.height),
                )
                return parsed.get("<CAPTION>", "").strip() or None

            elif _loaded_model_name == "blip":
                inputs = _processor(image, return_tensors="pt").to(_device)
                with torch.no_grad():
                    out = _model.generate(**inputs, max_new_tokens=50)
                return _processor.decode(out[0], skip_special_tokens=True)

            else:  # moondream2
                image_embeds = _model.encode_image(image)
                caption = _model.answer_question(
                    image_embeds,
                    "Describe this image briefly.",
                    _tokenizer,
                )
                return caption.strip() if caption else None

    except Exception:
        return None


def generate_captions_batch(image_paths: list[str]) -> list[Optional[str]]:
    """Batch-generate captions. Moondream2 falls back to sequential (single-image API)."""
    if not image_paths:
        return []

    _load()

    # Moondream2 has a single-image API (encode_image + answer_question) — no batch support
    if _loaded_model_name == "moondream2" or len(image_paths) == 1:
        return [generate_caption(p) for p in image_paths]

    try:
        import torch
        from PIL import Image

        images = []
        indices = []
        for i, path in enumerate(image_paths):
            try:
                images.append(Image.open(path).convert("RGB"))
                indices.append(i)
            except Exception:
                pass

        if not images:
            return [None] * len(image_paths)

        results: list[Optional[str]] = [None] * len(image_paths)

        with _inference_context():
            if _loaded_model_name == "florence2":
                inputs = _processor(
                    text=["<CAPTION>"] * len(images),
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(_device)
                with torch.no_grad():
                    generated_ids = _model.generate(
                        **inputs, max_new_tokens=100, num_beams=3
                    )
                texts = _processor.batch_decode(generated_ids, skip_special_tokens=False)
                for j, orig_idx in enumerate(indices):
                    parsed = _processor.post_process_generation(
                        texts[j],
                        task="<CAPTION>",
                        image_size=(images[j].width, images[j].height),
                    )
                    cap = parsed.get("<CAPTION>", "").strip()
                    results[orig_idx] = cap or None

            elif _loaded_model_name == "blip":
                inputs = _processor(images, return_tensors="pt", padding=True).to(_device)
                with torch.no_grad():
                    out = _model.generate(**inputs, max_new_tokens=50)
                for j, orig_idx in enumerate(indices):
                    results[orig_idx] = _processor.decode(out[j], skip_special_tokens=True)

        return results
    except Exception:
        # Fallback to sequential
        return [generate_caption(p) for p in image_paths]
