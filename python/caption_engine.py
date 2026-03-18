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
    _loaded_model_name = model_name


def generate_caption(image_path: str) -> Optional[str]:
    """Generate a natural language caption for the image. Returns None on error."""
    try:
        _load()
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

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
