"""
Utilities for running DeepSeek OCR (via Unsloth) to turn images into text.

This is intentionally lightweight: the model is only downloaded/loaded when
`DeepSeekOCR.run` is called. You can point `model_id` at a local path or a
Hugging Face repo ID; we default to the public `unsloth/DeepSeek-OCR`.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Union

try:
    from PIL import Image
except ImportError:
    Image = None  # Delayed import for lightweight environments

ImageInput = Union[str, bytes, "Image.Image"]


@dataclass
class DeepSeekOCR:
    """Lazy loader/wrapper around the Unsloth DeepSeek OCR vision model."""

    model_id: str = "unsloth/DeepSeek-OCR"
    local_dir: str = os.path.expanduser("~/.npcsh/models/deepseek_ocr")
    load_in_4bit: bool = False
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True

    def __post_init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def _ensure_weights(self) -> str:
        """Download weights if they are not already on-disk."""
        if os.path.isdir(self.local_dir) and os.listdir(self.local_dir):
            return self.local_dir

        os.makedirs(self.local_dir, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download DeepSeek OCR weights. "
                "Install with `pip install huggingface_hub` or pre-download manually."
            ) from exc

        snapshot_download(self.model_id, local_dir=self.local_dir)
        return self.local_dir

    def _load_model(self) -> None:
        """Load the Unsloth vision model once (lazy)."""
        if self._model is not None and self._tokenizer is not None:
            return

        weights_dir = self._ensure_weights()
        os.environ.setdefault("UNSLOTH_WARN_UNINITIALIZED", "0")

        try:
            from unsloth import FastVisionModel
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "unsloth and transformers are required to run DeepSeek OCR. "
                "Install with `pip install unsloth transformers` (and bitsandbytes if using 4bit)."
            ) from exc

        self._model, self._tokenizer = FastVisionModel.from_pretrained(
            weights_dir,
            load_in_4bit=self.load_in_4bit,
            auto_model=AutoModel,
            trust_remote_code=True,
            unsloth_force_compile=True,
            use_gradient_checkpointing="unsloth",
        )

    def _prepare_image_file(self, image: ImageInput) -> tuple[str, bool]:
        """Normalize various image inputs to a file path and say if we should clean it up."""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image path does not exist: {image}")
            return image, False

        if Image is None:
            raise ImportError("Pillow is required for OCR image handling. Install with `pip install pillow`.")

        if isinstance(image, bytes):
            import io

            pil = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pil.save(tmp, format="PNG")
        tmp.close()
        return tmp.name, True

    def run(
        self,
        image: ImageInput,
        prompt: str = "<image>\nFree OCR. ",
        output_path: Optional[str] = None,
        save_results: bool = False,
        test_compress: bool = False,
        **kwargs,
    ) -> str:
        """
        Run OCR on an image and return the recognized text.

        Args:
            image: Path, bytes, or PIL Image.
            prompt: Prompt passed to the vision model (keeps the default used
                in the reference notebook).
            output_path: Optional directory for saving debug outputs.
            save_results: If True, Unsloth will save visualization artifacts.
            test_compress: Forwarded to `model.infer`.
            kwargs: Additional overrides for infer (base_size, image_size, etc).
        """
        self._load_model()

        image_file, should_cleanup = self._prepare_image_file(image)
        infer_kwargs = {
            "prompt": prompt,
            "image_file": image_file,
            "output_path": output_path or "",
            "base_size": kwargs.pop("base_size", self.base_size),
            "image_size": kwargs.pop("image_size", self.image_size),
            "crop_mode": kwargs.pop("crop_mode", self.crop_mode),
            "save_results": save_results,
            "test_compress": test_compress,
        }

        try:
            result = self._model.infer(self._tokenizer, **infer_kwargs)
        finally:
            # Clean up temp files created from bytes/PIL inputs.
            if should_cleanup and os.path.exists(image_file):
                try:
                    os.remove(image_file)
                except OSError:
                    pass

        # Unsloth infer returns a dict-like object; stringify for callers.
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict) and "text" in result:
            return str(result["text"]).strip()
        return str(result).strip()


def deepseek_ocr(
    image: ImageInput,
    prompt: str = "<image>\nFree OCR. ",
    model_id: str = "unsloth/DeepSeek-OCR",
    local_dir: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Functional wrapper that mirrors the reference notebook defaults.

    Example:
        text = deepseek_ocr(\"invoice.png\")
    """
    runner = DeepSeekOCR(
        model_id=model_id,
        local_dir=local_dir or os.path.expanduser("~/.npcsh/models/deepseek_ocr"),
        load_in_4bit=kwargs.pop("load_in_4bit", False),
        base_size=kwargs.pop("base_size", 1024),
        image_size=kwargs.pop("image_size", 640),
        crop_mode=kwargs.pop("crop_mode", True),
    )
    return runner.run(
        image=image,
        prompt=prompt,
        output_path=kwargs.pop("output_path", None),
        save_results=kwargs.pop("save_results", False),
        test_compress=kwargs.pop("test_compress", False),
        **kwargs,
    )
