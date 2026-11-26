"""Depth Anything V2 Small relative prediction at the original RGB resolution."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


DAV2_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"


def load_relative_depth(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValueError("Use the original floating-point relative-depth .npy cache.")
    values = np.load(path, allow_pickle=False)
    if values.shape != image_size[::-1] or values.dtype.kind != "f":
        raise ValueError(f"Expected a floating-point relative-depth map of shape {image_size[::-1]}.")
    return values


def load_dav2_predictor(model_source: str = DAV2_SMALL, device: str = "auto"):
    try:
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError("DA-V2 inference requires PyTorch and Transformers; see README.") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_source, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(model_source, use_safetensors=True)
    if model.config.depth_estimation_type != "relative":
        raise ValueError("Expected the relative-depth DA-V2 Small checkpoint.")
    model = model.to(device).eval()
    revision_path = Path(model_source) / "revision.txt"
    revision = (revision_path.read_text(encoding="ascii").strip() if revision_path.is_file()
                else getattr(model.config, "_commit_hash", None))
    provenance = {
        "kind": "dav2_small_inference",
        "model": model_source,
        "revision": revision,
        "device": device,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
    }

    def predict(image: Image.Image) -> np.ndarray:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            restored = processor.post_process_depth_estimation(
                model(**inputs), target_sizes=[(image.height, image.width)])
            return restored[0]["predicted_depth"].cpu().numpy().astype(np.float32, copy=False)

    return predict, provenance


def predict_relative_depth(
    image: Image.Image, model_source: str = DAV2_SMALL, device: str = "auto"
) -> Tuple[np.ndarray, dict]:
    predict, provenance = load_dav2_predictor(model_source, device)
    return predict(image), provenance
