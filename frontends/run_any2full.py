"""Call the unmodified Any2Full Small network with a sparse metric prompt."""

from argparse import Namespace
from hashlib import sha256
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image


OFFICIAL_VITS_SHA256 = "c797e3b93a1b520033cb7d11a9d6a313326dc48c1ead0c4b417ac6570c95a03b"


def load_any2full_predictor(source: Path, checkpoint: Path, device: str):
    import torch
    from torchvision.transforms import Compose, Normalize, ToTensor

    if not (source / "model/ours/any2full.py").is_file():
        raise FileNotFoundError(f"Expected the official Any2Full source tree: {source}")
    sys.path.insert(0, str(source.resolve()))
    try:
        from model.ours.any2full import Any2Full
    finally:
        sys.path.pop(0)
    # Use the upstream inference defaults and its native metric scaling.
    options = Namespace(init_scailing=True, max_depth=1e3, min_depth=1e-6, stage=1)
    model = Any2Full(encoder="vits", da_ckpt_path=None, args=options)
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # The official archive contains NumPy training metadata which the
        # weights-only loader in torch 2.3 cannot read.
        if sha256(checkpoint.read_bytes()).hexdigest() != OFFICIAL_VITS_SHA256:
            raise
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("state_dict", checkpoint_data)
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()}, strict=True)
    model.to(device).eval()
    rgb_transform = Compose((ToTensor(), Normalize((.485, .456, .406), (.229, .224, .225))))

    def predict(image: Image.Image, sparse_depth: np.ndarray) -> np.ndarray:
        if sparse_depth.shape != image.size[::-1] or not np.any(sparse_depth > 0):
            raise ValueError("Any2Full requires an image-aligned positive sparse-depth prompt.")
        rgb = rgb_transform(image).unsqueeze(0).to(device)
        prompt = np.where(np.isfinite(sparse_depth), sparse_depth, 0).astype(np.float32)
        depth = torch.from_numpy(prompt)[None, None].to(device)
        with torch.inference_mode():
            result = model({"rgb": rgb, "dep": depth})["pred"][0, 0].cpu().numpy()
        if result.shape != sparse_depth.shape or not np.isfinite(result).any():
            raise ValueError("Any2Full output does not match the original RGB grid.")
        return result

    return predict
