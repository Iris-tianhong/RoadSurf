"""Load cached Cityscapes labels and align them to the original camera image."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


IGNORE = 255
ROAD = 0
SIDEWALK = 1
# Cityscapes label IDs and RGB colors, in PointPainting's trainId order.
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
RGB_COLORS = (
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32),
)
VOID_COLORS = (
    (0, 0, 0), (111, 74, 0), (81, 0, 81), (250, 170, 160), (230, 150, 140),
    (180, 165, 180), (150, 100, 100), (150, 120, 90), (0, 0, 90), (0, 0, 110),
)


def _read(path, label_format, npz_key):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        if npz_key is None:
            raise ValueError("Specify --semantic-key for .npz files (e.g. classes).")
        with np.load(path, allow_pickle=False) as archive:
            return archive[npz_key]
    if suffix not in {".png", ".bmp", ".tif", ".tiff"}:
        raise ValueError("Use a lossless label image, .npy, or .npz.")
    with Image.open(path) as image:
        if image.mode == "P" and label_format == "auto":
            raise ValueError("Palette image: specify whether to use indices or RGB colors.")
        if image.mode == "P" and label_format == "color":
            image = image.convert("RGB")
        return np.array(image)


def _decode_color(rgb):
    if rgb.dtype != np.uint8:
        raise ValueError("The semantic RGB palette requires uint8.")
    labels = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
    known = np.zeros(rgb.shape[:2], dtype=bool)
    for train_id, color in enumerate(RGB_COLORS):
        match = np.all(rgb == color, axis=2)
        labels[match] = train_id
        known |= match
    for color in VOID_COLORS:
        known |= np.all(rgb == color, axis=2)
    if not np.all(known):
        raise ValueError(
            f"{np.count_nonzero(~known)} pixels are outside the Cityscapes palette. "
            "Use the original prediction cache for blended or interpolated visualizations."
        )
    return labels


def _decode_indices(indices, label_format):
    if indices.dtype.kind not in "iu":
        raise ValueError("Index maps must contain integer class IDs.")
    values = np.unique(indices)
    train_ids = np.isin(values, (*range(19), -1, IGNORE)).all()
    label_ids = np.isin(values, (*range(34), -1, IGNORE)).all()
    if label_format == "auto":
        if train_ids:
            raise ValueError("Ambiguous index map: specify --semantic-format trainId or labelId.")
        label_format = "labelId"
    if label_format == "trainId" and train_ids:
        return indices.astype(np.uint8), label_format
    if label_format == "labelId" and label_ids:
        lookup = np.full(256, IGNORE, dtype=np.uint8)
        lookup[list(LABEL_IDS)] = np.arange(19, dtype=np.uint8)
        return lookup[indices.astype(np.uint8)], label_format
    raise ValueError(f"Invalid pixel values for {label_format}: {values[:40].tolist()}")


def load_semantic_labels(
    path: Path,
    image_size: Tuple[int, int],
    label_format: str = "auto",
    npz_key: Optional[str] = None,
    crop: Optional[Tuple[int, int, int, int]] = None,
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Tuple[np.ndarray, dict]:
    """Return trainId labels and source details. All rectangles are in pixels.

    Forward preprocessing order: RGB crop, resize to the unpadded label extent,
    then pad. Crop is (x, y, width, height) in RGB coordinates. Padding is
    (left, top, right, bottom) in the saved label grid. Unseen RGB pixels are 255.
    """
    source = _read(Path(path), label_format, npz_key)
    if not source.size or source.ndim not in (2, 3):
        raise ValueError(f"Invalid semantic map shape: {source.shape}")
    width, height = image_size
    x, y, crop_width, crop_height = crop or (0, 0, width, height)
    if (min(x, y) < 0 or min(crop_width, crop_height) <= 0
            or x + crop_width > width or y + crop_height > height):
        raise ValueError("The semantic crop must be inside the original RGB image.")
    left, top, right, bottom = padding
    source_height, source_width = source.shape[:2]
    if (min(padding) < 0 or left + right >= source_width or top + bottom >= source_height):
        raise ValueError("Invalid semantic padding for the saved label grid.")
    content = source[top:source_height - bottom, left:source_width - right]
    details = {"size_wh": [source_width, source_height], "dtype": str(source.dtype)}

    if content.ndim == 3 and content.shape[2] == 1:
        content = content[:, :, 0]
    if content.ndim == 3 and content.shape[2] == 3 and label_format != "color":
        values = content[:, :, 0]
        if np.all(content == content[:, :, :1]) and (
            label_format != "auto" or np.isin(np.unique(values), (*range(34), -1, IGNORE)).all()
        ):
            content = content[:, :, 0]
    if label_format == "probabilities":
        if content.ndim != 3 or content.shape[2] != 19 or content.dtype.kind != "f":
            raise ValueError("Expected a floating-point HxWx19 Cityscapes probability cache.")
        if (not np.isfinite(content).all() or content.min() < 0 or content.max() > 1
                or not np.allclose(content.sum(axis=2, dtype=np.float32), 1, atol=0.005)):
            raise ValueError("Invalid Cityscapes probability cache.")
        labels = content.argmax(axis=2).astype(np.uint8)
    elif content.ndim == 2:
        labels, label_format = _decode_indices(content, label_format)
        details["source_ids"] = np.unique(content).tolist()
    elif (content.ndim == 3 and content.shape[2] in (3, 4)
          and label_format in ("auto", "color")):
        if content.shape[2] == 4 and np.any(content[:, :, 3] != 255):
            raise ValueError("The semantic color image contains non-opaque pixels.")
        labels = _decode_color(content[:, :, :3])
        label_format = "color"
    else:
        raise ValueError("Expected a 2D index map or an exact Cityscapes RGB palette.")

    restored = np.full((height, width), IGNORE, dtype=np.uint8)
    resized = Image.fromarray(labels).resize(
        (crop_width, crop_height), resample=Image.Resampling.NEAREST)
    restored[y:y + crop_height, x:x + crop_width] = np.asarray(resized)
    details.update(format=label_format, crop_xywh=[x, y, crop_width, crop_height],
                   padding_ltrb=list(padding), output_train_ids=np.unique(restored).tolist())
    return restored, details
