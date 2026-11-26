"""Read matched KITTI Object RGB, raw Velodyne, and calibration data."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from data.kitti_calib import KittiCalibration


@dataclass
class KittiFrame:
    """Original RGB image and Nx4 float32 LiDAR points (XYZ in metres, intensity)."""

    frame_id: str
    image: Image.Image
    points: np.ndarray
    calibration: KittiCalibration


def _split_dir(root: Path, split: str) -> Path:
    if split not in ("training", "testing"):
        raise ValueError(f"Unknown KITTI Object split: {split}")
    return root / split


def list_kitti_frames(root: Path, split: str) -> List[str]:
    """List six-digit frame IDs shared by RGB, Velodyne, and calibration."""
    base = _split_dir(root, split)
    inventories = []
    for directory, suffix in (("image_2", ".png"), ("velodyne", ".bin"), ("calib", ".txt")):
        path = base / directory
        if not path.is_dir():
            raise FileNotFoundError(f"Missing KITTI directory: {path}")
        inventories.append({
            file.stem for file in path.glob(f"*{suffix}")
            if file.is_file() and len(file.stem) == 6
            and file.stem.isascii() and file.stem.isdecimal()
        })
    return sorted(set.intersection(*inventories))


def load_kitti_frame(root: Path, split: str, frame: Optional[int] = None) -> KittiFrame:
    """Load the requested frame or the first complete frame in the split."""
    base = _split_dir(root, split)
    if frame is None:
        matched = list_kitti_frames(root, split)
        if not matched:
            raise FileNotFoundError(f"No matched RGB, Velodyne, and calibration frames: {base}")
        frame_id = matched[0]
    else:
        if not 0 <= frame <= 999999:
            raise ValueError("KITTI frame ID must be in [0, 999999].")
        frame_id = f"{frame:06d}"

    rgb_path = base / "image_2" / f"{frame_id}.png"
    lidar_path = base / "velodyne" / f"{frame_id}.bin"
    calib_path = base / "calib" / f"{frame_id}.txt"
    with Image.open(rgb_path) as rgb:
        image = rgb.convert("RGB")
    byte_count = lidar_path.stat().st_size
    if byte_count == 0 or byte_count % 16:
        raise ValueError(f"Expected a nonempty KITTI LiDAR file with four float32 columns: {lidar_path}")
    points = np.fromfile(lidar_path, dtype="<f4").reshape(-1, 4)
    return KittiFrame(frame_id, image, points, KittiCalibration(calib_path))
