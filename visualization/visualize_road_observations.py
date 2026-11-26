"""Visualize Phase 2 road observations in camera and LiDAR coordinates."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_depth_previews(
    output: Path, image: Image.Image, relative: np.ndarray, calibrated: np.ndarray
) -> None:
    finite = np.isfinite(relative)
    preview = np.zeros(relative.shape, dtype=np.uint8)
    if finite.any():
        low, high = np.percentile(relative[finite], [2, 98])
        span = max(np.finfo(float).eps, high - low)
        preview[finite] = np.rint(255 * np.clip((relative[finite] - low) / span, 0, 1))
    Image.fromarray(preview).save(output / "dav2_relative_preview.png")

    valid = np.isfinite(calibrated)
    overlay = np.array(image)
    if valid.any():
        t = np.clip(calibrated[valid] / np.percentile(calibrated[valid], 98), 0, 1)
        color = np.column_stack((255 - 220 * t, 95 + 125 * t, 30 + 225 * t))
        overlay[valid] = np.rint(0.45 * overlay[valid] + 0.55 * color)
    Image.fromarray(overlay).save(output / "visual_road_projection.png")


def save_road_observations(output: Path, lidar: np.ndarray, visual: np.ndarray) -> None:
    canvas = Image.new("RGB", (1110, 465), (19, 25, 33))
    draw = ImageDraw.Draw(canvas)
    lidar_color, visual_color = (255, 184, 77), (71, 207, 224)
    draw.text((64, 15), "LiDAR observations", fill=lidar_color)
    draw.text((225, 15), "DA-V2 visual points", fill=visual_color)

    heights = np.concatenate((lidar[:, 2], visual[:, 2]))
    z_lo = min(-2.0, float(np.floor(heights.min()))) if heights.size else -2.0
    z_hi = max(0.0, float(np.ceil(heights.max()))) if heights.size else 0.0
    # Both panels use the same forward scale.
    for x0, axis, lo, hi, title in (
        (64, 1, -20, 20, "Top view  |  LiDAR y (m)"),
        (610, 2, z_lo, z_hi, "Longitudinal profile  |  LiDAR z (m)"),
    ):
        x1, y0, y1 = x0 + 455, 67, 409
        draw.text((x0, 47), title, fill="white")
        for value in range(0, 71, 10):
            x = round(x0 + (x1 - x0) * value / 70.4)
            draw.line((x, y0, x, y1), fill=(51, 62, 74))
            draw.text((x - 5, y1 + 6), str(value), fill=(205, 214, 223))
        for value in np.linspace(lo, hi, 5):
            y = round(y1 - (y1 - y0) * (value - lo) / (hi - lo))
            draw.line((x0, y, x1, y), fill=(51, 62, 74))
            draw.text((x0 - 38, y - 5), f"{value:g}", fill=(205, 214, 223))
        draw.text((x0 + 160, y1 + 23), "Forward distance x (m)", fill="white")

        for points, color in ((visual, visual_color), (lidar, lidar_color)):
            if not len(points):
                continue
            cols = x0 + (x1 - x0) * points[:, 0] / 70.4
            rows = y1 - (y1 - y0) * (points[:, axis] - lo) / (hi - lo)
            pixels = np.rint(np.column_stack((cols, rows))).astype(np.intp)
            draw.point(list(map(tuple, pixels.tolist())), fill=color)
    canvas.save(output / "road_observations.png")
