"""Shared-scale road height maps and an RGB domain check."""

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


def save_height_comparison(output: Path, heights: np.ndarray, names: Sequence[str]) -> None:
    canvas = Image.new("RGB", (916, 483), (20, 26, 34))
    draw = ImageDraw.Draw(canvas)
    finite = np.isfinite(heights)
    z_min, z_max = heights[finite].min(), heights[finite].max()
    span = max(z_max - z_min, 1e-10)
    panel_x = (26, 324, 622)
    for left, height, name in zip(panel_x, heights, names):
        draw.text((left, 12), name, fill="white")
        # Camera-forward x goes up; vehicle-left y goes to image left.
        raster = height[::-1, ::-1]
        valid = np.isfinite(raster)
        layer = np.full((*raster.shape, 3), (32, 41, 52), dtype=np.uint8)
        t = (raster[valid] - z_min) / span
        layer[valid] = np.rint(np.column_stack((55 + 235 * t - 37 * t**2,
                                                118 + 79 * t, 220 - 163 * t))).astype(np.uint8)
        panel = Image.fromarray(layer).resize((211, 371), Image.Resampling.NEAREST)
        canvas.paste(panel, (left + 32, 34))
        draw.rectangle((left + 32, 34, left + 243, 405), outline=(67, 81, 99))
        for value in range(0, 71, 10):
            y = round(405 - value / 70.4 * 371)
            draw.text((left + 10, y - 4), str(value), fill=(177, 190, 205))
        draw.text((left + 32, 413), "+20", fill=(211, 219, 231))
        draw.text((left + 122, 413), "y (m)", fill=(211, 219, 231))
        draw.text((left + 226, 413), "-20", fill=(211, 219, 231))
    draw.text((36, 452), f"LiDAR z (m)   blue {z_min:.2f}   gold {z_max:.2f}     dark cells outside valid domain",
              fill=(230, 236, 244))
    canvas.save(output / "road_height.png")


def save_surface_projection(output: Path, rgb: Image.Image, pixels: np.ndarray) -> None:
    image = rgb.copy()
    ImageDraw.Draw(image).point(list(map(tuple, pixels.tolist())), fill=(60, 255, 162))
    image.save(output / "road_projection.png")
