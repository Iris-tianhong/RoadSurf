"""Render the visible and inferred road masks and LiDAR support."""

import numpy as np
from PIL import Image, ImageDraw

from core.road_domain import projected_cells


def save_sgsr_views(output, image, calibration, surface, grid, height, lidar_support):
    colors = ((77, 202, 174), (255, 178, 73))
    visible, hidden = surface.visible, surface.hidden
    valid = visible | hidden
    domain = np.full((*visible.shape, 3), (29, 37, 49), dtype=np.uint8)
    domain[visible], domain[hidden] = colors
    support = np.full_like(domain, (29, 37, 49))
    t = lidar_support[valid]
    support[valid] = np.column_stack((35 + 204 * t, 80 + 150 * t, 155 - 79 * t))
    figure = Image.new("RGB", (430, 394), (19, 24, 33))
    draw = ImageDraw.Draw(figure)
    for x, tile, title in ((18, domain, "Road domain"), (225, support, "LiDAR support | 0 to 1")):
        figure.paste(Image.fromarray(tile[::-1, ::-1]).resize((186, 320), Image.Resampling.NEAREST),
                     (x, 28))
        draw.text((x, 10), title, fill="white")
        draw.text((x, 352), "Forward distance up (0-70.4 m)", fill=(210, 219, 232))
    draw.text((18, 371), "Visible", fill=colors[0])
    draw.text((88, 371), "Hidden / inferred", fill=colors[1])
    figure.save(output / "road_domain_support.png")

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for mask, color in zip((visible, hidden), colors):
        if mask.any():
            _, v, u, _ = projected_cells(grid.xy[mask.ravel()], height[mask], calibration, image.size[::-1])
            draw.point(list(zip(u.tolist(), v.tolist())), fill=color)
    overlay.save(output / "road_projection.png")
