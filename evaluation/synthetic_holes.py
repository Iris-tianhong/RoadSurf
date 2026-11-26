"""Choose rectangular holdouts before reconstruction, using observable road only."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter


HOLE_SIZES_XY = ((4, 2), (6, 3))


@dataclass
class SyntheticHole:
    center: np.ndarray
    size_xy: tuple

    @property
    def name(self):
        length, width = self.size_xy
        return f"hole_{width}x{length}"

    def contains(self, points, margin=0.0):
        half_size = np.asarray(self.size_xy) / 2 - margin
        return np.all(np.abs(points[:, :2] - self.center) < half_size, axis=1)


def choose_hole(grid, visible, size_xy):
    """A road-interior site with a measured collar, deterministic across beam settings."""
    # An even filter uses an asymmetric window by one cell. Pad the
    # physical requirement by .8 m on each axis to keep that distinction away
    # from the deleted rectangle.
    window = np.ceil((np.asarray(size_xy) + .8) / grid.resolution).astype(int) | 1
    surrounding = uniform_filter(visible.astype(float), size=window, mode="constant")
    eligible = (surrounding.ravel() >= .97) & (grid.xy[:, 0] >= 8) & (grid.xy[:, 0] <= 28)
    if not eligible.any():
        return None
    sites = grid.xy[eligible]
    priority = (sites[:, 0] - 15)**2 + 3 * sites[:, 1]**2
    return SyntheticHole(sites[np.argmin(priority)].copy(), size_xy)
