"""Separate camera-visible road from tightly bounded missing road cells."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from core.hierarchical_surface import RoadSurface
from core.sgsr_solver import observed_road_grid


def projected_cells(xy, z, calibration, shape):
    uv, depth = calibration.project_lidar_to_image(np.column_stack((xy, z)))
    h, w = shape
    ids = np.flatnonzero(((uv >= 0) & (uv < [w, h])).all(axis=1))
    u, v = np.floor(uv[ids]).astype(np.intp).T
    return ids, v, u, depth[ids]


def bounded_missing(visible: np.ndarray, resolution: float) -> np.ndarray:
    """Propose compact holes bracketed by visible cells along both grid axes."""
    radius = int(np.floor(.45 / resolution))
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
    closed = ndimage.binary_closing(visible, structure=x*x + y*y <= radius*radius) | visible
    candidates = ndimage.binary_fill_holes(closed) & ~visible
    # Morphological closing can add exterior cells. Require real visible
    # cells on all four sides before measuring each component's footprint.
    for axis in (0, 1):
        candidates &= np.logical_or.accumulate(visible, axis=axis)
        candidates &= np.flip(np.logical_or.accumulate(np.flip(visible, axis=axis), axis=axis), axis=axis)
    components, _ = ndimage.label(candidates)
    sizes = np.bincount(components.ravel())
    accepted = np.zeros(len(sizes), dtype=bool)
    for index, window in enumerate(ndimage.find_objects(components), start=1):
        span_x, span_y = [(part.stop - part.start) * resolution for part in window]
        accepted[index] = span_x <= 6.4 and span_y <= 3.4 and sizes[index] * resolution**2 <= 20
    return accepted[components] & (ndimage.distance_transform_edt(~visible) * resolution <= 1.8)


@dataclass
class DomainGrid:
    xy: np.ndarray
    shape: tuple
    nearest: np.ndarray
    has_observation: np.ndarray
    origin: np.ndarray
    resolution: float

    @classmethod
    def make(cls, lidar, visual, resolution=.20):
        x, y, nearest = observed_road_grid(lidar, resolution, 1.25)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        candidate = np.flatnonzero(nearest.ravel() >= 0)
        observed = np.zeros(len(xy), dtype=bool)
        sources = np.vstack((lidar[:, :2], visual[:, :2]))
        distance, _ = cKDTree(sources).query(xy[candidate], distance_upper_bound=.45)
        observed[candidate] = np.isfinite(distance)
        return cls(xy, xx.shape, nearest.ravel(), observed,
                   np.array([x[0], y[0]]) - resolution / 2, resolution)

    def visible_for(self, model, lidar, calibration, image_road):
        chosen = np.flatnonzero(self.has_observation)
        xy = self.xy[chosen]
        z, _ = model.evaluate(xy)
        ids, v, u, _ = projected_cells(xy, z, calibration, image_road.shape)
        hits = chosen[ids[image_road[v, u]]]
        anchors = lidar[self.nearest[hits]]
        agree = np.abs(model.evaluate(anchors[:, :2])[0] - anchors[:, 2]) <= .30
        visible = np.zeros(len(self.xy), dtype=bool)
        visible[hits[agree]] = True
        return visible.reshape(self.shape)

    def hidden_for(self, model, visible, calibration, labels, road, sparse, synthetic=False):
        allowed = bounded_missing(visible, self.resolution)
        indices = np.flatnonzero(allowed)
        hidden = np.zeros(len(self.xy), dtype=bool)
        if not len(indices):
            return hidden.reshape(self.shape)
        xy = self.xy[indices]
        z, _ = model.evaluate(xy)
        ids, v, u, depth = projected_cells(xy, z, calibration, labels.shape)
        if synthetic:
            # The evaluator's deleted rectangle is never passed to the domain.
            accept = road[v, u]
        else:
            foreground = (labels >= 11) & (labels <= 18)
            nearer_depth = ndimage.minimum_filter(
                np.where(foreground & np.isfinite(sparse), sparse, np.inf), size=7,
                mode="constant", cval=np.inf)
            accept = foreground[v, u] & (depth >= nearer_depth[v, u] + .35)
        hidden[indices[ids[accept]]] = True
        return hidden.reshape(self.shape)

    def make_surface(self, model, visible, hidden, support):
        return RoadSurface(model, self.origin, self.resolution, visible, hidden, support)
