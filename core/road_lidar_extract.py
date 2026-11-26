"""Intersect visible image-road returns with Patchwork++ ground indices."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

from data.kitti_calib import KittiCalibration
from frontends.semantic_loader import ROAD


ROAD_ROI_XY = ((0, 70.4), (-20, 20))


def road_roi_mask(points: np.ndarray) -> np.ndarray:
    bounds = np.asarray(ROAD_ROI_XY)
    xy = points[:, :2]
    return ((xy >= bounds[:, 0]) & (xy <= bounds[:, 1])).all(axis=1)


def road_interior(labels: np.ndarray, boundary_margin: int) -> np.ndarray:
    if boundary_margin < 0:
        raise ValueError("Semantic boundary margin must be nonnegative.")
    mask = labels == ROAD
    if boundary_margin:
        image = Image.fromarray(mask.astype(np.uint8))
        mask = np.asarray(image.filter(ImageFilter.MinFilter(2 * boundary_margin + 1)), dtype=bool)
    return mask


def _local_surface_inliers(xyz: np.ndarray) -> np.ndarray:
    """Reject height jumps and outliers only where a local plane is supported."""
    points = xyz.astype(np.float64)
    xy, z = points[:, :2], points[:, 2]
    cells = np.floor(xy / 2).astype(np.intp)
    grid, cell_ids = np.unique(cells, axis=0, return_inverse=True)
    keep = np.ones(len(points), dtype=bool)
    for cell_id, cell in enumerate(grid):
        offset = xy - (2 * cell + 1)
        near = np.max(np.abs(offset), axis=1) <= 3
        if near.sum() < 12:
            continue
        design = np.column_stack((offset[near], np.ones(near.sum())))
        heights = z[near]
        coefficients, _, rank, _ = np.linalg.lstsq(design, heights, rcond=None)
        if rank < 3:
            continue
        residual = heights - design @ coefficients
        median = np.median(residual)
        mad = np.median(np.abs(residual - median))
        inliers = np.abs(residual - median) <= max(0.035, 2.5 * 1.4826 * mad)
        if inliers.sum() < 12:
            continue
        coefficients, _, rank, _ = np.linalg.lstsq(design[inliers], heights[inliers], rcond=None)
        if rank < 3:
            continue
        fit_error = np.abs(heights[inliers] - design[inliers] @ coefficients)
        tolerance = max(0.045, 3 * 1.4826 * np.median(fit_error))
        own = cell_ids == cell_id
        predicted = offset[own] @ coefficients[:2] + coefficients[2]
        keep[own] = np.abs(z[own] - predicted) <= tolerance
    return keep


def load_patchwork_mask(path: Path, point_count: int) -> np.ndarray:
    """Read the VirConv ground cache indexed by the original Velodyne row."""
    with np.load(path, allow_pickle=False) as cache:
        mask = cache["ground_mask"]
        indices = cache["ground_idx"]
    if (mask.shape != (point_count,) or mask.dtype != np.bool_
            or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or not np.array_equal(np.flatnonzero(mask), np.sort(indices))):
        raise ValueError(f"Patchwork++ cache does not match the original scan: {path}")
    return mask


def estimate_patchwork_mask(points: np.ndarray) -> np.ndarray:
    try:
        import pypatchworkpp
    except ImportError as exc:
        raise RuntimeError("Provide a Patchwork++ cache or install pypatchworkpp.") from exc
    finite = np.flatnonzero(np.isfinite(points).all(axis=1))
    mask = np.zeros(len(points), dtype=bool)
    if finite.size:
        parameters = pypatchworkpp.Parameters()
        parameters.sensor_height = 1.723
        parameters.uprightness_thr = 0.707
        parameters.min_range = 2.7
        parameters.max_range = 80.0
        parameters.verbose = False
        patchwork = pypatchworkpp.patchworkpp(parameters)
        patchwork.estimateGround(points[finite])
        indices = np.asarray(patchwork.getGroundIndices(), dtype=np.intp).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= finite.size):
            raise ValueError("Patchwork++ returned out-of-range ground indices.")
        mask[finite[indices]] = True
    return mask


def extract_road_lidar(
    points: np.ndarray,
    calibration: KittiCalibration,
    labels: np.ndarray,
    ground: np.ndarray,
    boundary_margin: int = 2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return original Nx4 LiDAR rows, integer Nx2 image pixels, and counts."""
    height, width = labels.shape
    finite = np.flatnonzero(np.isfinite(points).all(axis=1))
    visible, pixels, _ = calibration.visible_pixels(points[finite], (width, height))
    visible = finite[visible]
    in_roi = road_roi_mask(points[visible])
    u, v = pixels.T
    semantic = in_roi & (labels[v, u] == ROAD)
    intersection = semantic & ground[visible]

    # Leave a conservative margin at semantic transitions, including curbs
    # and vehicle silhouettes where a small projection error changes the class.
    interior = road_interior(labels, boundary_margin)
    interior_ids = np.flatnonzero(intersection & interior[v, u])
    clean = _local_surface_inliers(points[visible[interior_ids], :3])
    keep = interior_ids[clean]
    stats = {
        "raw_points": len(points),
        "finite_points": len(finite),
        "visible_points_in_roi": int(in_roi.sum()),
        "semantic_road_points": int(semantic.sum()),
        "patchwork_ground_points_raw": int(ground.sum()),
        "road_points_after_geometry": int(intersection.sum()),
        "road_points_after_boundary": len(interior_ids),
        "road_points_final": len(keep),
        "semantic_boundary_margin_px": boundary_margin,
    }
    return points[visible[keep]], pixels[keep], stats
