"""Phase 3 plane and single-level, float64 surface fitting."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree

from core.bspline_basis import TensorSplineBasis, clamped_knots
from core.road_lidar_extract import ROAD_ROI_XY
from data.kitti_calib import KittiCalibration


def plane_height(xy: np.ndarray, plane: np.ndarray) -> np.ndarray:
    return xy[:, 0] * plane[0] + xy[:, 1] * plane[1] + plane[2]


def fit_ransac_plane(
    points: np.ndarray, threshold: float = 0.12, trials: int = 256, seed: int = 42
) -> Tuple[np.ndarray, int]:
    """RANSAC height plane, with the height-residual inliers refit in SciPy."""
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    if len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("RANSAC requires at least three finite LiDAR points.")
    rng = np.random.default_rng(seed)
    best = np.empty(0, dtype=np.intp)
    best_score = (0, -np.inf)
    for _ in range(trials):
        p, q, r = xyz[rng.choice(len(xyz), 3, replace=False)]
        normal = np.cross(q - p, r - p)
        if abs(normal[2]) <= 1e-8:
            continue
        slopes = -normal[:2] / normal[2]
        plane = np.r_[slopes, p[2] - p[:2] @ slopes]
        errors = np.abs(plane_height(xyz[:, :2], plane) - xyz[:, 2])
        ids = np.flatnonzero(errors <= threshold)
        if len(ids) < max(3, best_score[0]):
            continue
        score = (len(ids), -float(np.median(errors[ids])))
        if score > best_score:
            best, best_score = ids, score
    if len(best) < 3:
        raise ValueError("RANSAC could not identify a height plane.")
    inliers = xyz[best]
    design = np.column_stack((inliers[:, :2], np.ones(len(best))))
    plane, _, rank, _ = lstsq(design, inliers[:, 2])
    if rank < 3:
        raise ValueError("RANSAC inliers do not span a height plane.")
    return plane, len(best)


def _visual_cell_indices(visual: np.ndarray, spacing: float = 0.4) -> np.ndarray:
    grid = np.floor(visual[:, :2] / spacing).astype(np.intp)
    _, cell, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    order = np.lexsort((visual[:, 2], cell))
    ends = np.cumsum(counts)
    mid = ends - (counts + 1) // 2
    # Use a measured visual sample with median (lower-middle for an even bin) z.
    return order[mid]


@dataclass
class SingleSplineFit:
    coefficients: np.ndarray
    visual_fit_samples: int
    solver_iterations: int


def fit_single_spline(
    lidar: np.ndarray, basis: TensorSplineBasis, plane: np.ndarray,
    visual: Optional[np.ndarray] = None, visual_weight: float = 0.25,
    smoothness: float = 2.0, ridge: float = 0.001,
    prior_height: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    visual_reliability: Optional[np.ndarray] = None,
    extra_damping: Optional[np.ndarray] = None,
) -> SingleSplineFit:
    """Fit a robust surface residual with independent, normalized modality terms."""
    if not (np.isfinite([visual_weight, smoothness, ridge]).all()
            and visual_weight > 0 and smoothness >= 0 and ridge > 0):
        raise ValueError("Invalid spline weights.")
    lidar = np.asarray(lidar[:, :3], dtype=np.float64)
    groups = [(lidar, 1.0, 0.10, None)]
    visual_count = 0
    if visual is not None and len(visual):
        visual = np.asarray(visual[:, :3], dtype=np.float64)
        selected = _visual_cell_indices(visual)
        visual_count = len(selected)
        modifiers = None if visual_reliability is None else visual_reliability[selected]
        groups.append((visual[selected], visual_weight, 0.18, modifiers))
    matrices, targets, weights, deltas = [], [], [], []
    for points, strength, delta, modifiers in groups:
        xy = points[:, :2]
        matrices.append(basis.design(xy))
        reference = plane_height(xy, plane) if prior_height is None else prior_height(xy)
        targets.append(points[:, 2] - reference)
        _, cell, counts = np.unique(np.floor(xy), axis=0, return_inverse=True, return_counts=True)
        terms = strength / (len(counts) * counts[cell])
        if modifiers is not None:
            terms *= modifiers
            total = terms.sum()
            if not np.isfinite(total) or total <= 0 or np.any(terms < 0):
                raise ValueError("Visual reliability must give finite, positive total weight.")
            terms *= strength / total
        weights.append(terms)
        deltas.append(np.full(len(points), delta))
    design = vstack(matrices, format="csr")
    residual_z = np.concatenate(targets)
    initial_weight = np.concatenate(weights)
    delta = np.concatenate(deltas)
    curvature = basis.curvature()
    regularizer = curvature * np.sqrt(smoothness / curvature.shape[0])
    if extra_damping is not None:
        regularizer = vstack((regularizer, diags(extra_damping.ravel())), format="csr")
    weight = initial_weight
    for _ in range(3):
        root_weight = np.sqrt(weight)
        system = vstack((design.multiply(root_weight[:, None]), regularizer), format="csr")
        target = np.r_[root_weight * residual_z, np.zeros(regularizer.shape[0])]
        result = lsmr(system, target, damp=np.sqrt(ridge / design.shape[1]),
                      atol=1e-7, btol=1e-7, maxiter=4 * design.shape[1])
        coefficients, status, iterations = result[:3]
        if status in (3, 6, 7) or not np.isfinite(coefficients).all():
            raise ValueError(f"Single-level spline solver failed (LSMR status {status}).")
        error = np.abs(design @ coefficients - residual_z)
        weight = initial_weight * delta / np.maximum(delta, error)
    return SingleSplineFit(coefficients.reshape(basis.shape), visual_count, iterations)


def make_spline_basis(spacing: float) -> TensorSplineBasis:
    x_roi, y_roi = ROAD_ROI_XY
    return TensorSplineBasis(clamped_knots(x_roi, spacing), clamped_knots(y_roi, spacing))


def observed_road_grid(
    lidar: np.ndarray, resolution: float, radius: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid axes and nearest measurement IDs (-1 outside the observation envelope)."""
    if not np.isfinite([resolution, radius]).all() or min(resolution, radius) <= 0:
        raise ValueError("Grid spacing and observation radius must be positive.")
    axes = [lo + resolution * (np.arange(int(np.floor((hi - lo) / resolution + 1e-9))) + 0.5)
            for lo, hi in ROAD_ROI_XY]
    x, y = np.meshgrid(*axes, indexing="ij")
    xy = np.column_stack((x.ravel(), y.ravel()))
    distances, nearest = cKDTree(lidar[:, :2]).query(xy, distance_upper_bound=radius)
    nearby = np.isfinite(distances) & (xy[:, 0] >= lidar[:, 0].min()) & (xy[:, 0] <= lidar[:, 0].max())
    return axes[0], axes[1], np.where(nearby, nearest, -1).reshape(x.shape)


def project_road_domain(
    points: np.ndarray, calibration: KittiCalibration, image_road: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    uv, _ = calibration.project_lidar_to_image(points)
    height, width = image_road.shape
    ids = np.flatnonzero(((uv >= 0) & (uv < [width, height])).all(axis=1))
    pixels = np.floor(uv[ids]).astype(np.intp)
    keep = image_road[pixels[:, 1], pixels[:, 0]]
    return ids[keep], pixels[keep]
