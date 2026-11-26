"""Robust global inverse-depth calibration and visual road backprojection."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from core.road_lidar_extract import road_roi_mask
from data.kitti_calib import KittiCalibration


MIN_DEPTH_M = 1.0
MAX_DEPTH_M = 80.0


class DepthAlignmentError(ValueError):
    """The available anchors do not support a physically valid calibration."""


@dataclass
class InverseDepthFit:
    scale: float
    shift: float
    anchors: int
    occupied_distance_bins: int
    distance_bin_m: float
    condition_number: float
    anchor_depth_range_m: Tuple[float, float]
    anchor_inverse_depth_mae_per_m: float
    anchor_depth_mae_m: float


def fit_inverse_depth(
    relative: np.ndarray, depth: np.ndarray, distance_bin_m: float = 5.0,
    min_anchors: int = 30,
) -> InverseDepthFit:
    """Fit 1/Z = scale * q + shift with distance-bin balancing and Huber IRLS."""
    if not np.isfinite(distance_bin_m) or distance_bin_m <= 0 or min_anchors < 2:
        raise ValueError("Expected a positive distance bin width and at least two anchors.")
    usable = (np.isfinite(relative) & np.isfinite(depth)
              & (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M))
    q, z = relative[usable].astype(np.float64), depth[usable].astype(np.float64)
    if len(q) < min_anchors:
        raise DepthAlignmentError(f"Only {len(q)} usable LiDAR anchors; need {min_anchors}.")

    _, bins, counts = np.unique(np.floor(z / distance_bin_m), return_inverse=True, return_counts=True)
    if len(counts) < 2:
        raise DepthAlignmentError("LiDAR anchors cover fewer than two distance bins.")
    base_weight = 1.0 / counts[bins]
    q_scale = np.max(np.abs(q))
    if q_scale == 0:
        raise DepthAlignmentError("The predicted relative depth is zero at every LiDAR anchor.")
    design = np.column_stack((q / q_scale, np.ones(len(q))))
    target = 1 / z
    weight = base_weight
    for _ in range(10):
        root_weight = np.sqrt(weight)
        try:
            coefficients, _, rank, singular = np.linalg.lstsq(
                design * root_weight[:, None], target * root_weight, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise DepthAlignmentError("Inverse-depth least-squares solver did not converge.") from exc
        if rank != 2 or singular[0] > 1e4 * singular[1]:
            raise DepthAlignmentError("Inverse-depth fit is ill-conditioned.")
        residual = design @ coefficients - target
        mad = np.median(np.abs(residual - np.median(residual)))
        huber_delta = max(1e-5, 1.345 * 1.4826 * mad)
        weight = base_weight * huber_delta / np.maximum(huber_delta, np.abs(residual))

    scale, shift = float(coefficients[0] / q_scale), float(coefficients[1])
    if not np.isfinite([scale, shift]).all() or scale <= 0:
        raise DepthAlignmentError("Fit slope must be positive for the DA-V2 relative prediction.")
    inverse = scale * q + shift
    valid = (inverse >= 1 / MAX_DEPTH_M) & (inverse <= 1 / MIN_DEPTH_M)
    if not valid.any():
        raise DepthAlignmentError("Calibrated inverse depth is outside the supported range.")
    return InverseDepthFit(
        scale, shift, len(q), len(counts), distance_bin_m, float(singular[0] / singular[1]),
        (float(z.min()), float(z.max())), float(np.abs(residual).mean()),
        float(np.abs(1 / inverse[valid] - z[valid]).mean()),
    )


def calibrate_from_road_points(
    prediction: np.ndarray, points: np.ndarray, calibration: KittiCalibration
) -> InverseDepthFit:
    """Bilinear q samples at the unrounded UVs of the measured road points."""
    uv, depth = calibration.project_lidar_to_image(points)
    height, width = prediction.shape
    uv_valid = ((uv >= 0) & (uv <= [width - 1, height - 1])).all(axis=1)
    uv = uv[uv_valid]
    u0, v0 = np.floor(uv).astype(np.intp).T
    u1, v1 = np.minimum(u0 + 1, width - 1), np.minimum(v0 + 1, height - 1)
    du, dv = uv[:, 0] - u0, uv[:, 1] - v0
    left = (1 - dv) * prediction[v0, u0] + dv * prediction[v1, u0]
    right = (1 - dv) * prediction[v0, u1] + dv * prediction[v1, u1]
    return fit_inverse_depth(left * (1 - du) + right * du, depth[uv_valid])


def backproject_visual_road(
    prediction: np.ndarray, road: np.ndarray, fit: InverseDepthFit, calibration: KittiCalibration
) -> Tuple[np.ndarray, np.ndarray]:
    """Return visual XYZ in LiDAR metres and aligned optical depths (NaN elsewhere)."""
    inverse = fit.scale * prediction.astype(np.float64) + fit.shift
    valid = np.isfinite(inverse) & (inverse >= 1 / MAX_DEPTH_M) & (inverse <= 1 / MIN_DEPTH_M)
    depth = np.full(inverse.shape, np.nan, dtype=np.float64)
    np.divide(1, inverse, out=depth, where=valid)
    return backproject_road_depth(depth, road, calibration)


def backproject_road_depth(
    metric_depth: np.ndarray, road: np.ndarray, calibration: KittiCalibration,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the shared road/depth/ROI limits to a metric camera-Z prediction."""
    valid = (road & np.isfinite(metric_depth)
             & (metric_depth >= MIN_DEPTH_M) & (metric_depth <= MAX_DEPTH_M))
    v, u = np.nonzero(valid)
    depth = metric_depth[v, u]
    points = calibration.backproject_image_to_lidar(np.column_stack((u, v)), depth)
    keep = np.isfinite(points).all(axis=1) & road_roi_mask(points)
    calibrated = np.full(metric_depth.shape, np.nan, dtype=np.float32)
    calibrated[v[keep], u[keep]] = depth[keep]
    return points[keep].astype(np.float32), calibrated
