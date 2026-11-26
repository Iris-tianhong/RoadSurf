"""Evaluate reconstructed surfaces against held-out KITTI road references."""

from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from core.road_lidar_extract import road_roi_mask
from core.sgsr_solver import plane_height, project_road_domain


DISTANCE_BINS = (("all", 5, 70), ("5-20", 5, 20), ("20-40", 20, 40), ("40-70", 40, 70))


def in_evaluation_range(xyz):
    return road_roi_mask(xyz) & (xyz[:, 0] >= 5) & (xyz[:, 0] < 70)


def heldout_scan_points(full_road, current_scan):
    """Return reference road points whose raw records are absent from the input scan."""
    full = np.ascontiguousarray(full_road).view("V16").ravel()
    used = np.ascontiguousarray(current_scan).view("V16").ravel()
    held = full_road[~np.isin(full, used), :3].astype(np.float64)
    return held[in_evaluation_range(held)]


def dc_reference_points(depth, full_sparse, road, calibration, exclusion_pixels=1):
    if depth.shape != full_sparse.shape or depth.shape != road.shape:
        raise ValueError("KITTI DC reference and current input must share the original RGB grid.")
    early_overlap = np.isfinite(full_sparse) & np.isfinite(depth)
    shared = int(early_overlap.sum())
    alignment_mae = (float(np.median(np.abs(full_sparse[early_overlap] - depth[early_overlap])))
                     if shared else float("inf"))
    if shared < 100 or alignment_mae > .30:
        raise ValueError(f"Raw/DC alignment check failed ({shared} pixels; median {alignment_mae:.3f} m).")
    v, u = np.nonzero(road & np.isfinite(depth) & (depth >= 1) & (depth <= 80))
    points = calibration.backproject_image_to_lidar(np.column_stack((u, v)), depth[v, u])
    extent = in_evaluation_range(points)
    # The same conservative input-exclusion mask is used for all three beam counts.
    window = 2 * exclusion_pixels + 1
    near_input = binary_dilation(np.isfinite(full_sparse), structure=np.ones((window, window), dtype=bool))
    return points[extent], points[extent & ~near_input[v, u]], {
        "dc_full_scan_overlap_px": shared, "dc_full_scan_median_depth_delta_m": alignment_mae,
    }


def reference_slopes(points):
    """Estimate local ground normals on spatially spaced, independently supported sites."""
    slopes = np.full((len(points), 2), np.nan, dtype=np.float64)
    if len(points) < 12:
        return slopes
    xy = points[:, :2]
    cells = np.floor(xy)
    _, group = np.unique(cells, axis=0, return_inverse=True)
    distances = np.sum((xy - (cells + .5))**2, axis=1)
    order = np.lexsort((distances, group))
    _, first = np.unique(group[order], return_index=True)
    anchors = order[first]
    neighbors = cKDTree(xy).query_ball_point(xy[anchors], 1.5)
    for anchor, ids in zip(anchors, neighbors):
        if len(ids) < 12:
            continue
        pos = xy[ids] - xy[anchor]
        design = np.column_stack((pos, np.ones(len(ids))))
        z = points[ids, 2]
        result = np.linalg.lstsq(design, z, rcond=None)[0]
        error = np.abs(design @ result - z)
        inliers = error <= .08
        if inliers.sum() < max(12, .8 * len(ids)):
            continue
        support = pos[inliers] - pos[inliers].mean(axis=0)
        if np.linalg.eigvalsh(support.T @ support / len(support))[0] < .15**2:
            continue
        result = np.linalg.lstsq(design[inliers], z[inliers], rcond=None)[0]
        if np.median(np.abs(design[inliers] @ result - z[inliers])) <= .04:
            slopes[anchor] = result[:2]
    return slopes


def _surface_values(xy, fit, plane, basis):
    residual, gradient = basis.evaluate(xy, fit.coefficients)
    return plane_height(xy, plane) + residual, gradient + plane[:2]


def evaluate_surface_at_reference(points, lidar, calibration, road, plane, basis, fits):
    """Apply the Phase 3 valid-domain rules at exact reference XY coordinates."""
    xy = points[:, :2]
    distances, nearest = cKDTree(lidar[:, :2]).query(xy, distance_upper_bound=1.25)
    candidates = np.flatnonzero(np.isfinite(distances) & (xy[:, 0] >= lidar[:, 0].min())
                                & (xy[:, 0] <= lidar[:, 0].max()))
    result = {}
    for name, fit in fits.items():
        height = np.full(len(xy), np.nan)
        slopes = np.full((len(xy), 2), np.nan)
        if fit is not None and len(candidates):
            z, grad = _surface_values(xy[candidates], fit, plane, basis)
            idx, _ = project_road_domain(np.c_[xy[candidates], z], calibration, road)
            anchors = lidar[nearest[candidates[idx]]]
            anchor_z, _ = _surface_values(anchors[:, :2], fit, plane, basis)
            idx = idx[np.abs(anchor_z - anchors[:, 2]) <= .30]
            height[candidates[idx]] = z[idx]
            slopes[candidates[idx]] = grad[idx]
        result[name] = (height, slopes)
    return result


def _normals(slopes):
    normals = np.column_stack((-slopes, np.ones(len(slopes))))
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def _metrics(height, gradients, mask, ref_points, ref_slopes):
    error = height[mask] - ref_points[mask, 2]
    available = mask & np.isfinite(ref_slopes).all(axis=1)
    cosine = np.sum(_normals(gradients[available]) * _normals(ref_slopes[available]), axis=1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return {
        "covered": int(mask.sum()), "abs_height_sum": float(np.abs(error).sum()),
        "squared_height_sum": float(np.dot(error, error)), "height_sum": float(error.sum()),
        "normal_covered": int(available.sum()), "normal_angle_sum": float(angle.sum()),
    }


def metric_rows(reference, ref_slopes, predictions, common_mask=None):
    common = np.logical_and.reduce([np.isfinite(h) for h, _ in predictions.values()])
    if common_mask is not None:
        common &= common_mask
    for band, lo, hi in DISTANCE_BINS:
        eligible = (reference[:, 0] >= lo) & (reference[:, 0] < hi)
        for name, (height, gradients) in predictions.items():
            own = eligible & np.isfinite(height)
            row = {
                "method": name, "distance_m": band, "references": int(eligible.sum()),
                "normal_references": int((eligible & np.isfinite(ref_slopes).all(axis=1)).sum()),
            }
            row.update(_metrics(height, gradients, own, reference, ref_slopes))
            row.update({"common_" + key: value for key, value in
                        _metrics(height, gradients, eligible & common, reference, ref_slopes).items()})
            yield finalize_rates(row)


def finalize_rates(row):
    count, ref = row["covered"], row["references"]
    n_count, n_ref = row["normal_covered"], row["normal_references"]
    row["coverage_pct"] = 100 * count / ref if ref else None
    row["normal_coverage_pct"] = 100 * n_count / n_ref if n_ref else None
    for prefix in ("", "common_"):
        count, n_count = row[prefix + "covered"], row[prefix + "normal_covered"]
        row[prefix + "height_mae_m"] = row[prefix + "abs_height_sum"] / count if count else None
        row[prefix + "height_rmse_m"] = (row[prefix + "squared_height_sum"] / count)**.5 if count else None
        row[prefix + "height_bias_m"] = row[prefix + "height_sum"] / count if count else None
        row[prefix + "normal_mae_deg"] = row[prefix + "normal_angle_sum"] / n_count if n_count else None
    return row


def aggregate_rows(rows, by_drive=False):
    keys = ("beams", "reference", "distance_m", "method")
    if by_drive:
        keys = ("drive",) + keys
    sums = defaultdict(lambda: defaultdict(float))
    for row in rows:
        group = tuple(row[k] for k in keys)
        for field in ("references", "normal_references"):
            sums[group][field] += row[field]
        for prefix in ("", "common_"):
            for field in ("covered", "abs_height_sum", "squared_height_sum", "height_sum",
                          "normal_covered", "normal_angle_sum"):
                field = prefix + field
                sums[group][field] += row[field]
    return [finalize_rates({**dict(zip(keys, group)), **values}) for group, values in sorted(sums.items())]
