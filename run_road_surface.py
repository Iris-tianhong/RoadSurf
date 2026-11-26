"""Reconstruct the Phase 3 baselines from one completed Phase 2 directory."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import scipy

from core.road_export import read_road_ply, write_ply
from core.road_lidar_extract import ROAD_ROI_XY, road_interior, road_roi_mask
from core.sgsr_solver import (
    fit_ransac_plane,
    fit_single_spline,
    make_spline_basis,
    observed_road_grid,
    plane_height,
    project_road_domain,
)
from data.kitti_calib import KittiCalibration
from visualization.visualize_road import save_height_comparison, save_surface_projection


METHODS = ("ransac", "lidar_bspline", "multimodal_bspline")


def _max_neighbor_step(height: np.ndarray) -> float:
    deltas = np.concatenate([np.abs(np.diff(height, axis=axis)).ravel() for axis in (0, 1)])
    return float(np.max(deltas[np.isfinite(deltas)], initial=0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", required=True, type=Path)
    parser.add_argument("--kitti-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--knot-spacing", type=float, default=4.0)
    parser.add_argument("--grid-spacing", type=float, default=0.2)
    parser.add_argument("--observed-radius", type=float, default=1.25)
    parser.add_argument("--max-anchor-residual", type=float, default=0.30)
    parser.add_argument("--visual-weight", type=float, default=0.25)
    parser.add_argument("--smoothness", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.max_anchor_residual) or args.max_anchor_residual <= 0:
        parser.error("--max-anchor-residual must be positive and finite.")

    inputs = args.phase2_dir
    source = json.loads((inputs / "metadata.json").read_text(encoding="utf-8"))
    frame, split = source["frame"], source["split"]
    output = args.out or Path("outputs") / "phase3" / "kitti_object" / split / "64beams" / frame
    if output.exists():
        parser.error(f"Use a new output directory: {output}")
    lidar = read_road_ply(inputs / "lidar_road.ply")[:, :3].astype(np.float64)
    visual = read_road_ply(inputs / "visual_road.ply")[:, :3].astype(np.float64)
    if source.get("mode") != "lidar_and_visual" or not len(visual):
        raise ValueError("Phase 3 needs a successful Phase 2 run with visual road points.")
    for points in (lidar, visual):
        if not np.isfinite(points).all() or not road_roi_mask(points).all():
            raise ValueError("Phase 2 point clouds must contain finite XYZ inside the LiDAR ROI.")

    base = args.kitti_root / split
    calibration = KittiCalibration(base / "calib" / f"{frame}.txt")
    with Image.open(base / "image_2" / f"{frame}.png") as image:
        rgb = image.convert("RGB")
    labels = np.load(inputs / "semantic_labels.npy", allow_pickle=False)
    if labels.shape != rgb.size[::-1]:
        raise ValueError("Phase 2 labels must match the original RGB dimensions.")
    image_road = road_interior(labels, source["counts"]["semantic_boundary_margin_px"])

    plane, inliers = fit_ransac_plane(lidar)
    basis = make_spline_basis(args.knot_spacing)
    splines = [
        fit_single_spline(lidar, basis, plane, smoothness=args.smoothness),
        fit_single_spline(lidar, basis, plane, visual, args.visual_weight, args.smoothness),
    ]
    axis_x, axis_y, nearest = observed_road_grid(lidar, args.grid_spacing, args.observed_radius)
    ix, iy = np.nonzero(nearest >= 0)
    anchor_ids = nearest[ix, iy]
    xy = np.column_stack((axis_x[ix], axis_y[iy]))
    base_height = plane_height(xy, plane)
    base_residual = plane_height(lidar[:, :2], plane) - lidar[:, 2]
    grids = np.full((len(METHODS), *nearest.shape), np.nan, dtype=np.float64)
    gradients = np.full((*grids.shape, 2), np.nan, dtype=np.float64)
    point_clouds, stats = {}, {}
    for index, (method, fit) in enumerate(zip(METHODS, [None, *splines])):
        heights, slope_xy, residual = base_height, np.broadcast_to(plane[:2], xy.shape), base_residual
        if fit is not None:
            delta, grad = basis.evaluate(xy, fit.coefficients)
            heights, slope_xy = heights + delta, slope_xy + grad
            residual = residual + basis.design(lidar[:, :2]) @ fit.coefficients.ravel()
        points = np.column_stack((xy, heights)).astype(np.float32)
        valid, pixels = project_road_domain(points, calibration, image_road)
        # Check fit agreement at the nearest real return's own XY, so slope
        # between the return and the grid cell is not mistaken for a height jump.
        agrees = np.abs(residual[anchor_ids[valid]]) <= args.max_anchor_residual
        rejected = int(np.count_nonzero(~agrees))
        valid, pixels = valid[agrees], pixels[agrees]
        if not len(valid):
            raise ValueError(f"No valid road cells for {method}.")
        grids[index, ix[valid], iy[valid]] = heights[valid]
        gradients[index, ix[valid], iy[valid]] = slope_xy[valid]
        point_clouds[method] = points[valid]
        stats[method] = {
            "points": len(valid),
            "lidar_fit_mae_m": float(np.mean(np.abs(residual))),
            "local_height_rejected": rejected,
            "max_neighbor_height_step_m": _max_neighbor_step(grids[index]),
            "z_range_m": [float(heights[valid].min()), float(heights[valid].max())],
        }
        if fit is not None:
            stats[method].update(visual_fit_samples=fit.visual_fit_samples,
                                 lsmr_last_iterations=fit.solver_iterations)

    output.mkdir(parents=True, exist_ok=False)
    for method, points in point_clouds.items():
        write_ply(output / f"road_{method}.ply", points)
    np.savez_compressed(output / "road_surface.npz", methods=METHODS[1:], base_plane=plane,
                        knots_x=basis.knots_x, knots_y=basis.knots_y, degree=basis.degree,
                        coefficients=np.stack([fit.coefficients for fit in splines]),
                        roi_lidar_xy_m=ROAD_ROI_XY)
    np.savez_compressed(output / "road_bev.npz", methods=METHODS, x_m=axis_x, y_m=axis_y,
                        height_m=grids, gradient_xy=gradients, valid=np.isfinite(grids))
    save_height_comparison(output, grids, ("RANSAC plane", "LiDAR B-spline", "Multimodal B-spline"))
    save_surface_projection(output, rgb, pixels)
    report = json.dumps({
        "frame": frame, "split": split, "phase2_input": str(inputs.resolve()),
        "coordinate_system": "KITTI Velodyne (m)", "scipy_version": scipy.__version__,
        "solve_dtype": "float64", "grid_spacing_m": args.grid_spacing,
        "knot_spacing_target_m": args.knot_spacing, "observed_radius_m": args.observed_radius,
        "max_anchor_residual_m": args.max_anchor_residual,
        "visual_weight": args.visual_weight, "smoothness": args.smoothness,
        "lidar_observations": len(lidar), "visual_observations": len(visual),
        "ransac_inliers": inliers, "methods": stats,
    }, indent=2)
    (output / "metadata.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
