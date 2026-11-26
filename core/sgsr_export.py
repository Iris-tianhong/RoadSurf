"""Store Phase 5 models, masks, support, and separate visible/inferred point sets."""

import json

import numpy as np

from core.bspline_basis import TensorSplineBasis
from core.hierarchical_surface import RoadSurface, SplineModel, normal_from_gradient
from core.road_export import write_ply
from core.road_lidar_extract import ROAD_ROI_XY
from core.sgsr import visual_reliability_grid
from visualization.visualize_sgsr import save_sgsr_views


def export_reconstruction(output, result, image, calibration, provenance):
    surface = result.surfaces["A4"]
    grid, levels = result.grid, surface.model.levels
    valid = surface.visible | surface.hidden
    height, gradient = surface.values(*grid.xy.T)
    xyz = np.column_stack((grid.xy, height))
    support_visual, consistency = visual_reliability_grid(result, valid)
    height = height.reshape(grid.shape)
    gradient = gradient.reshape((*grid.shape, 2))
    output.mkdir(parents=True, exist_ok=False)
    for name, mask in (("visible", surface.visible), ("hidden", surface.hidden), ("complete", valid)):
        write_ply(output / f"road_{name}.ply", xyz[mask.ravel()])
    coarse, fine = levels
    np.savez_compressed(
        output / "road_surface.npz", plane=surface.model.plane, degree=coarse[0].degree,
        coarse_knots_x=coarse[0].knots_x, coarse_knots_y=coarse[0].knots_y, coarse_coefficients=coarse[1],
        fine_knots_x=fine[0].knots_x, fine_knots_y=fine[0].knots_y, fine_coefficients=fine[1],
        fine_control_support=result.fine_control_support, roi=ROAD_ROI_XY,
        coordinate_system="KITTI Velodyne, metres",
    )
    np.savez_compressed(
        output / "road_bev.npz", height_m=height, gradient_xy=gradient,
        normal_xyz=normal_from_gradient(gradient), slope_xy_rad=np.arctan(gradient),
        visible=surface.visible, hidden=surface.hidden, road_valid_mask=valid,
        support_lidar=surface.lidar_support, visual_reliability=support_visual,
        cross_modal_abs_error_m=consistency,
        grid_origin=grid.origin, grid_resolution=grid.resolution,
    )
    save_sgsr_views(output, image, calibration, surface, grid, height, surface.lidar_support)
    metadata = {
        **provenance, "version": "phase5-v1", "shape": list(grid.shape),
        "visible_points": int(surface.visible.sum()), "hidden_points": int(surface.hidden.sum()),
        "complete_points": int(valid.sum()),
        "observed_cells_by_method": {name: int(model.visible.sum()) for name, model in result.surfaces.items()},
        "fit_dtype": "float64", "visual_points": len(result.visual),
        "fine_coefficient_l2_by_method": {
            name: float(np.linalg.norm(result.surfaces[name].model.levels[1][1]))
            for name in ("A1", "A2", "A3")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_road_surface(output):
    """Load a Phase 5 export. Domain-aware queries return NaN outside valid cells."""
    with np.load(output / "road_surface.npz", allow_pickle=False) as data:
        levels = tuple(
            (TensorSplineBasis(data[f"{level}_knots_x"], data[f"{level}_knots_y"],
                               int(data["degree"])), data[f"{level}_coefficients"])
            for level in ("coarse", "fine"))
        model = SplineModel(data["plane"], levels)
    with np.load(output / "road_bev.npz", allow_pickle=False) as grid:
        return RoadSurface(model, grid["grid_origin"], float(grid["grid_resolution"]),
                           grid["visible"], grid["hidden"], grid["support_lidar"])
