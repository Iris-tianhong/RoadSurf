"""Construct the Phase 5 ablation surfaces from shared road observations."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from core.hierarchical_surface import fit_ablation_models
from core.road_domain import DomainGrid
from core.road_lidar_extract import road_interior


@dataclass
class Reconstruction:
    grid: DomainGrid
    surfaces: dict
    fine_control_support: np.ndarray
    visual: np.ndarray
    visual_reliability: np.ndarray
    consistency_error: np.ndarray
    visual_weights: np.ndarray
    ray_jacobian: np.ndarray


def reconstruct(lidar, visual, calib, labels, sparse, coarse, fine, synthetic=False,
                boundary_margin=2, include_baselines=False):
    models, support, means, reliability, agreement, weights, jacobian = fit_ablation_models(
        lidar, visual, calib, coarse, fine, include_baselines=include_baselines)
    grid = DomainGrid.make(lidar, visual)
    road = road_interior(labels, boundary_margin)
    masks = {name: grid.visible_for(model, lidar, calib, road) for name, model in models.items()}
    hidden = grid.hidden_for(models["A3"], masks["A3"], calib, labels, road, sparse, synthetic)
    use = np.logical_or.reduce([hidden, *masks.values()])
    values = np.zeros(grid.shape)
    values[use] = support.evaluate(grid.xy[use.ravel()])
    empty = np.zeros(grid.shape, dtype=bool)
    surfaces = {name: grid.make_surface(model, masks[name], empty, values)
                for name, model in models.items()}
    surfaces["A4"] = grid.make_surface(models["A3"], masks["A3"], hidden, values)
    return Reconstruction(grid, surfaces, means, visual, reliability, agreement,
                          weights, jacobian)


def visual_reliability_grid(result, valid):
    reliability = np.full(valid.shape, np.nan)
    consistency = np.full(valid.shape, np.nan)
    if len(result.visual):
        distance, index = cKDTree(result.visual[:, :2]).query(result.grid.xy[valid.ravel()])
        use = np.flatnonzero(valid)
        nearby = distance <= .60
        reliability.ravel()[use[nearby]] = result.visual_reliability[index[nearby]]
        consistency.ravel()[use[nearby]] = result.consistency_error[index[nearby]]
    return reliability, consistency
