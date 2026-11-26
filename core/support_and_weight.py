"""LiDAR observation support and camera-ray uncertainty proxies."""

import numpy as np
from scipy.spatial import cKDTree


class LidarSupport:
    def __init__(self, lidar: np.ndarray):
        xy = np.asarray(lidar[:, :2], dtype=np.float64)
        _, index = np.unique(np.floor(xy / .20), axis=0, return_index=True)
        self.tree = cKDTree(xy[index])

    def evaluate(self, xy: np.ndarray) -> np.ndarray:
        """Neighbour distance, occupied-cell density, and 2D directional coverage."""
        query = np.asarray(xy, dtype=np.float64)
        if not self.tree.n:
            return np.zeros(len(query))
        radius = 2.0
        d, idx = self.tree.query(query, k=64, distance_upper_bound=radius)
        local = (self.tree.data[np.minimum(idx, self.tree.n - 1)] - query[:, None, :]) / radius
        weight = np.maximum(1 - d / radius, 0)
        weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
        center = np.sum(local * weight[..., None], axis=1)
        offsets = local - center[:, None, :]
        covariance = np.einsum("nk,nki,nkj->nij", weight, offsets, offsets)
        eig = np.maximum(np.linalg.eigvalsh(covariance), 0)
        direction = (2 * eig[:, 0] / np.maximum(eig.sum(axis=1), 1e-12)
                     * np.minimum(1, eig[:, 0] / .04))
        count = self.tree.query_ball_point(query, radius, return_length=True)
        density = count / (np.pi * radius**2)
        return np.clip(np.exp(-d[:, 0] / 1.5) * density / (density + 2) * direction, 0, 1)


def coefficient_support(basis, support: LidarSupport) -> np.ndarray:
    """Basis-weighted average of the support field over each fine basis footprint."""
    axes = [np.arange(knots[0] + .25, knots[-1], .50)
            for knots in (basis.knots_x, basis.knots_y)]
    x, y = np.meshgrid(*axes, indexing="ij")
    xy = np.column_stack((x.ravel(), y.ravel()))
    design = basis.design(xy)
    mass = np.asarray(design.sum(axis=0)).ravel()
    weighted = design.T @ support.evaluate(xy)
    means = np.divide(weighted, mass, out=np.zeros_like(weighted), where=mass > 0)
    return np.clip(means.reshape(basis.shape), 0, 1)


def camera_ray_weights(rays, gradient, depth, reliability):
    """Weights with an optical-axis prior, lateral error, variance floor, and cap."""
    jacobian = rays[:, 2] - np.sum(gradient * rays[:, :2], axis=1)
    sigma_depth = .08 + .025 * depth
    variance = (jacobian * sigma_depth)**2 + .10**2 * np.sum(gradient**2, axis=1) + .035**2
    return np.minimum(250, reliability / variance), jacobian


def visual_weights(visual, lidar, calibration, provisional, lidar_model):
    """Use LiDAR consistency where measured, and a fixed reduced prior elsewhere."""
    uv, depth = calibration.project_lidar_to_image(visual)
    _, rays = calibration.pixel_to_lidar_ray(uv)
    _, gradient = provisional.evaluate(visual[:, :2])
    lidar_height, _ = lidar_model.evaluate(visual[:, :2])
    distance, _ = cKDTree(lidar[:, :2]).query(visual[:, :2])
    agreement = np.full(len(visual), np.nan)
    nearby = distance <= 1.25
    agreement[nearby] = np.abs(visual[nearby, 2] - lidar_height[nearby])
    reliability = np.full(len(visual), .35)
    reliability[nearby] = np.maximum(.05, np.exp(-agreement[nearby] / .15))
    weights, jz = camera_ray_weights(rays, gradient, depth, reliability)
    return weights, reliability, agreement, jz
