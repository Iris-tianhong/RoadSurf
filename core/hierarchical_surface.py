"""Sequential cubic residual fits and domain-aware surface queries."""

from dataclasses import dataclass

import numpy as np

from core.sgsr_solver import fit_ransac_plane, fit_single_spline, plane_height
from core.support_and_weight import LidarSupport, coefficient_support, visual_weights


@dataclass
class SplineModel:
    plane: np.ndarray
    levels: tuple

    def evaluate(self, xy: np.ndarray):
        height = plane_height(xy, self.plane)
        gradient = np.broadcast_to(self.plane[:2], xy.shape).copy()
        for basis, coefficients in self.levels:
            residual, slope = basis.evaluate(xy, coefficients)
            height += residual
            gradient += slope
        return height, gradient


def fit_ablation_models(lidar, visual, calibration, coarse, fine, include_baselines=False):
    plane, _ = fit_ransac_plane(lidar)
    support = LidarSupport(lidar)
    means = coefficient_support(fine, support)
    damping = np.sqrt(.10 / means.size) * (1 - means)
    plain = fit_single_spline(lidar, coarse, plane, visual).coefficients
    model0 = SplineModel(plane, ((coarse, plain),))
    lidar_model = model0
    if not len(visual):
        weights = reliability = agreement = jz = np.empty(0)
        weighted_coarse = plain
    else:
        lidar_coef = fit_single_spline(lidar, coarse, plane).coefficients
        lidar_model = SplineModel(plane, ((coarse, lidar_coef),))
        weights, reliability, agreement, jz = visual_weights(
            visual, lidar, calibration, model0, lidar_model)
        weighted_coarse = fit_single_spline(
            lidar, coarse, plane, visual, visual_reliability=weights).coefficients
    models = {"A0": model0}
    if include_baselines:
        models.update(B0=SplineModel(plane, ()), B1=lidar_model)
    for name, coarse_coef, suppress, modifier in (
        ("A1", plain, None, None),
        ("A2", plain, damping, None),
        ("A3", weighted_coarse, damping, weights if len(weights) else None),
    ):
        parent = SplineModel(plane, ((coarse, coarse_coef),))
        fitted = fit_single_spline(
            lidar, fine, plane, visual, smoothness=.5, extra_damping=suppress,
            prior_height=lambda xy: parent.evaluate(xy)[0], visual_reliability=modifier,
        )
        models[name] = SplineModel(plane, (*parent.levels, (fine, fitted.coefficients)))
    return models, support, means, reliability, agreement, weights, jz


@dataclass
class RoadSurface:
    model: SplineModel
    origin: np.ndarray
    resolution: float
    visible: np.ndarray
    hidden: np.ndarray
    lidar_support: np.ndarray

    def _locate(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        xy = np.column_stack((x.ravel(), y.ravel()))
        position = (xy - self.origin) / self.resolution
        inside = ((position >= 0) & (position < self.visible.shape)).all(axis=1)
        ids = np.flatnonzero(inside)
        i, j = np.floor(position[ids]).astype(np.intp).T
        return x.shape, xy, ids, i, j

    def is_valid(self, x, y):
        shape, xy, ids, i, j = self._locate(x, y)
        valid = np.zeros(len(xy), dtype=bool)
        valid[ids] = self.visible[i, j] | self.hidden[i, j]
        return valid.reshape(shape)

    def is_hidden(self, x, y):
        shape, xy, ids, i, j = self._locate(x, y)
        hidden = np.zeros(len(xy), dtype=bool)
        hidden[ids] = self.hidden[i, j]
        return hidden.reshape(shape)

    def values(self, x, y):
        shape, xy, ids, i, j = self._locate(x, y)
        valid = ids[self.visible[i, j] | self.hidden[i, j]]
        height, gradients = np.full(len(xy), np.nan), np.full((len(xy), 2), np.nan)
        if len(valid):
            height[valid], gradients[valid] = self.model.evaluate(xy[valid])
        return height.reshape(shape), gradients.reshape((*shape, 2))

    def height(self, x, y):
        return self.values(x, y)[0]

    def gradient(self, x, y):
        return self.values(x, y)[1]

    def normal(self, x, y):
        gradient = self.gradient(x, y)
        return normal_from_gradient(gradient)

    def support(self, x, y):
        shape, xy, ids, i, j = self._locate(x, y)
        values = np.full(len(xy), np.nan)
        keep = self.visible[i, j] | self.hidden[i, j]
        values[ids[keep]] = self.lidar_support[i[keep], j[keep]]
        return values.reshape(shape)


def normal_from_gradient(gradient):
    normals = np.concatenate((-gradient, np.ones((*gradient.shape[:-1], 1))), axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)
