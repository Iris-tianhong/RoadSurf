"""Clamped cubic tensor-product B-splines in LiDAR XY coordinates."""

from typing import Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix, diags, eye, kron, vstack


def clamped_knots(bounds: Tuple[float, float], spacing: float, degree: int = 3) -> np.ndarray:
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Knot spacing must be positive.")
    lo, hi = bounds
    edges = np.linspace(lo, hi, int(np.ceil((hi - lo) / spacing)) + 1, dtype=np.float64)
    return np.r_[np.repeat(lo, degree), edges, np.repeat(hi, degree)]


def _second_difference(knots: np.ndarray, degree: int) -> csr_matrix:
    count = len(knots) - degree - 1
    # Greville abscissae account for the shorter end-control spacings.
    centers = sum(knots[i:i + count] for i in range(1, degree + 1)) / degree
    left, right = np.diff(centers)[:-1], np.diff(centers)[1:]
    return diags(
        [2 / (left * (left + right)), -2 / (left * right), 2 / (right * (left + right))],
        [0, 1, 2], shape=(count - 2, count), format="csr")


class TensorSplineBasis:
    def __init__(self, knots_x: np.ndarray, knots_y: np.ndarray, degree: int = 3):
        self.knots_x, self.knots_y, self.degree = knots_x, knots_y, degree
        nx, ny = self.shape
        self.bx = BSpline(knots_x, np.eye(nx), degree, extrapolate=False)
        self.by = BSpline(knots_y, np.eye(ny), degree, extrapolate=False)
        self.dx = self.bx.derivative()
        self.dy = self.by.derivative()

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.knots_x) - self.degree - 1, len(self.knots_y) - self.degree - 1

    def design(self, xy: np.ndarray) -> csr_matrix:
        bx = BSpline.design_matrix(xy[:, 0], self.knots_x, self.degree)
        by = BSpline.design_matrix(xy[:, 1], self.knots_y, self.degree)
        # Each univariate row has degree+1 stored entries, including endpoint zeros.
        count = self.degree + 1
        nx, ny = self.shape
        columns = bx.indices.reshape(-1, count, 1) * ny + by.indices.reshape(-1, 1, count)
        values = bx.data.reshape(-1, count, 1) * by.data.reshape(-1, 1, count)
        row_offsets = np.arange(len(xy) + 1, dtype=np.intp) * count**2
        return csr_matrix((values.ravel(), columns.ravel(), row_offsets), shape=(len(xy), nx * ny))

    def curvature(self) -> csr_matrix:
        nx, ny = self.shape
        dx = _second_difference(self.knots_x, self.degree)
        dy = _second_difference(self.knots_y, self.degree)
        return vstack((kron(dx, eye(ny)), kron(eye(nx), dy)), format="csr")

    def evaluate(self, xy: np.ndarray, coefficients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the residual surface and its analytic XY derivatives."""
        bx, by = self.bx(xy[:, 0]), self.by(xy[:, 1])
        height = np.sum((bx @ coefficients) * by, axis=1)
        sx = np.sum((self.dx(xy[:, 0]) @ coefficients) * by, axis=1)
        sy = np.sum((bx @ coefficients) * self.dy(xy[:, 1]), axis=1)
        return height, np.column_stack((sx, sy))
