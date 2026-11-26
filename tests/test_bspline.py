import numpy as np
import pytest

from core.bspline_basis import TensorSplineBasis, clamped_knots


def test_clamped_tensor_basis_at_both_endpoints_and_linear_derivatives():
    kx, ky = clamped_knots((0, 9), 2.5), clamped_knots((-4, 4), 3)
    basis = TensorSplineBasis(kx, ky)
    xy = np.array([[0, -4], [9, 4], [4.2, -1.75], [8, 2.1]])
    design = basis.design(xy)
    assert design.dtype == np.float64
    np.testing.assert_allclose(np.asarray(design.sum(axis=1)).ravel(), 1, atol=1e-13)
    assert design[0, 0] == design[1, design.shape[1] - 1] == 1
    with pytest.raises(ValueError):
        basis.design(np.array([[9.01, 0]]))
    gx, gy = (np.convolve(knots[1:-1], np.ones(3) / 3, mode="valid") for knots in (kx, ky))
    coefficients = 2 + .1 * gx[:, None] - .2 * gy[None, :]
    height, gradient = basis.evaluate(xy, coefficients)
    np.testing.assert_allclose(height, 2 + .1 * xy[:, 0] - .2 * xy[:, 1], atol=1e-13)
    np.testing.assert_allclose(gradient, [[.1, -.2]] * len(xy), atol=1e-13)
    np.testing.assert_allclose(basis.curvature() @ coefficients.ravel(), 0, atol=1e-13)
