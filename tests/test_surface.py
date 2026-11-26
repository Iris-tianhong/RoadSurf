import numpy as np
import pytest

from core.road_export import read_road_ply, write_ply
from core.sgsr_solver import (
    fit_ransac_plane,
    fit_single_spline,
    make_spline_basis,
    observed_road_grid,
    project_road_domain,
)
from data.kitti_calib import KittiCalibration


def test_ransac_recovers_a_sloping_plane_with_outliers():
    xy = np.random.default_rng(5).uniform([2, -4], [24, 4], (100, 2))
    z = .09 * xy[:, 0] - .06 * xy[:, 1] - 1.7
    z[:20] += 1.5
    plane, inliers = fit_ransac_plane(np.c_[xy, z])
    np.testing.assert_allclose(plane, [.09, -.06, -1.7], atol=1e-12)
    assert inliers == 80


def test_visual_term_is_independent_of_redundant_visual_samples():
    x, y = np.meshgrid(np.linspace(5, 24, 45), np.linspace(-6, 6, 20))
    xy = np.column_stack((x.ravel(), y.ravel()))
    plane = np.array([0, 0, -1.5])
    z = plane[2] + .25 * np.sin(xy[:, 0] / 5) + .008 * xy[:, 1] ** 2
    lidar = np.c_[xy, z]
    visual = np.c_[xy, z + .15]
    basis = make_spline_basis(4)
    lidar_fit = fit_single_spline(lidar, basis, plane)
    fused = fit_single_spline(lidar, basis, plane, visual)
    repeat = fit_single_spline(lidar, basis, plane, np.repeat(visual, 35, axis=0))
    assert lidar_fit.coefficients.dtype == np.float64
    assert fused.visual_fit_samples == repeat.visual_fit_samples
    np.testing.assert_allclose(fused.coefficients, repeat.coefficients, atol=1e-12)
    lidar_z = basis.evaluate(xy, lidar_fit.coefficients)[0] + plane[2]
    fused_z = basis.evaluate(xy, fused.coefficients)[0] + plane[2]
    assert np.abs(lidar_z - z).mean() < .03
    assert .005 < np.mean(fused_z - lidar_z) < .075


def test_domain_uses_observations_and_actual_road_pixels(tmp_path):
    lidar = np.array([[2, 0, 1], [4, 0, 1]], dtype=float)
    x, y, nearest = observed_road_grid(lidar, .2, .55)
    domain = nearest >= 0
    assert domain.any()
    xx, yy = x[np.nonzero(domain)[0]], y[np.nonzero(domain)[1]]
    assert ((xx >= 2) & (xx <= 4) & (abs(yy) < .55)).all()
    assert (np.minimum(abs(xx - 2), abs(xx - 4)) < .55).all()
    expected = np.where(abs(xx - 2) < abs(xx - 4), 0, 1)
    np.testing.assert_array_equal(nearest[domain], expected)
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    mask = np.zeros((5, 6), dtype=bool)
    mask[0, 2] = True
    indices, pixels = project_road_domain(lidar, KittiCalibration(path), mask)
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_array_equal(pixels, [[2, 0]])


def test_ply_payload_count_is_checked(tmp_path):
    path = tmp_path / "points.ply"
    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    write_ply(path, points)
    np.testing.assert_array_equal(read_road_ply(path), points)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="vertex count"):
        read_road_ply(path)
