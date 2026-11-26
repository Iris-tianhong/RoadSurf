import numpy as np

from core.bspline_basis import TensorSplineBasis, clamped_knots
from core.hierarchical_surface import RoadSurface, SplineModel
from core.road_domain import DomainGrid, bounded_missing
from core.sgsr_solver import fit_single_spline, make_spline_basis
from core.support_and_weight import LidarSupport, camera_ray_weights, coefficient_support
from data.kitti_calib import KittiCalibration


def test_lidar_support_distinguishes_line_from_2d_coverage():
    x, y = np.meshgrid(np.arange(-1.8, 2, .3), np.arange(-1.8, 2, .3))
    area = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    line = np.c_[np.linspace(-1.9, 1.9, 90), np.zeros((90, 2))]
    probes = np.array([[0., 0.], [10., 0.]])
    support = LidarSupport(area).evaluate(probes)
    assert .2 < support[0] <= 1
    assert support[1] == 0
    assert LidarSupport(line).evaluate(probes)[0] == 0
    assert (coefficient_support(make_spline_basis(2), LidarSupport(area)) == 0).any()


def test_camera_ray_jacobian_matches_finite_difference_and_weights_are_bounded():
    rays = np.array([[.1, .2, .4], [1., 0., .2]])
    grad = np.array([[.3, .2], [.2, 0.]])
    w, jz = camera_ray_weights(rays, grad, np.array([20., 20.]), np.array([1., 1.]))
    residual = lambda depth: depth * rays[:, 2] - np.sum(grad * (depth * rays[:, :2]), axis=1)
    np.testing.assert_allclose(jz, (residual(20.01) - residual(19.99)) / .02, atol=1e-12)
    assert jz[1] == 0
    assert 0 < w[0] < w[1] <= 250


def test_low_support_damping_reduces_fine_coefficients():
    x, y = np.meshgrid(np.linspace(1, 15, 43), np.linspace(-4, 4, 20))
    xy = np.c_[x.ravel(), y.ravel()]
    points = np.c_[xy, .3 * np.sin(xy[:, 0])]
    base = np.zeros(3)
    coarse, fine = make_spline_basis(4), make_spline_basis(2)
    c = fit_single_spline(points, coarse, base).coefficients
    parent = SplineModel(base, ((coarse, c),))
    support = coefficient_support(fine, LidarSupport(points))
    damping = np.sqrt(.10 / support.size) * (1 - support)
    options = dict(prior_height=lambda query: parent.evaluate(query)[0], smoothness=.5)
    unsuppressed = fit_single_spline(points, fine, base, **options).coefficients
    suppressed = fit_single_spline(points, fine, base, extra_damping=damping, **options).coefficients
    low = support < .25
    assert np.linalg.norm(suppressed[low]) < np.linalg.norm(unsuppressed[low])


def test_bounded_holes_do_not_fill_edges_or_large_components():
    visible = np.zeros((85, 68), bool)
    visible[2:-2, 2:-2] = True
    visible[5:25, 9:19] = False
    visible[31:70, 20:42] = False
    visible[45:52, :9] = False
    filled = bounded_missing(visible, .2)
    assert filled[5:25, 9:19].all()
    assert not filled[31:70, 20:42].any()
    assert not filled[45:52, :9].any()
    assert not (filled & visible).any()


def test_real_hidden_requires_foreground_depth_and_keeps_queries_gated(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 5 0 0 0 0 5 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    calib = KittiCalibration(path)
    visible = np.ones((25, 25), dtype=bool)
    visible[6:10, 8:12] = False
    x, y = np.meshgrid(np.arange(25) * .2 + .1, np.arange(25) * .2 + .1, indexing="ij")
    xy = np.c_[x.ravel(), y.ravel()]
    grid = DomainGrid(xy, visible.shape, np.zeros(len(xy), dtype=int),
                      visible.ravel(), np.zeros(2), .2)
    labels = np.zeros(visible.shape, dtype=np.uint8)
    labels[8:12, 6:10] = 13
    labels[8, 6] = 1
    sparse = np.full(visible.shape, np.nan)
    model = SplineModel(np.array([0., 0., 1.]), ())
    road = labels == 0
    assert not grid.hidden_for(model, visible, calib, labels, road, sparse).any()
    sparse[9, 9] = .4
    hidden = grid.hidden_for(model, visible, calib, labels, road, sparse)
    assert hidden.sum() == 15 and not hidden[6, 8]
    surface = RoadSurface(model, np.zeros(2), .2, visible, hidden, np.full_like(x, .4))
    assert surface.is_hidden(1.5, 1.9) and surface.height(1.5, 1.9) == 1
    assert not surface.is_valid(1.3, 1.7) and np.isnan(surface.height(1.3, 1.7))
    assert np.isnan(surface.support(-1, 0))
