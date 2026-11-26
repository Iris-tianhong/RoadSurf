import numpy as np
import pytest

from core.depth_alignment import (
    DepthAlignmentError,
    backproject_visual_road,
    calibrate_from_road_points,
    fit_inverse_depth,
)
from data.kitti_calib import KittiCalibration


def test_huber_recovers_positive_affine_fit_with_outliers_and_uneven_sampling():
    depth = np.r_[np.linspace(4, 10, 900), np.linspace(10, 65, 100)]
    q = (1 / depth - .01) / .025
    q[::17] += 2.0
    fit = fit_inverse_depth(q, depth)
    assert fit.anchors == 1000
    assert fit.occupied_distance_bins > 2
    assert fit.scale == pytest.approx(.025, rel=.005)
    assert fit.shift == pytest.approx(.01, abs=.0001)


def test_calibration_rejects_unidentifiable_or_reversed_depth():
    depth = np.linspace(5, 60, 40)
    with pytest.raises(DepthAlignmentError, match="anchors"):
        fit_inverse_depth(1 / depth[:10], depth[:10])
    with pytest.raises(DepthAlignmentError, match="ill-conditioned"):
        fit_inverse_depth(np.ones(len(depth)), depth)
    with pytest.raises(DepthAlignmentError, match="positive"):
        fit_inverse_depth(-1 / depth, depth)


def test_calibration_samples_subpixel_and_backprojects_only_valid_road(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 20 0 0 0 0 20 0 0 0 0 1 0.05\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    calib = KittiCalibration(path)
    prediction = np.tile(np.arange(30, dtype=float), (5, 1))
    u = np.linspace(.2, 29, 200)
    uv = np.column_stack((u, np.full_like(u, 1.5)))
    anchors = calib.backproject_image_to_lidar(uv, 1 / (.006 * u + .016))
    fit = calibrate_from_road_points(prediction, anchors, calib)
    np.testing.assert_allclose([fit.scale, fit.shift], [.006, .016], atol=1e-10)
    road = np.zeros(prediction.shape, dtype=bool)
    road[1, :6] = True
    prediction[1, 0] = np.nan
    prediction[1, 1] = -10  # Negative calibrated inverse depth.
    visual, calibrated = backproject_visual_road(prediction, road, fit, calib)
    assert len(visual) == 4
    assert np.count_nonzero(np.isfinite(calibrated)) == 4
    reprojected, optical = calib.project_lidar_to_image(visual)
    np.testing.assert_allclose(reprojected, [[2, 1], [3, 1], [4, 1], [5, 1]], atol=1e-6)
    np.testing.assert_allclose(optical, calibrated[1, 2:6], rtol=1e-6)
