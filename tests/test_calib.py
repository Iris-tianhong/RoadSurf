import numpy as np
import pytest

from core.road_lidar_extract import _local_surface_inliers, extract_road_lidar
from data.kitti_calib import KittiCalibration


def test_projection_round_trip_includes_camera_translation_and_rectification(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 730 0 600 44 0 730 185 -2 0 0 1 0.003\n"
        "R0_rect: 1 0 0 0 0.9998 -0.02 0 0.02 0.9998\n"
        "Tr_velo_to_cam: 0 -1 0 0.01 0 0 -1 -0.08 1 0 0 -0.27\n"
    )
    xyz = np.array([[5.3, 1.1, -1.6], [20.1, -3.9, -0.9], [65.3, 1.2, 0.8]])
    calib = KittiCalibration(path)
    uv, depth = calib.project_lidar_to_image(xyz)
    first_rect = np.array([[1, 0, 0], [0, .9998, -.02], [0, .02, .9998]]) @ [
        -1.09, 1.52, 5.03]
    np.testing.assert_allclose(depth[0], first_rect[2] + 0.003)
    np.testing.assert_allclose(
        uv[0], ([730 * first_rect[0] + 600 * first_rect[2] + 44,
                 730 * first_rect[1] + 185 * first_rect[2] - 2] / depth[0]))
    recovered = calib.backproject_image_to_lidar(uv, depth)
    assert np.linalg.norm(recovered - xyz, axis=1).max() < 1e-5
    center, rays = calib.pixel_to_lidar_ray(uv)
    np.testing.assert_allclose(calib.projection @ np.r_[center, 1.0], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rays @ calib.projection[2, :3], np.ones(len(xyz)))

    # 裁剪 (40, 22) 后再对像素中心做横纵不同比例的缩放。
    transform = np.array([[.5, 0, -20.25], [0, .8, -17.7], [0, 0, 1]])
    cropped = KittiCalibration(path, image_transform=transform)
    crop_uv, crop_depth = cropped.project_lidar_to_image(xyz)
    np.testing.assert_allclose(crop_uv, (uv - [40, 22] + .5) * [.5, .8] - .5)
    np.testing.assert_allclose(crop_depth, depth)
    np.testing.assert_allclose(cropped.backproject_image_to_lidar(crop_uv, crop_depth), xyz)


def test_sparse_depth_uses_nearest_valid_point_and_leaves_empty_pixels_nan(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    calib = KittiCalibration(path)
    points = np.array([
        [2.2, 2.2, 2], [1.1, 1.1, 1], [4.9, 5.9, 1], [-.1, 0, 1],
        [5, 3, 1], [0, 0, 0], [-1, -1, -1], [np.inf, 1, 1],
    ])
    depth = calib.create_sparse_depth_map(points, (5, 6))
    assert depth.shape == (6, 5)
    assert np.count_nonzero(np.isfinite(depth)) == 2
    assert depth[1, 1] == depth[5, 4] == 1
    assert np.isnan(depth[0, 0])


def test_calibration_rejects_trailing_garbage_in_matrix(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0 garbage\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    with pytest.raises(ValueError, match="Invalid KITTI matrix P2"):
        KittiCalibration(path)


def test_road_uses_nearest_full_scan_and_preserves_intensity(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    # Camera depth is z here. The nearer return at pixel (1,1) is outside
    # the y ROI, but it must still occlude the farther point inside the ROI.
    points = np.array([
        [22.5, 19.0, 15.5, .75], [22.5, 21.0, 15.0, .3],
        [3.0, 1.0, 1.0, .8], [-0.1, 0.1, 1, .1],
        [2.0, 2.0, -1, .2], [4.0, 1.0, 1, np.nan], [0.1, 0.1, 2, .9],
        [1.0, 3.0, 1.0, .2],
    ], dtype=np.float32)
    labels = np.full((5, 5), 255, dtype=np.uint8)
    labels[1, 1] = labels[1, 3] = labels[3, 1] = 0
    labels[0, 0] = 1
    ground = np.array([True, True, True, True, True, True, True, False])
    calib = KittiCalibration(path)
    selected, pixels, counts = extract_road_lidar(points, calib, labels, ground, 0)
    np.testing.assert_array_equal(selected, points[[2]])
    np.testing.assert_array_equal(pixels, [[3, 1]])
    assert counts["semantic_road_points"] == 2
    assert counts["road_points_after_geometry"] == 1
    depth = calib.create_sparse_depth_map(points[:2], (5, 5))
    np.testing.assert_allclose(depth[1, 1], 15.0)


def test_local_residual_keeps_a_slope_and_removes_height_jump():
    x, y = np.meshgrid(np.linspace(10, 14, 21), np.linspace(-2, 2, 21))
    xyz = np.column_stack((x.ravel(), y.ravel(), (.2 * x - .08 * y).ravel()))
    raised = xyz[[181, 180, 182]].copy()
    raised[:, 2] += .12
    keep = _local_surface_inliers(np.vstack((xyz, raised)))
    assert keep[:len(xyz)].all()
    assert not keep[len(xyz):].any()
