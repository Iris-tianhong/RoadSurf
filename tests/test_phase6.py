import numpy as np

from evaluation.eval_road import dc_reference_points, heldout_scan_points
from evaluation.synthetic_holes import SyntheticHole


def test_reference_pixel_guard_expands_around_full_scan():
    class Calibration:
        @staticmethod
        def backproject_image_to_lidar(uv, depth):
            u, v = uv.T
            return np.column_stack((u + 5, v * .10, depth))

    depth = np.full((30, 20), 12.)
    scan = np.full_like(depth, np.nan)
    scan[:10] = depth[:10]
    road = np.ones(depth.shape, dtype=bool)
    all_road, v1, audit = dc_reference_points(depth, scan, road, Calibration())
    _, v2, _ = dc_reference_points(depth, scan, road, Calibration(), exclusion_pixels=2)
    assert len(all_road) == 600
    assert audit["dc_full_scan_overlap_px"] == 200
    assert len(v1) == 380 and np.isclose(v1[:, 1].min(), 1.1)
    assert len(v2) == 360 and np.isclose(v2[:, 1].min(), 1.2)


def test_heldout_references_exclude_input_records_and_core_excludes_hole_fringe():
    road = np.array([[15, 0, -1.6, .1], [16.8, 0, -1.6, .2],
                     [14, .75, -1.6, .3], [18, 0, -1.6, .4]], dtype=np.float32)
    hole = SyntheticHole(np.array([15., 0.]), (4, 2))
    kept = road[~hole.contains(road)]
    held = heldout_scan_points(road, kept)
    np.testing.assert_array_equal(held, road[:3, :3])
    assert hole.contains(held).all()
    np.testing.assert_array_equal(held[hole.contains(held, margin=.60)], road[:1, :3])
