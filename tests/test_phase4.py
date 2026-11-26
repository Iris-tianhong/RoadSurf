import numpy as np
from PIL import Image
import pytest

from data.beam_downsample import beam_mask, recover_rings
from data.kitti_calib import KittiCalibration
from data.kitti_dc import read_dc_depth
from evaluation.eval_road import aggregate_rows, dc_reference_points, metric_rows, reference_slopes


def test_ordered_nonuniform_rings_with_separate_origins():
    az = np.deg2rad(np.linspace(2, 352, 90))
    radius = 18 + 7 * np.sin(az)
    angles = np.r_[np.linspace(2.5, -7.8, 32), np.linspace(-8.35, -23.6, 32)]
    scan = np.vstack([
        np.c_[radius * np.cos(az), radius * np.sin(az), radius * np.tan(np.deg2rad(angle))
              + .006 * ring]
        for ring, angle in enumerate(angles)])
    rings, fitted, spans, spread = recover_rings(scan)
    np.testing.assert_array_equal(rings, np.repeat(np.arange(64), len(az)))
    np.testing.assert_allclose(fitted, angles, atol=1e-12)
    assert min(spans) > 300 and max(spread) < 1e-10
    assert beam_mask(rings, 64).sum() == len(scan)
    assert beam_mask(rings, 32).sum() == len(scan) // 2
    assert beam_mask(rings, 16).sum() == len(scan) // 4
    assert not (beam_mask(rings, 16) & ~beam_mask(rings, 32)).any()
    with pytest.raises(ValueError, match="azimuth"):
        recover_rings(scan[::-1])


def test_kitti_dc_reference_excludes_full_scan_pixel_neighborhood(tmp_path):
    depth_png = tmp_path / "depth.png"
    codes = np.full((20, 20), 5120, dtype=np.uint16)
    codes[19, 19] = 0
    Image.fromarray(codes).save(depth_png)
    depth = read_dc_depth(depth_png)
    assert depth[0, 0] == 20 and np.isnan(depth[-1, -1])
    calibration = tmp_path / "calib.txt"
    calibration.write_text(
        "P2: 20 0 0 0 0 20 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    sparse = np.full_like(depth, np.nan)
    sparse[:10, :10] = 20
    road = np.ones_like(depth, dtype=bool)
    all_points, non_input, audit = dc_reference_points(
        depth, sparse, road, KittiCalibration(calibration))
    assert audit["dc_full_scan_overlap_px"] == 100
    assert len(non_input) < len(all_points)
    assert not ((non_input[:, 0] <= 10) & (non_input[:, 1] <= 10)).any()


def test_local_reference_normals_and_identical_common_denominator():
    x, y = np.meshgrid(np.linspace(5, 15, 40), np.linspace(-3, 3, 30))
    reference = np.c_[x.ravel(), y.ravel(), .15*x.ravel() - .07*y.ravel()]
    slopes = reference_slopes(reference)
    known = np.isfinite(slopes).all(axis=1)
    assert known.sum() > 30
    np.testing.assert_allclose(slopes[known], [[.15, -.07]] * known.sum(), atol=1e-12)

    grad = np.tile([.15, -.07], (len(reference), 1))
    a, b = reference[:, 2] + .02, reference[:, 2] - .03
    a[:20] = np.nan
    b[-25:] = np.nan
    rows = [dict(beams=16, reference="dc_non_input", **row) for row in
            metric_rows(reference, slopes, {"a": (a, grad), "b": (b, grad)})]
    stats = {r["method"]: r for r in aggregate_rows(rows) if r["distance_m"] == "all"}
    assert stats["a"]["common_covered"] == stats["b"]["common_covered"] == len(reference) - 45
    assert stats["a"]["height_bias_m"] == pytest.approx(.02)
    assert stats["b"]["common_height_rmse_m"] == pytest.approx(.03)
    assert stats["a"]["common_normal_mae_deg"] < 1e-5
    joint = np.arange(len(reference)) > 50
    cross_beam = [r for r in metric_rows(reference, slopes, {"a": (a, grad), "b": (b, grad)}, joint)
                  if r["distance_m"] == "all"]
    assert all(r["common_covered"] == len(reference) - 51 - 25 for r in cross_beam)
    assert cross_beam[0]["covered"] == len(reference) - 20
