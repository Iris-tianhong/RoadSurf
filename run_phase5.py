"""Run sequential SGSR ablations on the locked Phase 4 frames and synthetic holdouts."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from core.depth_alignment import (
    DepthAlignmentError, backproject_visual_road, calibrate_from_road_points,
)
from core.road_lidar_extract import estimate_patchwork_mask, extract_road_lidar, road_interior
from core.sgsr import reconstruct
from core.sgsr_export import export_reconstruction
from core.sgsr_solver import make_spline_basis
from data.beam_downsample import beam_mask, recover_rings
from data.dataset_loader import load_kitti_frame
from data.kitti_dc import read_dc_depth
from evaluation.eval_road import (
    aggregate_rows, dc_reference_points, heldout_scan_points, in_evaluation_range, metric_rows,
    reference_slopes,
)
from evaluation.synthetic_holes import HOLE_SIZES_XY, choose_hole
from frontends.semantic_loader import load_semantic_labels


def _observations(frame, scan, labels, road_image, relative, hole, input_audit=None):
    raw_before = len(scan)
    if hole is not None:
        scan = scan[~hole.contains(scan)]
        if hole.contains(scan).any():
            raise AssertionError("Synthetic holdout reached raw scan preprocessing.")
    ground = estimate_patchwork_mask(scan)
    real, _, _ = extract_road_lidar(scan, frame.calibration, labels, ground)
    lidar = real[:, :3].astype(np.float64)
    failure = None
    visual = np.empty((0, 3), dtype=np.float64)
    visual_removed = 0
    try:
        alignment = calibrate_from_road_points(relative, lidar, frame.calibration)
        visual = backproject_visual_road(relative, road_image, alignment, frame.calibration)[0]
        if hole is not None:
            in_hole = hole.contains(visual)
            visual_removed = int(in_hole.sum())
            visual = visual[~in_hole]
    except DepthAlignmentError as exc:
        failure = str(exc)
    if input_audit is not None:
        input_audit.update(raw_retained=len(scan), raw_removed=raw_before - len(scan),
                           visual_removed=visual_removed)
    return real, visual, frame.calibration.create_sparse_depth_map(scan, frame.image.size), failure


def _metrics_for(result, references, context, inferred_only=False):
    slopes = reference_slopes(references)
    x, y = references[:, :2].T
    if inferred_only:
        surface = result.surfaces["A4"]
        z, gradient = surface.values(x, y)
        keep = surface.is_hidden(x, y)
        z[~keep] = np.nan
        gradient[~keep] = np.nan
        predictions = {"A4_hidden": (z, gradient)}
    else:
        predictions = {method: surface.values(x, y) for method, surface in result.surfaces.items()}
    return [dict(**context, **item) for item in metric_rows(references, slopes, predictions)]


def _diagnostics(result, lidar, visual, failure, elapsed):
    low = result.fine_control_support < .25
    rms = {}
    for name in ("A1", "A2", "A3"):
        coeff = result.surfaces[name].model.levels[1][1]
        rms[name] = float(np.sqrt(np.mean(coeff[low]**2))) if low.any() else None
    return {
        "lidar": int(len(lidar)), "visual": int(len(visual)), "alignment_failure": failure,
        "fit_and_domain_s": elapsed, "visible_grid": int(result.surfaces["A4"].visible.sum()),
        "hidden_grid": int(result.surfaces["A4"].hidden.sum()),
        "low_support_control_rms_m": rms,
        "vision_weight_range": ([float(result.visual_weights.min()), float(result.visual_weights.max())]
                                if len(result.visual_weights) else []),
        "median_abs_ray_jacobian": (float(np.median(np.abs(result.ray_jacobian)))
                                    if len(result.ray_jacobian) else None),
    }


def _cached_prediction(args, frame):
    path = args.dav2_cache / f"{frame.frame_id}.npz"
    with np.load(path, allow_pickle=False) as cache:
        prediction = cache["depth"]
        rgb_hash = sha256(frame.image.tobytes()).hexdigest()
        if (not str(cache["fingerprint"]).endswith(":" + rgb_hash)
                or prediction.shape != frame.image.size[::-1] or prediction.dtype != np.float32):
            raise ValueError(f"Phase 4 prediction cache does not match this frame: {path}")
        return prediction


def process_ablation_frame(record, args, coarse, fine, acceptance=False):
    id_ = record["frame"]
    frame = load_kitti_frame(args.kitti_root, "training", int(id_))
    labels, _ = load_semantic_labels(
        args.semantic_dir / f"{id_}.npy", frame.image.size, args.semantic_format)
    road_image = road_interior(labels, 2)
    relative = _cached_prediction(args, frame)
    rings, _, _, _ = recover_rings(frame.points)
    native, erased, audit = [], [], []
    holes = []
    for beams in (64, 32, 16):
        scan = frame.points[beam_mask(rings, beams)]
        start = perf_counter()
        observation_audit = {}
        road_returns, visual, sparse, failure = _observations(
            frame, scan, labels, road_image, relative, None, observation_audit if acceptance else None)
        lidar = road_returns[:, :3].astype(np.float64)
        result = reconstruct(lidar, visual, frame.calibration, labels, sparse, coarse, fine,
                             include_baselines=acceptance)
        native.append((beams, result))
        audit.append(dict(beams=beams, condition="native", **observation_audit, **_diagnostics(
            result, lidar, visual, failure, perf_counter() - start)))
        if beams == 64:
            full_sparse, full_lidar, full_road_returns = sparse, lidar, road_returns
            holes = [choose_hole(result.grid, result.surfaces["A0"].visible, size)
                     for size in HOLE_SIZES_XY]
            export_reconstruction(args.out / "frames" / id_, result, frame.image, frame.calibration,
                                  {"frame": id_, "beams": 64, "gap_kind": "observed_sensor_occlusion"})

        for hole in holes:
            if hole is None:
                continue
            start = perf_counter()
            observation_audit = {}
            erased_returns, v_erased, s_erased, failure = _observations(
                frame, scan, labels, road_image, relative, hole, observation_audit if acceptance else None)
            l_erased = erased_returns[:, :3].astype(np.float64)
            if hole.contains(l_erased).any() or hole.contains(v_erased).any():
                raise AssertionError("Synthetic holdout reached the fitting inputs.")
            filled = reconstruct(l_erased, v_erased, frame.calibration, labels, s_erased,
                                 coarse, fine, synthetic=True, include_baselines=acceptance)
            erased.append((beams, hole, filled))
            audit.append(dict(beams=beams, condition=hole.name, xy=hole.center.tolist(),
                              **observation_audit,
                              **_diagnostics(filled, l_erased, v_erased, failure,
                                             perf_counter() - start)))
            sample = args.out / "examples" / f"{hole.name}_{beams:02d}beams"
            if beams in (64, 16) and not sample.exists() and filled.surfaces["A4"].hidden.any():
                export_reconstruction(sample, filled, frame.image, frame.calibration,
                                      {"frame": id_, "beams": beams,
                                       "gap_kind": "synthetic_geometry_holdout",
                                       "rectangle_center_xy": hole.center.tolist(),
                                       "rectangle_size_xy": list(hole.size_xy)})

    # Reference depth is deliberately opened only after native and hole fits.
    depth_path = args.kitti_root / record["depth_png"]
    depth = read_dc_depth(depth_path)
    _, non_input, check = dc_reference_points(depth, full_sparse, road_image, frame.calibration)
    if acceptance:
        _, guarded, _ = dc_reference_points(depth, full_sparse, road_image, frame.calibration,
                                            exclusion_pixels=2)
        # A wider image exclusion and a 3D exclusion test sensitivity to
        # shifted or independently quantized projections of the input scan.
        finite_scan = frame.points[np.isfinite(frame.points[:, :3]).all(axis=1), :3]
        distances, _ = cKDTree(finite_scan).query(guarded)
        guarded = guarded[distances > .10]
        check.update(dc_png_sha256=sha256(depth_path.read_bytes()).hexdigest(),
                     dc_non_input_points=len(non_input), dc_guarded_points=len(guarded))
    rows = []
    for beams, result in native:
        context = dict(frame=id_, drive=record["drive"], beams=beams, reference="dc_non_input")
        rows.extend(_metrics_for(result, non_input, context))
        if acceptance:
            rows.extend(_metrics_for(result, guarded, dict(context, reference="dc_guarded")))
            if beams < 64:
                held = heldout_scan_points(full_road_returns, frame.points[beam_mask(rings, beams)])
                rows.extend(_metrics_for(result, held, dict(context, reference="removed_ring_lidar")))
    reference_lidar = full_lidar[in_evaluation_range(full_lidar)]
    for beams, hole, filled in erased:
        for suffix, all_points in (("heldout_lidar", reference_lidar), ("dc_non_input", non_input)):
            xyz = all_points[hole.contains(all_points)]
            context = dict(frame=id_, drive=record["drive"], beams=beams,
                           reference=f"{hole.name}_{suffix}")
            rows.extend(_metrics_for(filled, xyz, context))
            context["reference"] += "_inferred_only"
            rows.extend(_metrics_for(filled, xyz, context, inferred_only=True))
            if acceptance:
                core = xyz[hole.contains(xyz, margin=.60)]
                rows.extend(_metrics_for(filled, core, dict(context, reference=f"{hole.name}_{suffix}_core")))
    metadata = dict(frame=id_, drive=record["drive"], **check,
                    eligible_holes=[hole.name for hole in holes if hole is not None], cases=audit)
    return rows, metadata


def _write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "kitti-root", "semantic-dir", "dav2-cache", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--semantic-format", choices=("probabilities", "trainId"), default="probabilities")
    parser.add_argument("--frames", nargs="+")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["frames"]
    if args.frames:
        records = [item for item in records if item["frame"] in args.frames]
        if {item["frame"] for item in records} != set(args.frames):
            parser.error("--frames must belong to the locked manifest.")
    if not records or args.out.exists():
        parser.error("Provide at least one frame and a new output directory.")
    coarse, fine = make_spline_basis(4), make_spline_basis(2)
    rows, audit = [], []
    for record in records:
        row, info = process_ablation_frame(record, args, coarse, fine)
        rows.extend(row)
        audit.append(info)
        print(f'{info["frame"]}: {len(info["cases"])} setups; holes {info["eligible_holes"]}', flush=True)
    _write_csv(args.out / "ablation_per_frame.csv", rows)
    _write_csv(args.out / "ablation_summary.csv", aggregate_rows(rows))
    (args.out / "audit.json").write_text(json.dumps({
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "ablations": {"A0": "coarse_only", "A1": "add_fine_residual", "A2": "add_support_damping",
                      "A3": "add_visual_geometry_weights", "A4": "add_bounded_hidden_domain"},
        "frames": audit,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
