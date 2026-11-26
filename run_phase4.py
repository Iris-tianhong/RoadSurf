"""Compare DA-V2 and Any2Full at 64/32/16 simulated KITTI beam counts."""

import argparse
import csv
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from core.depth_alignment import (
    DepthAlignmentError, backproject_road_depth, backproject_visual_road, calibrate_from_road_points,
)
from core.road_lidar_extract import estimate_patchwork_mask, extract_road_lidar, road_interior
from core.sgsr_solver import fit_ransac_plane, fit_single_spline, make_spline_basis
from data.beam_downsample import beam_mask, recover_rings
from data.dataset_loader import load_kitti_frame
from data.kitti_dc import read_dc_depth
from evaluation.eval_road import (
    aggregate_rows, dc_reference_points, evaluate_surface_at_reference,
    heldout_scan_points, metric_rows, reference_slopes,
)
from frontends.run_any2full import load_any2full_predictor
from frontends.run_dav2 import load_dav2_predictor
from frontends.semantic_loader import load_semantic_labels


def _file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_depth(path, fingerprint, shape, predict):
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            if (str(data["fingerprint"]) != fingerprint or data["depth"].shape != shape
                    or data["depth"].dtype != np.float32):
                raise ValueError(f"Stale or invalid prediction cache: {path}")
            return data["depth"], float(data["elapsed_s"])
    start = perf_counter()
    depth = predict().astype(np.float32, copy=False)
    elapsed = perf_counter() - start
    if depth.shape != shape or not np.isfinite(depth).any():
        raise ValueError(f"Depth prediction has an invalid output grid: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, depth=depth, fingerprint=fingerprint, elapsed_s=elapsed)
    partial.replace(path)
    return depth, elapsed


def _write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_frame(record, args, dav2, any2full, fingerprints, basis):
    id_ = record["frame"]
    frame = load_kitti_frame(args.kitti_root, "training", int(id_))
    image, raw, calib = frame.image, frame.points, frame.calibration
    size = image.size
    shape = size[::-1]
    labels, _ = load_semantic_labels(
        args.semantic_dir / f"{id_}.npy", size, args.semantic_format)
    road = road_interior(labels, 2)
    rings, angles, spans, spreads = recover_rings(raw)
    image_key = sha256(image.tobytes()).hexdigest()
    relative, dav2_s = _cached_depth(
        args.cache_dir / "dav2" / f"{id_}.npz",
        f"{fingerprints['dav2']}:{image_key}", shape, lambda: dav2(image))

    setups, timing = [], []
    failures = []
    beam_counts = {}
    for beams in (64, 32, 16):
        keep = beam_mask(rings, beams)
        scan = raw[keep]
        sparse = calib.create_sparse_depth_map(scan, size)
        ground = estimate_patchwork_mask(scan)
        road_points, _, _ = extract_road_lidar(scan, calib, labels, ground)
        lidar = road_points[:, :3].astype(np.float64)
        if beams == 64:
            full_road, full_sparse = road_points, sparse
        beam_counts[beams] = {"raw_points": int(keep.sum()), "input_depth_pixels": int(np.isfinite(sparse).sum()),
                              "road_anchors": len(lidar)}
        sparse_key = sha256(sparse.astype("<f4").tobytes()).hexdigest()
        metric, any2full_s = _cached_depth(
            args.cache_dir / "any2full" / f"{id_}_{beams}.npz",
            f"{fingerprints['any2full']}:{image_key}:{sparse_key}", shape,
            lambda: any2full(image, sparse))
        plane, _ = fit_ransac_plane(lidar)
        fits = {"lidar": fit_single_spline(lidar, basis, plane)}
        start = perf_counter()
        try:
            alignment = calibrate_from_road_points(relative, lidar, calib)
            visual, _ = backproject_visual_road(relative, road, alignment, calib)
            if not len(visual):
                raise DepthAlignmentError("No valid calibrated visual road points.")
            fits["dav2"] = fit_single_spline(lidar, basis, plane, visual)
        except DepthAlignmentError as exc:
            fits["dav2"] = None
            failures.append({"beams": beams, "method": "dav2", "reason": str(exc)})
        dav2_fusion_s = perf_counter() - start
        start = perf_counter()
        visual, _ = backproject_road_depth(metric, road, calib)
        fits["any2full"] = fit_single_spline(lidar, basis, plane, visual) if len(visual) else None
        if not len(visual):
            failures.append({"beams": beams, "method": "any2full", "reason": "No valid visual road points."})
        timing.extend([
            dict(frame=id_, beams=beams, method="dav2", inference_s=dav2_s,
                 calibration_and_fusion_s=dav2_fusion_s),
            dict(frame=id_, beams=beams, method="any2full", inference_s=any2full_s,
                 calibration_and_fusion_s=perf_counter() - start),
        ])
        held = heldout_scan_points(full_road, scan) if beams < 64 else None
        setups.append((beams, lidar, plane, fits, held))

    # DC is opened after every method has already produced its surface.
    depth = read_dc_depth(args.kitti_root / record["depth_png"])
    all_dc, non_input, checks = dc_reference_points(depth, full_sparse, road, calib)
    dc_sets = [(key, xyz, reference_slopes(xyz)) for key, xyz in (
        ("dc_all", all_dc), ("dc_non_input", non_input))]
    rows, by_beam = [], {}
    for beams, lidar, plane, fits, held in setups:
        references = list(dc_sets)
        if held is not None:
            references.append(("heldout_lidar", held, reference_slopes(held)))
        for key, points, slopes in references:
            predictions = evaluate_surface_at_reference(points, lidar, calib, road, plane, basis, fits)
            if key != "heldout_lidar":
                by_beam[(key, beams)] = predictions
            rows.extend(dict(frame=id_, drive=record["drive"], beams=beams, reference=key, **row)
                        for row in metric_rows(points, slopes, predictions))
    cross_beam = []
    for key, points, slopes in dc_sets:
        joint = np.logical_and.reduce([
            np.isfinite(height) for beams in (64, 32, 16)
            for height, _ in by_beam[(key, beams)].values()])
        for beams in (64, 32, 16):
            cross_beam.extend(
                dict(frame=id_, drive=record["drive"], beams=beams, reference=key, **row)
                for row in metric_rows(points, slopes, by_beam[(key, beams)], common_mask=joint))
    audit = dict(frame=id_, drive=record["drive"], ring_count=len(angles),
                 beam_elevations_deg=angles.tolist(), ring_min_azimuth_span_deg=float(min(spans)),
                 ring_max_corrected_angle_mad_deg=float(max(spreads)), beams=beam_counts,
                 dc_road_points=len(all_dc), dc_non_input_road_points=len(non_input), failures=failures, **checks)
    return rows, cross_beam, timing, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "kitti-root", "semantic-dir", "dav2-model", "any2full-repo",
                 "any2full-checkpoint", "cache-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--semantic-format", choices=("probabilities", "trainId"), default="probabilities")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", nargs="+", help="Optional subset of the locked manifest")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new --out directory.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [r for r in manifest["frames"] if not args.frames or r["frame"] in args.frames]
    if not records or (args.frames and {r["frame"] for r in records} != set(args.frames)):
        parser.error("--frames must be in the locked manifest.")

    import torch
    torch.manual_seed(42)
    fingerprints = {
        "dav2": ":".join(_file_hash(args.dav2_model / name) for name in
                         ("model.safetensors", "config.json", "preprocessor_config.json")),
        "any2full": _file_hash(args.any2full_checkpoint) + ":" +
                    _file_hash(args.any2full_repo / "model/ours/any2full.py"),
    }
    dav2, provenance = load_dav2_predictor(str(args.dav2_model), args.device)
    any2full = load_any2full_predictor(args.any2full_repo, args.any2full_checkpoint, args.device)
    basis = make_spline_basis(4)
    statistics, cross_beam_statistics, timings, audit = [], [], [], []
    for record in records:
        rows, cross_beam, timing, info = _evaluate_frame(record, args, dav2, any2full, fingerprints, basis)
        statistics.extend(rows)
        cross_beam_statistics.extend(cross_beam)
        timings.extend(timing)
        audit.append(info)
        print(f'{info["frame"]}: {info["dc_non_input_road_points"]} DC road points outside full-scan pixels',
              flush=True)

    args.out.mkdir(parents=True, exist_ok=False)
    _write_csv(args.out / "per_frame.csv", statistics)
    _write_csv(args.out / "summary.csv", aggregate_rows(statistics))
    _write_csv(args.out / "per_drive.csv", aggregate_rows(statistics, by_drive=True))
    _write_csv(args.out / "cross_beam_summary.csv", aggregate_rows(cross_beam_statistics))
    _write_csv(args.out / "timings.csv", timings)
    (args.out / "audit.json").write_text(json.dumps({
        "manifest": str(args.manifest), "manifest_sha256": _file_hash(args.manifest),
        "dav2": provenance, "any2full_checkpoint": str(args.any2full_checkpoint),
        "any2full_repository": str(args.any2full_repo), "prediction_fingerprints": fingerprints,
        "versions": {"pypatchworkpp": version("pypatchworkpp"), "scipy": scipy.__version__},
        "frames": audit,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
