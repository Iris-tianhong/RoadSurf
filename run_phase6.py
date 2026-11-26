"""Prepare sealed RGB predictions and run the Phase 6 held-out acceptance evaluation."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from PIL import Image

from data.phase6_holdout import file_hash, input_hashes
from evaluation.eval_road import aggregate_rows
from frontends.run_dav2 import load_dav2_predictor
from core.sgsr_solver import make_spline_basis
from run_phase4 import _cached_depth, _write_csv
from run_phase5 import process_ablation_frame


METHOD_SOURCES = (
    "run_phase6.py", "run_phase5.py", "core/depth_alignment.py", "core/road_lidar_extract.py",
    "core/bspline_basis.py", "core/sgsr_solver.py", "core/hierarchical_surface.py", "core/sgsr.py",
    "core/support_and_weight.py", "core/road_domain.py", "data/beam_downsample.py", "data/kitti_calib.py",
    "data/kitti_dc.py", "evaluation/eval_road.py", "evaluation/synthetic_holes.py",
    "frontends/run_dav2.py", "frontends/semantic_loader.py",
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def _verify_inputs(kitti, records):
    for record in records:
        if input_hashes(kitti, record["frame"]) != record["input_sha256"]:
            raise ValueError(f"KITTI input no longer matches the holdout lock: {record['frame']}")


def _semantic_source(args):
    return {"checkpoint": str(args.bisenet_weights), "checkpoint_sha256": file_hash(args.bisenet_weights),
            "exporter_sha256": file_hash(args.semantic_exporter),
            "source_files": {name: file_hash(args.bisenet_repo / name) for name in (
                "BiSeNetv2/model/BiseNetv2.py", "BiSeNetv2/utils/utils.py", "BiSeNetv2/dataset.py")}}


def _semantic_stage(args, records, manifest_hash):
    staged = args.cache_dir / "rgb"
    semantic = args.cache_dir / "semantic"
    seal = args.cache_dir / "semantic_provenance.json"
    source = _semantic_source(args)
    if seal.is_file():
        saved = _read_json(seal)
        actual = {r["frame"]: file_hash(semantic / f"{r['frame']}.npy") for r in records}
        if (saved["manifest_sha256"] != manifest_hash or saved["source"] != source
                or saved["semantic_sha256"] != actual):
            raise ValueError("Cannot resume from mismatched BiSeNet cache contents.")
        return saved
    args.cache_dir.mkdir(parents=True, exist_ok=False)
    staged.mkdir()
    for record in records:
        name = f"{record['frame']}.png"
        shutil.copyfile(args.kitti_root / "training/image_2" / name, staged / name)
    start = perf_counter()
    subprocess.run([
        sys.executable, "-B", str(args.semantic_exporter), "--reference", str(args.bisenet_repo),
        "--weights", str(args.bisenet_weights), "--images", str(staged), "--output", str(semantic),
    ], check=True)
    elapsed = perf_counter() - start
    if _semantic_source(args) != source:
        raise ValueError("BiSeNet inputs changed during export.")
    saved = {"manifest_sha256": manifest_hash, "source": source, "elapsed_batch_s": elapsed,
             "semantic_sha256": {r["frame"]: file_hash(semantic / f"{r['frame']}.npy") for r in records}}
    _write_json(seal, saved)
    return saved


def _prepare(args, records):
    if (args.cache_dir / "provenance.json").exists():
        raise FileExistsError("Prediction cache is already complete.")
    _verify_inputs(args.kitti_root, records)
    manifest_hash = file_hash(args.manifest)
    predict, depth_source = load_dav2_predictor(str(args.dav2_model), args.device)
    dav2_inputs = {name: file_hash(args.dav2_model / name) for name in
                   ("model.safetensors", "config.json", "preprocessor_config.json")}
    stamp = ":".join(dav2_inputs.values())
    semantic = _semantic_stage(args, records, manifest_hash)
    outputs = []
    for record in records:
        id_ = record["frame"]
        with Image.open(args.kitti_root / "training/image_2" / f"{id_}.png") as source:
            image = source.convert("RGB")
        depth_path = args.cache_dir / "dav2" / f"{id_}.npz"
        _, elapsed = _cached_depth(depth_path, f"{stamp}:{record['input_sha256']['rgb']}",
                                  image.size[::-1], lambda: predict(image))
        outputs.append({"frame": id_, "rgb_sha256": record["input_sha256"]["rgb"],
                        "semantic_sha256": semantic["semantic_sha256"][id_],
                        "dav2_sha256": file_hash(depth_path), "dav2_inference_s": elapsed})
        print(f"{id_}: sealed semantic and relative depth", flush=True)
    _verify_inputs(args.kitti_root, records)
    _write_json(args.cache_dir / "provenance.json", {
        "manifest_sha256": manifest_hash, "frames": outputs,
        "bisenet": {**semantic["source"], "elapsed_batch_s": semantic["elapsed_batch_s"],
                    "recovered_from_completed_export": semantic.get("recovered_from_completed_export", False)},
        "dav2": {**depth_source, "files_sha256": dav2_inputs},
    })


def _evaluate(args, records):
    provenance = _read_json(args.cache_dir / "provenance.json")
    manifest_hash = file_hash(args.manifest)
    if manifest_hash != provenance["manifest_sha256"]:
        raise ValueError("Prediction provenance belongs to a different holdout.")
    observed = {r["frame"]: r for r in provenance["frames"]}
    if set(observed) != {r["frame"] for r in records} or len(observed) != len(provenance["frames"]):
        raise ValueError("Missing or duplicate sealed prediction records.")
    _verify_inputs(args.kitti_root, records)
    args.semantic_dir = args.cache_dir / "semantic"
    args.dav2_cache = args.cache_dir / "dav2"
    args.semantic_format = "probabilities"
    for record in records:
        id_ = record["frame"]
        p = observed[id_]
        if (p["rgb_sha256"] != record["input_sha256"]["rgb"]
                or p["semantic_sha256"] != file_hash(args.semantic_dir / f"{id_}.npy")
                or p["dav2_sha256"] != file_hash(args.dav2_cache / f"{id_}.npz")):
            raise ValueError(f"Prediction input or content changed for {id_}.")
    root = Path(__file__).resolve().parent
    audit = {
        "manifest_sha256": manifest_hash, "prediction_provenance": provenance,
        "source_files_sha256": {name: file_hash(root / name) for name in METHOD_SOURCES},
        "versions": {name: version(name) for name in ("numpy", "scipy", "pypatchworkpp", "Pillow")},
        "conditions": ["native", "hole_2x4", "hole_3x6"], "beams": [64, 32, 16],
        "coverages": ["own", "common across A0-A4 and B0-B1"],
        "reference_independence": {
            "dc_non_input": "Excludes complete current scan projected pixels plus a one-pixel margin.",
            "dc_guarded": "Two-pixel margin and more than 0.10 m from any current scan point in XYZ.",
            "removed_ring_lidar": "Full-scan road points absent from the simulated sparse raw scan.",
            "hole": "Raw observations deleted before Patchwork++ and calibration. Visual XYZ deleted before fit. Original RGB retained.",
            "hole_core": "Evaluation only; at least 0.60 m from the four edges of the deleted rectangle.",
        },
    }
    args.out.mkdir(parents=True, exist_ok=False)
    _write_json(args.out / "input_audit.json", audit)
    coarse, fine = make_spline_basis(4), make_spline_basis(2)
    metrics, evaluated, runtimes = [], [], []
    for record in records:
        rows, frame = process_ablation_frame(record, args, coarse, fine, acceptance=True)
        metrics.extend(rows)
        evaluated.append(frame)
        runtimes.extend({"frame": frame["frame"], "drive": frame["drive"], "beams": case["beams"],
                         "condition": case["condition"], "observation_and_reconstruction_s": case["fit_and_domain_s"]}
                        for case in frame["cases"])
        print(f"{frame['frame']}: {len(frame['cases'])} setups; holes {frame['eligible_holes']}", flush=True)
    _write_csv(args.out / "per_frame.csv", metrics)
    _write_csv(args.out / "summary.csv", aggregate_rows(metrics))
    _write_csv(args.out / "per_drive.csv", aggregate_rows(metrics, by_drive=True))
    _write_csv(args.out / "runtime.csv", runtimes)
    _write_json(args.out / "frame_audit.json", {"frames": evaluated})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    evaluate = commands.add_parser("evaluate")
    for command in (prepare, evaluate):
        for name in ("manifest", "kitti-root", "cache-dir"):
            command.add_argument("--" + name, type=Path, required=True)
    for name in ("dav2-model", "bisenet-repo", "bisenet-weights", "semantic-exporter"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--device", default="auto")
    evaluate.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = _read_json(args.manifest)
    if manifest["version"] != "phase6-holdout-v1" or not manifest["frames"]:
        parser.error("Use a locked nonempty Phase 6 holdout manifest.")
    operation = _prepare if args.command == "prepare" else _evaluate
    operation(args, manifest["frames"])


if __name__ == "__main__":
    main()
