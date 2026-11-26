"""Lock a previously unused KITTI DC validation cohort and audit known scene overlap."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from data.check_vkitti_overlap import TRACKING_SEEDS
from data.dataset_loader import list_kitti_frames
from data.kitti_dc import _archive_file, read_object_mapping


def file_hash(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(root, frame):
    base = root / "training"
    with Image.open(base / "image_2" / f"{frame}.png") as image:
        rgb = sha256(image.convert("RGB").tobytes()).hexdigest()
    return {"rgb": rgb, "velodyne": file_hash(base / "velodyne" / f"{frame}.bin"),
            "calibration": file_hash(base / "calib" / f"{frame}.txt")}


def _detection_overlap(root, mapping, semantic_split):
    results = {}
    for split in ("train", "val", "test"):
        path = root / "ImageSets" / f"{split}.txt"
        if not path.is_file():
            continue
        frames = path.read_text(encoding="ascii").split()
        results[split] = {"frames": len(frames), "ids_sha256": file_hash(path)}
        if split == "test":
            # Object test uses its own frame number namespace.
            results[split]["raw_mapping_available"] = False
            continue
        raw = [mapping.get(i) for i in frames]
        results[split]["mapped_frames"] = sum(bool(r) for r in raw)
        for name, pairs in semantic_split.items():
            known = set(filter(None, pairs))
            drives = {r[0] for r in known}
            results[split][name] = {
                "same_frame": sum(r in known for r in raw),
                "same_drive": sum(bool(r) and r[0] in drives for r in raw),
            }
    return results


def make_holdout(kitti, devkits, previous, vkitti_path):
    phase4_path = previous / "phase4/manifest.json"
    phase4 = json.loads(phase4_path.read_text(encoding="utf-8"))
    mapping, hashes = read_object_mapping(devkits / "devkit_object.zip")
    with ZipFile(devkits / "devkit_scene_flow.zip") as archive:
        rows = _archive_file(archive, "train_mapping.txt")
    hashes["scene_flow_mapping"] = sha256(rows).hexdigest()
    if phase4["mapping_sha256"] != hashes:
        raise ValueError("Official mapping rows differ from the Phase 4 split.")
    semantic = [tuple(row.split()[1:]) for row in rows.decode("ascii").splitlines()]
    if len(semantic) != 200:
        raise ValueError("Expected 200 rows from the semantic training/validation source.")
    semantic_split = {"bisenet_train": semantic[:180], "bisenet_validation": semantic[180:],
                      "bisenet_all": semantic}
    semantic_drives = {row[0] for row in semantic if row}

    development = {p.parent.name for name in ("kitti_object", "phase2", "phase3")
                   for p in (previous / name).rglob("metadata.json") if p.parent.name.isdecimal()}
    if not development or any(not mapping.get(frame) for frame in development):
        raise ValueError("Missing Raw identity for a RoadSurf development frame.")
    development_drives = {mapping[frame][0] for frame in development}
    phase4_ids = {record["frame"] for record in phase4["frames"]}
    phase4_drives = {record["drive"] for record in phase4["frames"]}
    for record in phase4["frames"]:
        if mapping[record["frame"]] != (record["drive"], record["raw_frame"]):
            raise ValueError("Phase 4 manifest conflicts with the official Object mapping.")

    depth_root = kitti / "data_depth_annotated"
    dc_drives = {split: {p.name for p in (depth_root / split).glob("*_sync") if p.is_dir()}
                 for split in ("train", "val")}
    if not dc_drives["train"] or not dc_drives["val"]:
        raise ValueError("Expected both installed KITTI DC partitions for split auditing.")
    candidate = dc_drives["val"] - dc_drives["train"] - semantic_drives - development_drives - phase4_drives
    candidates = []
    for frame in list_kitti_frames(kitti, "training"):
        raw = mapping.get(frame)
        if not raw or raw[0] not in candidate:
            continue
        drive, raw_frame = raw
        png = Path("data_depth_annotated/val") / drive / "proj_depth/groundtruth/image_02" / f"{raw_frame}.png"
        if (kitti / png).is_file():
            candidates.append((frame, drive, raw_frame, png))
    matched_drives = {drive for _, drive, _, _ in candidates}
    vkitti = json.loads(vkitti_path.read_text(encoding="utf-8"))
    checked = vkitti["checked_raw_drives"]
    if (tuple(vkitti["tracking_seeds"]) != TRACKING_SEEDS
            or not (matched_drives | phase4_drives) <= checked.keys()
            or any(checked[drive]["raw_pose_count"] < 3 for drive in matched_drives | phase4_drives)):
        raise ValueError("Incomplete GPS/IMU check against Virtual KITTI tracking seeds.")
    vkitti_overlap = {drive for drive, check in checked.items() if check["tracking_seed_matching_poses"]}

    known_hashes = {field: set() for field in ("rgb", "velodyne")}
    for frame in sorted(development | phase4_ids):
        h = input_hashes(kitti, frame)
        for field in known_hashes:
            known_hashes[field].add(h[field])
    records, same_payload, raw_ids = [], [], set()
    for frame, drive, raw_frame, png in candidates:
        if drive in vkitti_overlap:
            continue
        hashes_in = input_hashes(kitti, frame)
        if any(hashes_in[field] in values for field, values in known_hashes.items()):
            same_payload.append(frame)
            continue
        if (drive, raw_frame) in raw_ids:
            raise ValueError(f"Duplicate matched Raw frame: {drive}/{raw_frame}")
        raw_ids.add((drive, raw_frame))
        records.append({"frame": frame, "drive": drive, "raw_frame": raw_frame,
                        "depth_png": png.as_posix(), "input_sha256": hashes_in})
        for field in known_hashes:
            known_hashes[field].add(hashes_in[field])
    if not records:
        raise ValueError("There are no remaining local, independent DC validation input frames.")
    records.sort(key=lambda r: (r["drive"], r["raw_frame"]))
    return {
        "version": "phase6-holdout-v1", "dc_partition": "val",
        "selection": "All complete Object/Raw/DC matches from permitted validation drives; no metric or reference depth used.",
        "mapping_sha256": hashes,
        "phase4_manifest_sha256": file_hash(phase4_path),
        "excluded_project": {"development_ids": sorted(development), "development_drives": sorted(development_drives),
                             "phase4_and_phase5_ids": sorted(phase4_ids), "phase4_and_phase5_drives": sorted(phase4_drives)},
        "bisenet_mapping": {name: {"mapped": sum(bool(r) for r in entries),
                                   "unmapped": sum(not r for r in entries)} for name, entries in semantic_split.items()},
        "excluded_known_bisenet_drives": sorted(semantic_drives),
        "dc_train_val_drive_overlap": sorted(dc_drives["train"] & dc_drives["val"]),
        "unused_dc_val_drives_without_object_matches": sorted(candidate - matched_drives),
        "vkitti_audit_sha256": file_hash(vkitti_path), "vkitti": vkitti,
        "excluded_duplicate_rgb_or_velodyne_ids": same_payload,
        "detection_overlap": _detection_overlap(kitti, mapping, semantic_split),
        "holdout_counts": dict(Counter(record["drive"] for record in records)), "frames": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kitti-root", "devkits", "previous", "vkitti-audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = make_holdout(args.kitti_root, args.devkits, args.previous, args.vkitti_audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["holdout_counts"], indent=2))


if __name__ == "__main__":
    main()
