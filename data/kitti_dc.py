"""Match KITTI Object input frames to KITTI Raw/DC reference maps."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def read_dc_depth(path: Path) -> np.ndarray:
    """Read uint16 depth/256 in metres, with zero encoded as NaN."""
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in ("I", "I;16", "I;16B", "I;16L"):
            raise ValueError(f"Expected a 16-bit KITTI DC PNG: {path}")
        values = np.asarray(image)
    if values.min() < 0 or values.max() > 65535:
        raise ValueError(f"Invalid KITTI DC depth code: {path}")
    depth = values.astype(np.float64) / 256.0
    depth[values == 0] = np.nan
    return depth


def _archive_file(archive: ZipFile, suffix: str) -> bytes:
    matches = [name for name in archive.namelist() if name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} in {archive.filename}.")
    return archive.read(matches[0])


def read_object_mapping(devkit: Path):
    with ZipFile(devkit) as archive:
        rows = _archive_file(archive, "train_mapping.txt")
        order = _archive_file(archive, "train_rand.txt")
    records = rows.decode("ascii").splitlines()
    indices = [int(value) for value in order.decode("ascii").strip().split(",")]
    if sorted(indices) != list(range(1, len(records) + 1)):
        raise ValueError("KITTI Object train_rand must be a 1-based permutation.")
    mapping = {f"{id_:06d}": tuple(records[idx - 1].split()[1:]) for id_, idx in enumerate(indices)}
    return mapping, {"object_mapping": hashlib.sha256(rows).hexdigest(),
                     "object_permutation": hashlib.sha256(order).hexdigest()}


def make_manifest(root: Path, object_devkit: Path, scene_flow_devkit: Path, development_ids):
    mapping, hashes = read_object_mapping(object_devkit)
    with ZipFile(scene_flow_devkit) as archive:
        rows = _archive_file(archive, "train_mapping.txt")
    hashes["scene_flow_mapping"] = hashlib.sha256(rows).hexdigest()
    # Also exclude mapped scenes appearing in the semantic validation images.
    semantic_drives = {row.split()[1] for row in rows.decode("ascii").splitlines() if row.strip()}
    dev_drives = {mapping[id_][0] for id_ in development_ids}
    pools = defaultdict(dict)
    for frame, raw in mapping.items():
        if not raw:
            continue
        drive, raw_frame = raw
        if drive in semantic_drives or drive in dev_drives:
            continue
        ref = Path("data_depth_annotated") / "val" / drive / "proj_depth/groundtruth/image_02"
        ref /= f"{raw_frame}.png"
        rgb = root / "training/image_2" / f"{frame}.png"
        if not (root / ref).is_file() or not rgb.is_file():
            continue
        with Image.open(root / ref) as depth_img, Image.open(rgb) as rgb_img:
            if depth_img.size != rgb_img.size:
                continue
        pools[drive].setdefault(raw_frame, {
            "frame": frame, "drive": drive, "raw_frame": raw_frame, "depth_png": ref.as_posix(),
        })

    records = []
    for drive, pool in sorted(pools.items()):
        ordered = [pool[key] for key in sorted(pool)]
        if len(ordered) >= 10:
            records.extend(ordered[(len(ordered) - 1) * quartile // 4] for quartile in (1, 2, 3))
    if not records:
        raise ValueError("No eligible disjoint KITTI DC validation drives.")
    return {
        "source": "KITTI DC val / groundtruth / image_02; Object training inputs matched via Raw IDs",
        "selection": "Three quartile frames per drive with >=10 distinct matched Raw frames.",
        "mapping_sha256": hashes,
        "excluded_phase3_drives": sorted(dev_drives),
        "excluded_known_bisenet_drives": sorted(semantic_drives),
        "eligible_counts": {drive: len(pool) for drive, pool in sorted(pools.items())},
        "frames": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kitti-root", "object-devkit", "scene-flow-devkit", "phase3-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    development = [p.parent.name for p in args.phase3_dir.glob("*/metadata.json")]
    if not development:
        parser.error("--phase3-dir must contain the development frame directories.")
    manifest = make_manifest(args.kitti_root, args.object_devkit, args.scene_flow_devkit, development)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print("Evaluation frames:", " ".join(record["frame"] for record in manifest["frames"]))


if __name__ == "__main__":
    main()
