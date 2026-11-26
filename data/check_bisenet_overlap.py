"""Compare held-out KITTI RGB with the inputs from the upstream BiSeNet fine-tuning split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from data.check_vkitti_overlap import _RawZipRanges
from data.kitti_dc import _archive_file, read_object_mapping
from data.phase6_holdout import file_hash


URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip"


def _rgb_signature(image):
    rgb = image.convert("RGB")
    small = rgb.resize((64, 20), Image.Resampling.BILINEAR)
    return sha256(rgb.tobytes()).hexdigest(), np.asarray(small, dtype=np.float64)


def _training_signatures(start):
    source = _RawZipRanges(URL)
    signatures = {}
    with ZipFile(source) as archive:
        for first in range(start, start + 50, 5):
            group = [archive.getinfo(f"training/image_2/{i:06d}_10.png") for i in range(first, first + 5)]
            end = group[-1].header_offset + group[-1].compress_size + 1024
            if end - group[0].header_offset < 4 * 1024 * 1024:
                source.prefetch(group[0].header_offset, min(source.size, end))
            for i, entry in enumerate(group, start=first):
                with Image.open(BytesIO(archive.read(entry))) as image:
                    signatures[i] = _rgb_signature(image)
    return signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kitti-root", "manifest", "devkits", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mapping, _ = read_object_mapping(args.devkits / "devkit_object.zip")
    with ZipFile(args.devkits / "devkit_scene_flow.zip") as archive:
        rows = _archive_file(archive, "train_mapping.txt")
    if sha256(rows).hexdigest() != manifest["mapping_sha256"]["scene_flow_mapping"]:
        raise ValueError("Scene Flow mapping differs from the lock.")
    by_raw = {raw: frame for frame, raw in mapping.items() if raw}
    pairs = [(i, by_raw[raw]) for i, row in enumerate(rows.decode("ascii").splitlines())
             if (raw := tuple(row.split()[1:])) in by_raw]
    training = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(_training_signatures, (0, 50, 100, 150)):
            training.update(result)
    controls = []
    for index, frame in pairs[:4]:
        with Image.open(args.kitti_root / "training/image_2" / f"{frame}.png") as image:
            rgb, _ = _rgb_signature(image)
        controls.append({"semantic_id": index, "object_frame": frame,
                         "same_decoded_rgb": rgb == training[index][0]})
    if not any(control["same_decoded_rgb"] for control in controls):
        raise ValueError("Could not detect a known Object/semantic image overlap.")
    results = []
    for record in manifest["frames"]:
        id_ = record["frame"]
        with Image.open(args.kitti_root / "training/image_2" / f"{id_}.png") as image:
            rgb, small = _rgb_signature(image)
        if rgb != record["input_sha256"]["rgb"]:
            raise ValueError(f"Holdout RGB changed for {id_}.")
        exact = [i for i, (digest, _) in training.items() if digest == rgb]
        nearest = min((float(np.mean(np.abs(small - signature))), i)
                      for i, (_, signature) in training.items())
        results.append({"frame": id_, "same_decoded_rgb_semantic_ids": exact,
                        "closest_at_64x20_semantic_id": nearest[1], "rgb_mae_0_to_255": nearest[0]})
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"archive": URL, "semantic_input_count": len(training),
                   "manifest_sha256": file_hash(args.manifest), "positive_controls": controls,
                   "semantic_rgb_sha256": {i: h for i, (h, _) in training.items()},
                   "frames": results}, stream, indent=2)
    if any(item["same_decoded_rgb_semantic_ids"] for item in results):
        raise ValueError("At least one held-out RGB was used by the semantic training/validation data.")
    print(f"Compared {len(results)} held-out RGBs with all {len(training)} BiSeNet inputs; no exact duplicates.")


if __name__ == "__main__":
    main()
