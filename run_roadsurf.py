"""Validate the RoadSurf semantic and LiDAR front end on one KITTI Object frame."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core.depth_alignment import (
    DepthAlignmentError,
    backproject_visual_road,
    calibrate_from_road_points,
)
from core.road_export import write_ply
from core.road_lidar_extract import (
    ROAD_ROI_XY,
    estimate_patchwork_mask,
    extract_road_lidar,
    load_patchwork_mask,
    road_interior,
)
from data.dataset_loader import load_kitti_frame
from frontends.run_dav2 import DAV2_SMALL, load_relative_depth, predict_relative_depth
from frontends.semantic_loader import ROAD, SIDEWALK, load_semantic_labels
from visualization.visualize_road_observations import save_depth_previews, save_road_observations


def write_outputs(output: Path, image: Image.Image, labels: np.ndarray, points: np.ndarray,
                  pixels: np.ndarray) -> None:
    output.mkdir(parents=True, exist_ok=False)
    road = labels == ROAD
    sidewalk = labels == SIDEWALK
    np.save(output / "semantic_labels.npy", labels, allow_pickle=False)
    Image.fromarray(road.astype(np.uint8) * 255).save(output / "road_mask.png")
    Image.fromarray(sidewalk.astype(np.uint8) * 255).save(output / "sidewalk_mask.png")

    overlay = np.array(image)
    for mask, color in ((road, (24, 205, 117)), (sidewalk, (244, 80, 187))):
        overlay[mask] = np.rint(0.52 * overlay[mask] + 0.48 * np.array(color))
    Image.fromarray(overlay).save(output / "road_overlay.png")

    projection = image.copy()
    draw = ImageDraw.Draw(projection)
    # Single-pixel marks preserve the exact pixel used to query the mask.
    if len(pixels):
        draw.point([tuple(xy) for xy in pixels.tolist()], fill=(30, 255, 110))
    projection.save(output / "lidar_road_projection.png")

    write_ply(output / "lidar_road.ply", points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", required=True, type=Path)
    parser.add_argument("--split", choices=("training", "testing"), default="training")
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--semantic", required=True, type=Path, help="Semantic file for this frame")
    parser.add_argument("--semantic-format", choices=("auto", "trainId", "labelId", "color", "probabilities"),
                        default="auto")
    parser.add_argument("--semantic-key", help="Array name inside a .npz semantic cache")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--padding", type=int, nargs=4, metavar=("L", "T", "R", "B"),
                        default=(0, 0, 0, 0))
    parser.add_argument("--ground-cache", type=Path, help="Override the frame's Patchwork++ .npz")
    parser.add_argument("--boundary-margin", type=int, default=2)
    depth = parser.add_mutually_exclusive_group()
    depth.add_argument("--dav2-model", nargs="?", const=DAV2_SMALL,
                       help="Run DA-V2 Small using its default HF model or a local snapshot")
    depth.add_argument("--relative-depth", type=Path, help="Reuse a raw DA-V2 HxW float .npy prediction")
    parser.add_argument("--device", default="auto", help="DA-V2 device (auto, cpu, cuda, cuda:0, ...)")
    parser.add_argument("--out", type=Path, help="New output directory")
    args = parser.parse_args()
    if not 0 <= args.frame <= 999999:
        parser.error("--frame must be in [0, 999999].")

    frame = f"{args.frame:06d}"
    split = args.kitti_root / args.split
    output_root = Path("outputs") / "phase2" if args.dav2_model or args.relative_depth else Path("outputs")
    output = args.out or output_root / "kitti_object" / args.split / "64beams" / frame
    if output.exists():
        parser.error(f"Use a new output directory: {output}")
    image_path = split / "image_2" / f"{frame}.png"
    lidar_path = split / "velodyne" / f"{frame}.bin"
    calib_path = split / "calib" / f"{frame}.txt"
    ground_path = args.ground_cache or split / "ground_patchworkpp" / f"{frame}.npz"
    semantic_id = args.semantic.stem
    if args.semantic.name == "image_semantics.npz":
        semantic_id = args.semantic.parent.name
    if semantic_id.isascii() and semantic_id.isdecimal() and int(semantic_id) != args.frame:
        parser.error("The semantic cache's frame ID differs from --frame.")
    if args.relative_depth:
        depth_id = args.relative_depth.stem
        if args.relative_depth.name == "dav2_relative_depth.npy":
            depth_id = args.relative_depth.parent.name
        if depth_id.isascii() and depth_id.isdecimal() and int(depth_id) != args.frame:
            parser.error("The relative-depth cache's frame ID differs from --frame.")
    kitti = load_kitti_frame(args.kitti_root, args.split, args.frame)
    image, points, calibration = kitti.image, kitti.points, kitti.calibration
    labels, details = load_semantic_labels(
        args.semantic, image.size, args.semantic_format, args.semantic_key, args.crop, args.padding)
    if args.ground_cache or ground_path.is_file():
        ground = load_patchwork_mask(ground_path, len(points))
        ground_source = str(ground_path.resolve())
    else:
        ground = estimate_patchwork_mask(points)
        ground_source = "pypatchworkpp"
    road_points, pixels, counts = extract_road_lidar(
        points, calibration, labels, ground, args.boundary_margin)
    metadata = {
        "frame": frame,
        "split": args.split,
        "image_size_wh": list(image.size),
        "image_file": str(image_path.resolve()),
        "semantic_file": str(args.semantic.resolve()),
        "semantic_npz_key": args.semantic_key,
        "semantic": details,
        "road_pixels": int(np.count_nonzero(labels == ROAD)),
        "sidewalk_pixels": int(np.count_nonzero(labels == SIDEWALK)),
        "lidar_file": str(lidar_path.resolve()),
        "calibration_file": str(calib_path.resolve()),
        "ground_method": "Patchwork++",
        "ground_source": ground_source,
        "roi_lidar_xy_m": ROAD_ROI_XY,
        "counts": counts,
    }
    prediction = None
    if args.dav2_model or args.relative_depth:
        if args.relative_depth:
            prediction = load_relative_depth(args.relative_depth, image.size)
            source = {"kind": "saved_relative_depth", "file": str(args.relative_depth.resolve())}
        else:
            prediction, source = predict_relative_depth(image, args.dav2_model, args.device)
        metadata["relative_depth"] = source
        depth_map = np.full(labels.shape, np.nan, dtype=np.float32)
        visual = np.empty((0, 3), dtype=np.float32)
        try:
            fit = calibrate_from_road_points(prediction, road_points, calibration)
            metadata["depth_alignment"] = asdict(fit)
            visual, depth_map = backproject_visual_road(
                prediction, road_interior(labels, args.boundary_margin), fit, calibration)
            if not len(visual):
                raise DepthAlignmentError("No visual road pixels survived depth and LiDAR ROI limits.")
            metadata["mode"] = "lidar_and_visual"
        except DepthAlignmentError as exc:
            metadata.update(mode="lidar_only", depth_alignment_failure=str(exc))
        metadata["counts"]["visual_road_points"] = len(visual)
    write_outputs(output, image, labels, road_points, pixels)
    if prediction is not None:
        np.save(output / "dav2_relative_depth.npy", prediction, allow_pickle=False)
        np.save(output / "visual_road_depth.npy", depth_map, allow_pickle=False)
        write_ply(output / "visual_road.ply", visual)
        save_depth_previews(output, image, prediction, depth_map)
        save_road_observations(output, road_points, visual)
    report = json.dumps(metadata, indent=2)
    (output / "metadata.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
