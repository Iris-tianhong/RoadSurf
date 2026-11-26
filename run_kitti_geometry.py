"""Export the original LiDAR projection and sparse camera depth for one KITTI Object frame."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from data.dataset_loader import load_kitti_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "configs" / "kitti.yaml")
    parser.add_argument("--kitti-root", type=Path, help="Override kitti_root in the configuration")
    parser.add_argument("--split", choices=("training", "testing"))
    parser.add_argument("--frame", type=int, help="Default: first frame with all three inputs")
    parser.add_argument("--out", type=Path, help="New output directory")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = (args.kitti_root or args.config.parent / config["kitti_root"]).resolve()
    split = args.split or config["split"]
    frame = load_kitti_frame(root, split, args.frame)

    calibration = frame.calibration
    depth_map = calibration.create_sparse_depth_map(frame.points, frame.image.size)
    valid = np.isfinite(depth_map)
    depth = depth_map[valid]
    if not depth.size:
        raise ValueError(f"No valid LiDAR projections in frame {frame.frame_id}.")
    # 色彩只用于显示。最近点偏暖，远点偏冷。
    t = np.clip(depth / np.percentile(depth, 98), 0, 1)
    colors = np.rint(np.column_stack((255 - 215 * t, 125 + 105 * t, 35 + 220 * t))).astype(np.uint8)
    overlay = np.array(frame.image)
    overlay[valid] = colors
    preview = np.zeros_like(overlay)
    preview[valid] = colors

    uv, optical_depth = calibration.project_lidar_to_image(frame.points)
    width, height = frame.image.size
    in_image = ((uv >= 0) & (uv < [width, height])).all(axis=1)
    xyz = calibration.backproject_image_to_lidar(uv[in_image], optical_depth[in_image])
    error = float(np.linalg.norm(xyz - frame.points[in_image, :3], axis=1).max())

    output = args.out or Path("outputs") / "kitti_geometry" / split / frame.frame_id
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(overlay).save(output / "lidar_projection.png")
    Image.fromarray(preview).save(output / "sparse_depth.png")
    np.save(output / "sparse_depth.npy", depth_map, allow_pickle=False)
    report = json.dumps({
        "kitti_root": str(root),
        "split": split,
        "frame": frame.frame_id,
        "image_size_wh": [width, height],
        "raw_points": len(frame.points),
        "projected_points_in_image": int(in_image.sum()),
        "valid_depth_pixels": len(depth),
        "depth_range_m": [float(depth.min()), float(depth.max())],
        "round_trip_max_error_m": error,
    }, indent=2)
    (output / "metadata.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
