"""Reconstruct a Phase 2 frame with Phase 5 SGSR and split visible/inferred output."""

import argparse
import json
from pathlib import Path

import numpy as np

from core.road_export import read_road_ply
from core.road_lidar_extract import road_roi_mask
from core.sgsr import reconstruct
from core.sgsr_export import export_reconstruction
from core.sgsr_solver import make_spline_basis
from data.dataset_loader import load_kitti_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase2-dir", "kitti-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory.")
    meta = json.loads((args.phase2_dir / "metadata.json").read_text(encoding="utf-8"))
    frame = load_kitti_frame(args.kitti_root, meta["split"], int(meta["frame"]))
    labels = np.load(args.phase2_dir / "semantic_labels.npy", allow_pickle=False)
    if labels.shape != frame.image.size[::-1]:
        raise ValueError("Phase 2 labels and KITTI RGB must share the original image grid.")
    lidar = read_road_ply(args.phase2_dir / "lidar_road.ply")[:, :3].astype(np.float64)
    visual = np.empty((0, 3), dtype=np.float64)
    if meta.get("relative_depth") or (args.phase2_dir / "visual_road.ply").exists():
        visual = read_road_ply(args.phase2_dir / "visual_road.ply")[:, :3].astype(np.float64)
    for points in (lidar, visual):
        if not np.isfinite(points).all() or not road_roi_mask(points).all():
            raise ValueError("Phase 2 point clouds must contain finite XYZ inside the LiDAR ROI.")
    sparse = frame.calibration.create_sparse_depth_map(frame.points, frame.image.size)
    result = reconstruct(lidar, visual, frame.calibration, labels, sparse,
                         make_spline_basis(4), make_spline_basis(2),
                         boundary_margin=meta["counts"]["semantic_boundary_margin_px"])
    info = export_reconstruction(args.out, result, frame.image, frame.calibration, {
        "frame": meta["frame"], "split": meta["split"], "beams": 64,
        "phase2_dir": str(args.phase2_dir.resolve()), "gap_kind": "observed_sensor_occlusion",
    })
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
