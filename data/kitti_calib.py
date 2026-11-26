"""KITTI Object camera 2 calibration in the original RGB image grid."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class KittiCalibration:
    def __init__(self, path: Path, image_transform: Optional[np.ndarray] = None):
        """Read calibration, optionally mapping raw UV to a cropped/resized image grid."""
        matrices = {}
        sizes = {"P2": 12, "R0_rect": 9, "R_rect": 9, "Tr_velo_to_cam": 12, "Tr_velo_cam": 12}
        for line in path.read_text().splitlines():
            key, sep, values = line.partition(":")
            key = key.strip()
            if sep and key in sizes:
                try:
                    matrix = np.asarray(values.split(), dtype=np.float64)
                except ValueError as exc:
                    raise ValueError(f"Invalid KITTI matrix {key}: {path}") from exc
                if matrix.size != sizes[key] or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid KITTI matrix {key}: {path}")
                matrices[key] = matrix
        rect = matrices.get("R0_rect", matrices.get("R_rect"))
        velo = matrices.get("Tr_velo_to_cam", matrices.get("Tr_velo_cam"))
        if "P2" not in matrices or rect is None or velo is None:
            raise ValueError(f"Incomplete KITTI calibration: {path}")
        self.velo_to_rect = rect.reshape(3, 3) @ velo.reshape(3, 4)
        p2 = matrices["P2"].reshape(3, 4)
        scale = np.linalg.norm(p2[2, :3])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid KITTI calibration: {path}")
        self.projection = p2 / scale @ np.vstack((self.velo_to_rect, [0, 0, 0, 1]))
        if image_transform is not None:
            transform = np.asarray(image_transform, dtype=np.float64)
            if (transform.shape != (3, 3) or not np.isfinite(transform).all()
                    or not np.array_equal(transform[2], [0, 0, 1])):
                raise ValueError("Expected a 3x3 affine transform from raw UV to output UV.")
            self.projection = transform @ self.projection
        try:
            self.inverse = np.linalg.inv(self.projection[:, :3])
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Singular KITTI projection: {path}") from exc
        self.camera_center = -self.inverse @ self.projection[:, 3]

    def project_lidar_to_image(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map Nx3/Nx4 LiDAR rows to float64 UV and camera-2 optical depth (m)."""
        xyz = np.asarray(points[:, :3], dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(xyz).all(axis=1))
        valid_xyz = xyz[finite]
        projected = valid_xyz @ self.projection[:, :3].T + self.projection[:, 3]
        rect_z = valid_xyz @ self.velo_to_rect[2, :3] + self.velo_to_rect[2, 3]
        uv = np.full((len(xyz), 2), np.nan)
        depth = np.full(len(xyz), np.nan)
        depth[finite] = projected[:, 2]
        front = (projected[:, 2] > 0) & (rect_z > 0)
        uv[finite[front]] = projected[front, :2] / projected[front, 2, None]
        return uv, depth

    def pixel_to_lidar_ray(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the camera-2 center and rays parametrized by optical depth."""
        uv = np.asarray(uv, dtype=np.float64)
        pixels = np.concatenate((uv, np.ones((*uv.shape[:-1], 1))), axis=-1)
        return self.camera_center.copy(), pixels @ self.inverse.T

    def backproject_image_to_lidar(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        origin, rays = self.pixel_to_lidar_ray(uv)
        return origin + np.asarray(depth)[..., None] * rays

    def visible_pixels(
        self, points: np.ndarray, image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest valid LiDAR return per pixel, using all supplied points."""
        width, height = image_size
        uv, depth = self.project_lidar_to_image(points)
        ids = np.flatnonzero(
            (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        xy = np.floor(uv[ids]).astype(np.intp)
        flat_pixels = xy[:, 1] * width + xy[:, 0]
        order = np.lexsort((depth[ids], flat_pixels))
        _, first = np.unique(flat_pixels[order], return_index=True)
        winners = np.sort(order[first])
        return ids[winners], xy[winners], depth[ids[winners]]

    def create_sparse_depth_map(
        self, points: np.ndarray, image_size: Tuple[int, int]
    ) -> np.ndarray:
        _, pixels, depth = self.visible_pixels(points, image_size)
        result = np.full(image_size[::-1], np.nan, dtype=np.float64)
        u, v = pixels.T
        result[v, u] = depth
        return result
