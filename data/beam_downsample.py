"""Recover KITTI scan rings from the order of raw Velodyne returns."""

import numpy as np


def recover_rings(points: np.ndarray):
    """Require 64 ordered, near-full azimuth sweeps with decreasing elevation."""
    xyz = points[:, :3].astype(np.float64, copy=False)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Ring recovery requires a finite, complete raw KITTI scan.")
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360
    splits = np.r_[0, np.flatnonzero(np.diff(azimuth) < -180) + 1, len(xyz)]
    counts = np.diff(splits)
    if len(counts) != 64 or counts.min() < 64:
        raise ValueError(f"Expected 64 azimuth sweeps; found {len(counts)}.")
    span = azimuth[splits[1:] - 1] - azimuth[splits[:-1]]
    steps = np.diff(azimuth)
    if np.min(span) < 250 or np.count_nonzero((steps < -1) & (steps >= -180)) > len(xyz) / 1000:
        raise ValueError("Point order does not follow 64 near-monotonic azimuth sweeps.")
    radii = np.hypot(xyz[:, 0], xyz[:, 1])
    centers, spreads = [], []
    for i, j in zip(splits[:-1], splits[1:]):
        radius, z = radii[i:j], xyz[i:j, 2]
        # HDL-64 laser origins differ vertically. Estimate each sweep's origin
        # instead of assigning rings by a spherical nearest-angle lookup.
        params, _, rank, _ = np.linalg.lstsq(np.c_[radius, np.ones(j - i)], z, rcond=None)
        if rank != 2:
            raise ValueError("A recovered sweep has no radial span for verification.")
        angle = np.degrees(np.arctan(params[0]))
        centers.append(angle)
        spreads.append(np.median(np.abs(np.degrees(np.arctan2(z - params[1], radius)) - angle)))
    centers = np.asarray(centers)
    if np.any(np.diff(centers) >= 0) or not 20 <= centers[0] - centers[-1] <= 30:
        raise ValueError("Recovered ring elevations do not follow the KITTI HDL-64 scan order.")
    if max(spreads) > .08:
        raise ValueError("A recovered sweep is inconsistent with a single laser elevation.")
    return np.repeat(np.arange(64, dtype=np.uint8), counts), centers, span, np.asarray(spreads)


def beam_mask(rings: np.ndarray, beams: int) -> np.ndarray:
    if beams not in (64, 32, 16):
        raise ValueError("KITTI simulated beam count must be 64, 32, or 16.")
    return rings % (64 // beams) == 0
