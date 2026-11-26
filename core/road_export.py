"""Binary PLY used by the road observation and surface stages."""

from pathlib import Path

import numpy as np


def write_ply(path: Path, points: np.ndarray) -> None:
    fields = ("x", "y", "z", "intensity") if points.shape[1] == 4 else ("x", "y", "z")
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(points)}\n"
              + "".join(f"property float {field}\n" for field in fields) + "end_header\n")
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(points.astype("<f4", copy=False).tobytes())


def read_road_ply(path: Path) -> np.ndarray:
    header, separator, data = path.read_bytes().partition(b"end_header\n")
    lines = header.decode("ascii").splitlines()
    if (not separator or len(lines) not in (6, 7)
            or lines[:2] != ["ply", "format binary_little_endian 1.0"]
            or lines[3:6] != ["property float x", "property float y", "property float z"]
            or (len(lines) == 7 and lines[6] != "property float intensity")):
        raise ValueError(f"Unsupported RoadSurf PLY header: {path}")
    prefix = "element vertex "
    count = int(lines[2][len(prefix):]) if lines[2].startswith(prefix) else -1
    columns = len(lines) - 3
    if count < 0 or len(data) != count * columns * 4:
        raise ValueError(f"PLY vertex count does not match its payload: {path}")
    return np.frombuffer(data, dtype="<f4").reshape(count, columns)
