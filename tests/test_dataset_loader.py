import numpy as np
from PIL import Image
import pytest

from data.dataset_loader import list_kitti_frames, load_kitti_frame


def test_selects_the_first_matched_frame_and_preserves_raw_input(tmp_path):
    base = tmp_path / "training"
    for directory in ("image_2", "velodyne", "calib"):
        (base / directory).mkdir(parents=True)
    Image.new("RGB", (13, 7), (12, 35, 97)).save(base / "image_2" / "000000.png")
    Image.new("RGB", (13, 7), (4, 8, 16)).save(base / "image_2" / "000002.png")
    points = np.array([[0.1, 0.2, 1, .75], [3.0, 0.0, -2, .125]], dtype="<f4")
    lidar_path = base / "velodyne" / "000002.bin"
    points.tofile(lidar_path)
    (base / "calib" / "000002.txt").write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )

    assert list_kitti_frames(tmp_path, "training") == ["000002"]
    frame = load_kitti_frame(tmp_path, "training")
    assert frame.frame_id == "000002"
    assert frame.image.size == (13, 7)
    assert frame.image.getpixel((0, 0)) == (4, 8, 16)
    assert frame.points.dtype == np.float32
    np.testing.assert_array_equal(frame.points, points)
    with pytest.raises(FileNotFoundError):
        load_kitti_frame(tmp_path, "training", 0)
    lidar_path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="four float32 columns"):
        load_kitti_frame(tmp_path, "training", 2)
