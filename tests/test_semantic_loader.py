import numpy as np
import pytest
from PIL import Image

from frontends.semantic_loader import load_semantic_labels


def test_same_value_has_different_semantics_in_index_formats(tmp_path):
    image = tmp_path / "index.png"
    Image.fromarray(np.array([[0, 1, 7, 8, 13, 18, 255]], dtype=np.uint8)).save(image)
    train, _ = load_semantic_labels(image, (7, 1), "trainId")
    labels, _ = load_semantic_labels(image, (7, 1), "labelId")
    np.testing.assert_array_equal(train, [[0, 1, 7, 8, 13, 18, 255]])
    np.testing.assert_array_equal(labels, [[255, 255, 0, 1, 4, 255, 255]])
    with pytest.raises(ValueError, match="Ambiguous"):
        load_semantic_labels(image, (7, 1))


def test_color_png_distinguishes_road_and_sidewalk_and_rejects_mixed_rgb(tmp_path):
    path = tmp_path / "palette.png"
    colors = np.array([[(128, 64, 128), (244, 35, 232), (70, 70, 70)]], dtype=np.uint8)
    Image.fromarray(colors).save(path)
    labels, meta = load_semantic_labels(path, (3, 1))
    np.testing.assert_array_equal(labels, [[0, 1, 2]])
    assert meta["format"] == "color"
    colors[0, 0] = [129, 65, 130]
    Image.fromarray(colors).save(path)
    with pytest.raises(ValueError, match="outside the Cityscapes palette"):
        load_semantic_labels(path, (3, 1))


def test_unpadding_and_nearest_resize_restore_the_crop(tmp_path):
    path = tmp_path / "cache.npz"
    padded = np.pad(np.array([[0, 1], [13, 18]], dtype=np.uint8),
                    ((2, 1), (1, 3)), constant_values=200)
    np.savez_compressed(path, classes=padded)
    labels, details = load_semantic_labels(
        path, (8, 7), "trainId", "classes", crop=(2, 1, 4, 4), padding=(1, 2, 3, 1))
    expected = np.full((7, 8), 255, dtype=np.uint8)
    expected[1:5, 2:6] = [[0, 0, 1, 1], [0, 0, 1, 1], [13, 13, 18, 18], [13, 13, 18, 18]]
    np.testing.assert_array_equal(labels, expected)
    assert details["size_wh"] == [6, 5]


def test_saved_server_probabilities_use_cityscapes_argmax(tmp_path):
    path = tmp_path / "000486.npy"
    probabilities = np.zeros((1, 3, 19), dtype=np.float16)
    probabilities[0, [0, 1, 2], [0, 1, 13]] = 1
    np.save(path, probabilities)
    labels, details = load_semantic_labels(path, (3, 1), "probabilities")
    np.testing.assert_array_equal(labels, [[0, 1, 13]])
    assert details["dtype"] == "float16"
    probabilities[0, 0, 1] = 1
    np.save(path, probabilities)
    with pytest.raises(ValueError, match="Invalid Cityscapes"):
        load_semantic_labels(path, (3, 1), "probabilities")
