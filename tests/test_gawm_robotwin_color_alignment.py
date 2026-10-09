"""Regression for RoboTwin RGB arrays encoded directly by OpenCV upstream."""
import cv2
import numpy as np
import pytest

from examples.Robotwin.eval_files.gawm_hdf5_server import preprocess_policy_image
from starVLA.dataloader.robotwin_official_hdf5 import RoboTwinOfficialDataset


def test_matches_historical_training_with_upstream_jpeg_convention():
    rgb = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    # Upstream images_encoding passes SAPIEN RGB directly to imencode.
    encoded = cv2.imencode('.jpg', rgb)[1]
    decoded_physical_rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    original = decoded_physical_rgb.copy()
    loader = object.__new__(RoboTwinOfficialDataset)
    loader.image_size = (320, 240)
    loader.image_channel_order = "bgr"  # Explicit historical training convention
    expected = loader._image(encoded).numpy()
    actual = preprocess_policy_image(decoded_physical_rgb, (320, 240), 'bgr')
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(decoded_physical_rgb, original)
    assert not np.array_equal(preprocess_policy_image(decoded_physical_rgb, (320, 240), 'rgb'), expected)


def test_red_is_swapped_before_channel_specific_normalization():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    actual = preprocess_policy_image(rgb, (32, 24), 'bgr')
    expected = (np.array([0., 0., 1.], dtype=np.float32) - np.array([.485,.456,.406], dtype=np.float32)) / np.array([.229,.224,.225], dtype=np.float32)
    np.testing.assert_array_equal(actual[:, 0, 0], expected)
    assert (rgb[..., 0] == 255).all()


def test_rejects_unknown_color_convention():
    with pytest.raises(ValueError, match='channel order'):
        preprocess_policy_image(np.zeros((2, 2, 3), np.uint8), (2, 2), 'guess')


def test_new_training_keeps_physical_rgb_and_matches_online_rgb():
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    encoded = cv2.imencode('.jpg', rgb)[1]
    decoded_rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    loader = object.__new__(RoboTwinOfficialDataset)
    loader.image_size = (320, 240)
    loader.image_channel_order = 'rgb'
    pixels = loader._image(encoded).numpy()
    np.testing.assert_array_equal(pixels, preprocess_policy_image(decoded_rgb, (320, 240), 'rgb'))
    restored = (pixels[:, 0, 0]*np.array([.229,.224,.225]) + np.array([.485,.456,.406]))*255
    np.testing.assert_allclose(restored, [255, 0, 0], atol=2)
