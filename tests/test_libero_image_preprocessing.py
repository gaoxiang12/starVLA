"""Keep the LIBERO client RGB resize aligned with GAWM training inputs."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from starVLA.model.framework.WM4A.GAWM import GAWM


@pytest.mark.parametrize("data_cfg, expected", [
    ({}, "bicubic"),
    ({"image_resize_resample": "bilinear"}, "bicubic"),
    ({"packed_image_size": [224, 224]}, "bicubic"),
    ({"packed_image_size": [224, 224], "image_resize_resample": "bilinear"}, "bilinear"),
])
def test_resize_metadata_follows_training_pack_path(data_cfg, expected):
    model = SimpleNamespace(config=SimpleNamespace(
        datasets=SimpleNamespace(vla_data=data_cfg)))
    assert GAWM.image_resize_resample.fget(model) == expected


def test_client_sends_same_pixels_as_training_rgb_resize():
    model = SimpleNamespace(config=SimpleNamespace(
        datasets=SimpleNamespace(vla_data={})))

    class Policy:
        def __init__(self, *args):
            self.payload = None

        def get_server_metadata(self):
            return {"action_chunk_size": 8,
                    "image_resize_resample": GAWM.image_resize_resample.fget(model)}

        def predict_action(self, payload):
            self.payload = payload
            return {"data": {"actions": np.zeros((1, 8, 7))}}

    frames = list(np.random.default_rng(42).integers(
        0, 256, size=(2, 256, 256, 3), dtype=np.uint8))
    with patch("examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy", Policy):
        client = ModelClient()
        client.step({"image": frames, "lang": "test"}, step=0)
    actual = client.client.payload["examples"][0]["image"]
    for raw, resized in zip(frames, actual):
        expected = np.asarray(Image.fromarray(raw).resize((224, 224)))
        np.testing.assert_array_equal(resized, expected)
