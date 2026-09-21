import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tests.test_lila_training import config, sample, TinyVision
from tests.test_libero_execute_horizon import _FakePolicy
from starVLA.model.framework.WM4A.LiLaWAMTrain import LiLaWAMTrain
from examples.LIBERO.eval_files.model2libero_interface import ModelClient


class BicubicPolicy(_FakePolicy):
    def get_server_metadata(self):
        return dict(action_chunk_size=8, image_resize_resample='bicubic')


class InferenceAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        cfg = config()
        cfg['framework']['action_model']['embodiment_heads']['franka']['gripper_indices'] = [6]
        self.model = LiLaWAMTrain(cfg, vision_encoder=TinyVision()).eval()
        self.raw = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)

    def test_raw_pixels_match_training_packer(self):
        # RGB Image.resize's default is what the real shared loader uses.
        trained = Image.fromarray(self.raw).resize((32, 32))
        mask = torch.tensor([[True, True, False]])
        actual = self.model._pixels([[self.raw, self.raw]], mask)
        expected = self.model._pixels([[trained, trained]], mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @patch('examples.LIBERO.eval_files.model2libero_interface.WebsocketClientPolicy', BicubicPolicy)
    def test_client_uses_model_resize_contract(self):
        client = ModelClient(image_size=(32, 32))
        client.step(dict(image=[self.raw, self.raw], lang='test'), step=0)
        sent = client.client.payloads[0]['examples'][0]['image']
        expected = np.asarray(Image.fromarray(self.raw).resize((32, 32)))
        for image in sent:
            np.testing.assert_array_equal(image, expected)

    def test_final_continuous_actions_clipped_gripper_preserved(self):
        torch.manual_seed(23)
        noise = torch.randn(1, 4, 7)
        self.assertGreater(noise[..., :6].abs().max().item(), 1)
        def zero_velocity(x, *args, **kwargs):
            return torch.zeros_like(x), None, None
        with patch.object(self.model.core, 'forward', side_effect=zero_velocity):
            torch.manual_seed(23)
            output = self.model.predict_action([sample()])['normalized_actions']
        np.testing.assert_array_equal(output[..., :6], noise[..., :6].clamp(-1, 1).numpy())
        np.testing.assert_array_equal(output[..., 6], noise[..., 6].numpy())


if __name__ == '__main__':
    unittest.main()
