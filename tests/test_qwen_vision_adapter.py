import unittest

import torch

from starVLA.model.modules.world_model.QwenVision import _QwenVision_Interface


class QwenVisionAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = object.__new__(_QwenVision_Interface)
        torch.nn.Module.__init__(self.adapter)
        self.adapter.merge_size = 2
        self.adapter.feature_dim = 8

    def test_reshape_merged_tokens_preserves_image_order(self):
        image_embeds = torch.arange(2 * 64 * 8, dtype=torch.float32).reshape(
            2 * 64, 8
        )
        grid_thw = torch.tensor([[1, 16, 16], [1, 16, 16]])

        tokens = self.adapter._reshape_tokens(
            image_embeds,
            grid_thw,
            batch_size=1,
            num_frames=1,
            num_views=2,
        )

        self.assertEqual(tokens.shape, (1, 1, 2, 64, 8))
        self.assertTrue(torch.equal(tokens[0, 0, 0], image_embeds[:64]))
        self.assertTrue(torch.equal(tokens[0, 0, 1], image_embeds[64:]))

    def test_reshape_rejects_non_square_grid(self):
        image_embeds = torch.zeros(48, 8)
        grid_thw = torch.tensor([[1, 12, 16]])

        with self.assertRaisesRegex(ValueError, "square grid"):
            self.adapter._reshape_tokens(
                image_embeds,
                grid_thw,
                batch_size=1,
                num_frames=1,
                num_views=1,
            )

    def test_flatten_frames_validates_view_count(self):
        with self.assertRaisesRegex(ValueError, "same number of views"):
            self.adapter._flatten_frames([[["a", "b"], ["c"]]])


if __name__ == "__main__":
    unittest.main()
