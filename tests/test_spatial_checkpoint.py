import unittest
import torch
from starVLA.model.modules.spatial_checkpoint import resize_spatial_checkpoint


def source():
    spatial = torch.cat([torch.full((16, 6), float(i)) for i in range(3)])
    return {
        'visual_token_pooler.row_embedding.weight': torch.arange(4.).view(4, 1).expand(-1, 6),
        'visual_token_pooler.col_embedding.weight': torch.arange(4.).view(4, 1).expand(-1, 6),
        'visual_token_pooler.view_embedding.weight': torch.randn(3, 6),
        'world_model.residual_predictor.token_embedding': spatial.reshape(1, 1, 48, 6),
        'action_models.aloha.token_embedding.weight': spatial.clone(),
        'action_models.franka.token_embedding.weight': spatial.clone(),
        'backbone.weight': torch.randn(6, 6),
    }


class SpatialCheckpointTests(unittest.TestCase):
    def test_native_grid_is_exact_noop(self):
        before = source()
        after = resize_spatial_checkpoint(before, num_views=3, grid_size=4)
        for key in before:
            self.assertIs(before[key], after[key])

    def test_resize_preserves_views_and_content_weights(self):
        before = source()
        after = resize_spatial_checkpoint(before, num_views=3, grid_size=8)
        self.assertIs(after['backbone.weight'], before['backbone.weight'])
        self.assertEqual(before['visual_token_pooler.row_embedding.weight'].shape, (4, 6))
        row = after['visual_token_pooler.row_embedding.weight'][:, 0]
        torch.testing.assert_close(row, torch.tensor([0., .25, .75, 1.25, 1.75, 2.25, 2.75, 3.]))
        for key in ['action_models.aloha.token_embedding.weight', 'action_models.franka.token_embedding.weight',
                    'world_model.residual_predictor.token_embedding']:
            view = after[key].reshape(3, 64, 6)
            for i in range(3):
                torch.testing.assert_close(view[i], torch.full((64, 6), float(i)))

    def test_rejects_inconsistent_source_geometry(self):
        before = source()
        with self.assertRaisesRegex(ValueError, 'camera view'):
            resize_spatial_checkpoint(before, num_views=2, grid_size=8)
        before['action_models.aloha.token_embedding.weight'] = torch.zeros(47, 6)
        with self.assertRaisesRegex(ValueError, 'source spatial grid'):
            resize_spatial_checkpoint(before, num_views=3, grid_size=8)

    def test_bfloat16_weights_are_preserved(self):
        before = {k: v.bfloat16() for k, v in source().items()}
        after = resize_spatial_checkpoint(before, num_views=3, grid_size=8)
        self.assertTrue(all(v.dtype == torch.bfloat16 for v in after.values()))


if __name__ == '__main__':
    unittest.main()
