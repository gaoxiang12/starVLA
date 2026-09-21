"""Guard full-state handoffs against silent counter and optimizer mismatches."""
import tempfile
import unittest
from pathlib import Path

import torch
from examples.LiLaWAM.resume_when_saved import validate_checkpoint


class HandoffCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'latest').write_text('pytorch_model')
        (self.root / 'pytorch_model').mkdir()
        torch.save({'global_steps': 10}, self.root / 'pytorch_model/mp_rank_00_model_states.pt')
        torch.save({'last_epoch': 10}, self.root / 'scheduler.bin')
        torch.save({'step': 160}, self.root / 'random_states_0.pkl')
        self.optim = self.root / 'pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
        torch.save({'optimizer': {}}, self.optim)

    def test_aligned_accumulation_change(self):
        self.assertEqual(validate_checkpoint(self.root, 10, 4)['optimizer_step'], 10)

    def test_optimizer_step_mismatch(self):
        torch.save({'global_steps': 9}, self.root / 'pytorch_model/mp_rank_00_model_states.pt')
        with self.assertRaisesRegex(ValueError, 'optimizer/scheduler'):
            validate_checkpoint(self.root, 10, 4)

    def test_accumulation_boundary_mismatch(self):
        torch.save({'step': 3}, self.root / 'random_states_0.pkl')
        with self.assertRaisesRegex(ValueError, 'accumulation boundary'):
            validate_checkpoint(self.root, 10, 4)

    def test_missing_optimizer(self):
        self.optim.unlink()
        with self.assertRaisesRegex(ValueError, 'optimizer state'):
            validate_checkpoint(self.root, 10, 4)


if __name__ == '__main__':
    unittest.main()
