import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate import Accelerator
from transformers import get_scheduler

from examples.LiLaWAM.prepare_continuation import cosine_base
from starVLA.training.train_starvla import build_accelerator


class ContinuationTests(unittest.TestCase):
    def test_loaded_scheduler_preserves_lr_then_decays_to_new_target(self):
        start, end, current, minimum = 19998, 100000, 5e-5, 1e-5
        base = cosine_base(start, end, current, minimum)
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=base)
        scheduler = get_scheduler('cosine_with_min_lr', optimizer,
            num_warmup_steps=0, num_training_steps=end,
            scheduler_specific_kwargs={'min_lr': minimum})
        state = scheduler.state_dict()
        state.update(base_lrs=[base], last_epoch=start, _step_count=start+1, _last_lr=[current])
        optimizer.param_groups[0]['lr'] = current
        scheduler.load_state_dict(state)
        self.assertEqual(optimizer.param_groups[0]['lr'], current)
        previous = current
        for step in range(start+1, end+1):
            # Exercise the actual Transformers lambda used after state restore.
            value = scheduler.base_lrs[0]*scheduler.lr_lambdas[0](step)
            self.assertLessEqual(value, previous)
            previous = value
        self.assertAlmostEqual(previous, minimum, places=14)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], current, delta=1e-9)

    @patch('starVLA.training.train_starvla.Accelerator')
    @patch('starVLA.training.train_starvla.DeepSpeedPlugin')
    def test_accumulation_does_not_flush_at_loader_boundaries(self, ds, accelerator):
        build_accelerator(SimpleNamespace(trainer=SimpleNamespace(
            gradient_accumulation_steps=4, sync_with_dataloader=False)))
        plugin = accelerator.call_args.kwargs['gradient_accumulation_plugin']
        self.assertFalse(plugin.sync_with_dataloader)
        sync = []
        state = SimpleNamespace(sync_with_dataloader=plugin.sync_with_dataloader,
            end_of_dataloader=False, num_steps=plugin.num_steps, _set_sync_gradients=sync.append)
        fake = SimpleNamespace(step=0, gradient_state=state)
        for microstep in range(1, 13):
            state.end_of_dataloader = microstep % 5 == 0
            Accelerator._do_sync(fake)
        self.assertEqual([i+1 for i, yes in enumerate(sync) if yes], [4, 8, 12])


if __name__ == '__main__':
    unittest.main()
