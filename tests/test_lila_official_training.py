"""Regression checks for failure handling and resumable official training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from omegaconf import OmegaConf
import torch

from examples.LiLaWAM.official_robotwin_data import extract, validate_episode_ids, expected_episode_count
from examples.LiLaWAM.train_official_robotwin import EpochSampler, save_checkpoint, strict_collate


class OfficialTrainingTests(unittest.TestCase):
    def test_eight_ranks_preserve_global_batches_and_resume(self):
        size, batch, world = 1031, 128, 8
        full=list(EpochSampler(size,42))[:size//batch*batch]
        ranks=[list(EpochSampler(size,42,0,r,world,batch)) for r in range(world)]
        merged=[]
        local=batch//world
        for step in range(size//batch):
            for r in range(world):
                merged.extend(ranks[r][step*local:(step+1)*local])
        self.assertEqual(merged,full)
        self.assertEqual(len(set(merged)),len(merged))
        for r in range(world):
            self.assertEqual(list(EpochSampler(size,42,3*batch,r,world,batch)),ranks[r][3*local:])
        with self.assertRaises(ValueError):
            EpochSampler(size,42,1,0,world,batch)

    def test_only_documented_outliers_may_be_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder)/'outliers.txt'
            report.write_text('[Action] /data/task/demo_clean/data/episode2.hdf5 -> Dims: [0]\n')
            names={'demo_clean':[f'episode{i}.hdf5' for i in range(50) if i != 2],
                   'demo_randomized':[f'episode{i}.hdf5' for i in range(500)]}
            self.assertEqual(validate_episode_ids('task',names,report)['demo_clean'],[2])
            self.assertEqual(expected_episode_count(report),27499)
            names['demo_clean'].remove('episode3.hdf5')
            with self.assertRaisesRegex(ValueError,'missing=\\[3\\]'):
                validate_episode_ids('task',names,report)
    def test_resume_sampler_preserves_remaining_samples(self):
        full = list(EpochSampler(103, 42))
        resumed = list(EpochSampler(103, 42, 32))
        self.assertEqual(resumed, full[32:])
        self.assertEqual(len(set(full)), 103)
        self.assertNotEqual(full, list(EpochSampler(103, 43)))

    def test_bad_sample_cannot_silently_shrink_batch(self):
        with self.assertRaisesRegex(ValueError, 'failed sample'):
            strict_collate([{'x':torch.ones(2)}, None])
        self.assertEqual(strict_collate([{'x':torch.ones(2)}, {'x':torch.zeros(2)}])['x'].shape, (2,2))

    def test_checkpoint_exports_safe_weights_and_optimizer_state(self):
        with tempfile.TemporaryDirectory() as folder:
            model = torch.nn.Linear(3,2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            model(torch.ones(4,3)).square().mean().backward()
            optimizer.step()
            scheduler.step()
            with patch('torch.cuda.get_rng_state', return_value=torch.get_rng_state()):
                save_checkpoint(Path(folder)/'latest.pt',model,optimizer,scheduler,0,1,1,
                                OmegaConf.create({'seed':42}),'initial_hash')
            weights = torch.load(Path(folder)/'policy.pt',weights_only=True)
            clone = torch.nn.Linear(3,2)
            clone.load_state_dict(weights['model_state_dict'],strict=True)
            self.assertTrue(torch.equal(model(torch.ones(1,3)),clone(torch.ones(1,3))))
            full = torch.load(Path(folder)/'latest.pt',weights_only=False)
            self.assertEqual(full['step_in_epoch'],1)
            self.assertTrue(full['optimizer_state_dict']['state'])
            self.assertEqual(full['scheduler_state_dict']['last_epoch'],1)

    def test_archive_rejects_traversal_and_incomplete_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with zipfile.ZipFile(root/'bad.zip','w') as z:
                z.writestr('../escape.hdf5',b'bad')
            with self.assertRaisesRegex(ValueError,'Unsafe archive'):
                extract(root/'bad.zip','task',root)
            with zipfile.ZipFile(root/'short.zip','w') as z:
                z.writestr('task/demo_clean/data/episode0.hdf5',b'bad')
            with self.assertRaisesRegex(ValueError,'unexpected published counts'):
                extract(root/'short.zip','task',root)


if __name__ == '__main__':
    unittest.main()
