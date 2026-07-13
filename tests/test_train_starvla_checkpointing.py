import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from starVLA.training.train_starvla import VLATrainer


class FakeAccelerator:
    def __init__(self, is_main_process=True):
        self.is_main_process = is_main_process
        self.saved_states = []
        self.loaded_states = []
        self.printed = []

    def get_state_dict(self, model):
        return {"weight": torch.tensor([1.0])}

    def wait_for_everyone(self):
        return None

    def save_state(self, path):
        self.saved_states.append(path)

    def load_state(self, path):
        self.loaded_states.append(path)

    def print(self, message):
        self.printed.append(message)


class CheckpointingTest(unittest.TestCase):
    def make_trainer(self, output_dir, is_resume=True, is_main_process=True):
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.config = SimpleNamespace(
            output_dir=str(output_dir),
            trainer=SimpleNamespace(
                pretrained_checkpoint=None,
                is_resume=is_resume,
                save_format="pt",
            ),
        )
        trainer.accelerator = FakeAccelerator(is_main_process=is_main_process)
        trainer.model = object()
        trainer.completed_steps = 0
        trainer.lr_scheduler = MagicMock()
        trainer.load_pretrained_backbones = MagicMock(return_value=trainer.model)
        return trainer

    def test_full_state_is_preferred_for_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir()
            weights_path = checkpoint_dir / "steps_20_pytorch_model.pt"
            weights_path.touch()
            state_dir = checkpoint_dir / "steps_20_training_state"
            state_dir.mkdir()
            trainer = self.make_trainer(output_dir)
            trainer._get_latest_checkpoint = MagicMock(return_value=(str(weights_path), 20))

            trainer._init_checkpointing()
            trainer._adjust_lr_scheduler_for_resume()

            self.assertEqual(trainer.resume_training_state, str(state_dir))
            trainer.load_pretrained_backbones.assert_not_called()
            trainer.lr_scheduler.step.assert_not_called()

    def test_legacy_checkpoint_falls_back_to_weights_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir()
            weights_path = checkpoint_dir / "steps_20_pytorch_model.pt"
            weights_path.touch()
            trainer = self.make_trainer(output_dir)
            trainer._get_latest_checkpoint = MagicMock(return_value=(str(weights_path), 20))

            with self.assertLogs(level=logging.WARNING):
                trainer._init_checkpointing()
            trainer._adjust_lr_scheduler_for_resume()

            self.assertIsNone(trainer.resume_training_state)
            trainer.load_pretrained_backbones.assert_called_once()
            self.assertEqual(trainer.lr_scheduler.step.call_count, 20)

    def test_repair_lr_scheduler_rewinds_to_external_step(self):
        trainer = self.make_trainer("unused")
        trainer.config.trainer.repair_lr_scheduler_on_resume = True
        trainer.completed_steps = 30
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-4}])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(step / 10, 1.0),
        )
        scheduler.step(120)
        trainer.lr_scheduler = SimpleNamespace(scheduler=scheduler)

        trainer._repair_lr_scheduler_after_resume()

        self.assertEqual(scheduler.last_epoch, 30)
        self.assertEqual(scheduler._step_count, 31)
        self.assertEqual(scheduler.get_last_lr(), [1e-4])

    def test_save_checkpoint_writes_training_state_on_every_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "checkpoints").mkdir()
            trainer = self.make_trainer(output_dir, is_resume=False, is_main_process=False)
            trainer.checkpoint_dir = str(output_dir / "checkpoints")
            trainer.completed_steps = 30

            trainer._save_checkpoint()

            self.assertEqual(
                trainer.accelerator.saved_states,
                [str(output_dir / "checkpoints" / "steps_30_training_state")],
            )

    def test_main_rank_keeps_deployment_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "checkpoints").mkdir()
            trainer = self.make_trainer(output_dir, is_resume=False)
            trainer.checkpoint_dir = str(output_dir / "checkpoints")
            trainer.completed_steps = 30

            with patch("torch.save") as save:
                trainer._save_checkpoint()

            save.assert_called_once()
            self.assertEqual(
                save.call_args.args[1],
                str(output_dir / "checkpoints" / "steps_30_pytorch_model.pt"),
            )


if __name__ == "__main__":
    unittest.main()
