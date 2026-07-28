import unittest

import numpy as np
import torch
import torch.nn.functional as F

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.model.framework.WM4A.LeWMOFT import LeWM_OFT
from starVLA.model.modules.world_model.latent_progress import (
    LatentGoalPredictor,
    LatentProgressChecker,
    ProgressActionConditioner,
    latent_goal_loss,
    progress_ranking_loss,
)


class LatentProgressComponentsTest(unittest.TestCase):
    def test_goal_prediction_and_triplet_progress_are_differentiable(self):
        torch.manual_seed(3)
        goal_predictor = LatentGoalPredictor(
            latent_dim=12,
            task_dim=8,
            hidden_dim=24,
            num_tokens=6,
        )
        checker = LatentProgressChecker(latent_dim=12, hidden_dim=24)
        start = torch.randn(2, 6, 12)
        current = torch.randn(2, 6, 12)
        target_goal = torch.randn(2, 6, 12)
        task = torch.randn(2, 8)

        predicted_goal = goal_predictor(start, task)
        output = checker(start, current, predicted_goal)
        loss = (
            output["progress"].mean()
            + output["geometric_progress"].mean()
            + latent_goal_loss(predicted_goal, target_goal)
        )
        loss.backward()

        self.assertEqual(predicted_goal.shape, target_goal.shape)
        self.assertEqual(output["progress"].shape, (2,))
        self.assertTrue(torch.all(output["progress"] >= 0))
        self.assertTrue(torch.all(output["progress"] <= 1))
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in goal_predictor.parameters()
            )
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in checker.parameters())
        )

    def test_action_conditioner_is_exact_noop_at_initialization(self):
        conditioner = ProgressActionConditioner(
            action_hidden_dim=16,
            chunk_len=4,
            hidden_dim=12,
            dropout=0.0,
        )
        queries = torch.randn(3, 4, 16, requires_grad=True)
        progress = torch.tensor([0.0, 0.5, 1.0])

        initial = conditioner(queries, progress)
        self.assertTrue(torch.equal(initial, queries))
        initial.square().mean().backward()
        self.assertGreater(
            float(conditioner.conditioner[-1].weight.grad.abs().sum()), 0.0
        )

        torch.nn.init.normal_(conditioner.conditioner[-1].weight, std=0.02)
        conditioned = conditioner(queries.detach(), progress)
        self.assertFalse(torch.allclose(conditioned, queries.detach()))
        self.assertFalse(torch.allclose(conditioned[0], conditioned[1]))

    def test_degenerate_geometry_remains_finite_across_optimizer_steps(self):
        """A collapsed predicted goal must not poison the trainable branch."""
        torch.manual_seed(11)
        checker = LatentProgressChecker(latent_dim=8, hidden_dim=16)
        conditioner = ProgressActionConditioner(
            action_hidden_dim=12,
            chunk_len=4,
            hidden_dim=8,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(
            list(checker.parameters()) + list(conditioner.parameters()),
            lr=1e-3,
        )
        start = torch.randn(3, 4, 8)
        current = start.clone()
        # Exercise the singular case ||goal - start|| == 0 directly.
        collapsed_goal = start.clone()
        target = torch.tensor([0.0, 0.5, 1.0])
        queries = torch.randn(3, 4, 12)

        for _ in range(5):
            optimizer.zero_grad()
            output = checker(start, current, collapsed_goal)
            conditioned = conditioner(queries, output["progress"])
            loss = F.smooth_l1_loss(output["progress"], target)
            loss = loss + 0.01 * conditioned.square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in list(checker.parameters())
                    + list(conditioner.parameters())
                )
            )
            optimizer.step()
            self.assertTrue(
                all(
                    torch.isfinite(parameter).all()
                    for parameter in list(checker.parameters())
                    + list(conditioner.parameters())
                )
            )

    def test_ranking_loss_uses_only_same_episode_order(self):
        progress = torch.tensor([0.8, 0.2, 0.9])
        target = torch.tensor([0.0, 1.0, 0.5])
        loss = progress_ranking_loss(
            progress, target, ["episode-a", "episode-a", "episode-b"], margin=0.02
        )
        self.assertTrue(torch.isclose(loss, torch.tensor(0.62)))

        no_pairs = progress_ranking_loss(
            progress, target, ["a", "b", "c"], margin=0.02
        )
        self.assertEqual(float(no_pairs), 0.0)

    def test_inference_cache_resets_on_episode_start(self):
        framework = object.__new__(LeWM_OFT)
        torch.nn.Module.__init__(framework)
        framework.progress_goal_predictor = LatentGoalPredictor(
            latent_dim=8,
            task_dim=6,
            hidden_dim=16,
            num_tokens=4,
        )
        framework.progress_checker = LatentProgressChecker(
            latent_dim=8, hidden_dim=16
        )
        framework.progress_ema = 0.5
        framework.reset_progress_state()
        task = torch.randn(1, 6)
        first = torch.randn(1, 4, 8)
        second = torch.randn(1, 4, 8)

        first_output = framework._inference_progress(
            current_latent=first,
            task_embedding=task,
            instructions=["pick up the cup"],
            examples=[{"episode_start": True}],
        )
        cached_start = framework._progress_start_latent.clone()
        second_output = framework._inference_progress(
            current_latent=second,
            task_embedding=task,
            instructions=["pick up the cup"],
            examples=[{}],
        )
        self.assertTrue(torch.equal(framework._progress_start_latent, cached_start))
        self.assertTrue(
            torch.allclose(
                second_output["progress"],
                0.5 * first_output["progress"]
                + 0.5 * second_output["raw_progress"],
            )
        )

        framework._inference_progress(
            current_latent=second,
            task_embedding=task,
            instructions=["pick up the cup"],
            examples=[{"episode_start": True}],
        )
        self.assertTrue(torch.equal(framework._progress_start_latent, second))

    def test_runtime_ablation_selects_learned_fixed_or_disabled_progress(self):
        framework = object.__new__(LeWM_OFT)
        torch.nn.Module.__init__(framework)
        framework.use_progress_checker = True
        framework.progress_ema = 0.8
        framework.reset_progress_state()
        learned = torch.tensor([0.2, 0.9])

        framework.configure_progress_inference(mode="learned", ema=0.4)
        self.assertIs(
            framework._select_inference_progress_for_action(learned), learned
        )
        self.assertEqual(framework.progress_ema, 0.4)

        framework.configure_progress_inference(mode="fixed", fixed_value=0.5)
        self.assertTrue(
            torch.equal(
                framework._select_inference_progress_for_action(learned),
                torch.tensor([0.5, 0.5]),
            )
        )

        framework.configure_progress_inference(mode="disabled")
        self.assertIsNone(
            framework._select_inference_progress_for_action(learned)
        )
        with self.assertRaisesRegex(ValueError, "fixed progress"):
            framework.configure_progress_inference(
                mode="fixed", fixed_value=1.1
            )

    def test_training_progress_detaches_visual_coordinates(self):
        framework = object.__new__(LeWM_OFT)
        torch.nn.Module.__init__(framework)
        framework.progress_goal_predictor = LatentGoalPredictor(
            latent_dim=8,
            task_dim=6,
            hidden_dim=16,
            num_tokens=4,
        )
        framework.progress_checker = LatentProgressChecker(
            latent_dim=8, hidden_dim=16
        )
        framework.progress_detach_latents = True
        framework.progress_loss_weight = 0.2
        framework.progress_anchor_weight = 0.5
        framework.progress_ranking_weight = 0.1
        framework.progress_goal_weight = 0.2
        start = torch.randn(2, 4, 8, requires_grad=True)
        current = torch.randn(2, 4, 8, requires_grad=True)
        goal = torch.randn(2, 4, 8, requires_grad=True)
        task = torch.randn(2, 6, requires_grad=True)

        output = framework._training_progress(
            start_latent=start,
            current_latent=current,
            target_goal_latent=goal,
            task_embedding=task,
            target=torch.tensor([0.2, 0.8]),
            episode_ids=["same", "same"],
        )
        output["progress_auxiliary_loss"].backward()

        self.assertIsNone(start.grad)
        self.assertIsNone(current.grad)
        self.assertIsNone(goal.grad)
        self.assertIsNone(task.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in framework.progress_checker.parameters()
            )
        )


class ProgressDatasetFieldsTest(unittest.TestCase):
    def _make_dataset(self, enabled=True):
        dataset = object.__new__(LeRobotSingleDataset)
        dataset.data_cfg = {"include_progress": enabled}
        dataset._trajectory_lengths = np.asarray([11])
        dataset._modality_keys = {"video": ["video.primary", "video.wrist"]}
        dataset._delta_indices = {
            "video.primary": np.asarray([-1, 0, 4]),
            "video.wrist": np.asarray([-1, 0, 4]),
        }
        dataset._dataset_name = "fixture"
        dataset.get_trajectory_index = lambda trajectory_id: 0

        def get_video(trajectory_id, video_key, base_index):
            value = 10 if base_index == 10 else 1
            frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
            frames[1].fill(value)
            return frames

        dataset.get_video = get_video
        return dataset

    def test_endpoint_images_and_progress_target_are_attached(self):
        dataset = self._make_dataset(enabled=True)
        output = dataset._attach_progress_fields({}, trajectory_id=7, base_index=5)

        self.assertEqual(float(output["progress_target"]), 0.5)
        self.assertEqual(output["progress_episode_id"], "fixture:7")
        self.assertEqual(len(output["progress_start_image"]), 2)
        self.assertEqual(len(output["progress_goal_image"]), 2)
        self.assertEqual(
            int(np.asarray(output["progress_start_image"][0])[0, 0, 0]), 1
        )
        self.assertEqual(
            int(np.asarray(output["progress_goal_image"][0])[0, 0, 0]), 10
        )

    def test_disabled_progress_preserves_sample_identity(self):
        dataset = self._make_dataset(enabled=False)
        sample = {"unchanged": True}
        self.assertIs(
            dataset._attach_progress_fields(sample, trajectory_id=7, base_index=5),
            sample,
        )


if __name__ == "__main__":
    unittest.main()
