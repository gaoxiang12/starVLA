import unittest

import torch
from omegaconf import OmegaConf

from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.encoder = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)


class ParamLearningRateGroupsTest(unittest.TestCase):
    def test_nested_module_gets_independent_learning_rate(self):
        model = NestedModel()
        config = wrap_config(OmegaConf.create(
            {
                "trainer": {
                    "learning_rate": {
                        "base": 1e-4,
                        "backbone": {"encoder": 1e-6},
                    },
                    "freeze_modules": "",
                }
            }
        ))

        groups = build_param_lr_groups(model, config)

        self.assertEqual(
            [(group["name"], group["lr"]) for group in groups],
            [("backbone.encoder", 1e-6), ("base", 1e-4)],
        )
        self.assertEqual(
            {id(parameter) for parameter in groups[0]["params"]},
            {id(parameter) for parameter in model.backbone.encoder.parameters()},
        )


if __name__ == "__main__":
    unittest.main()
