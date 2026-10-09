"""Compatibility name for the retired spatial-focus experiment."""
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.rgb_object_readout import RGBObjectReadout


class CoreSupervisedObjectReadout(RGBObjectReadout):
    def supervision(self, prediction, arrays, head_valid, cfg):
        if not cfg.get('object_label_red_core', False):
            return super().supervision(prediction, arrays, head_valid, cfg)
        # Candidate image rules remain exclusively in training supervision.
        from starVLA.model.modules.rgb_object_core_loss import object_core_supervision
        return object_core_supervision(prediction, arrays, head_valid, cfg)


@FRAMEWORK_REGISTRY.register("GAWMObjectFusion")
class GAWMObjectFusion:
    def __init__(self, cfg):
        raise ValueError(
            "GAWMObjectFusion has been retired with spatial_focus; "
            "use the experiment's archived source_snapshot"
        )
