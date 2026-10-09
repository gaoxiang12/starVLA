"""Compatibility name for the retired spatial-focus experiment."""
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("GAWMCompactExpert")
class GAWMCompactExpert:
    def __init__(self, cfg):
        raise ValueError(
            "GAWMCompactExpert has been retired with spatial_focus; "
            "use the experiment's archived source_snapshot"
        )
