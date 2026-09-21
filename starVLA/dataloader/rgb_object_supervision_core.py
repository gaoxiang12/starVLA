"""Candidate training-label safeguard, opt-in and not used by running experiments."""
import cv2
import numpy as np

from starVLA.dataloader.rgb_object_supervision import candidates


def candidates_with_core(rgb):
    """Require a saturated interior before supervising a visible red component.

    Low-saturation robot lettering can pass the original hue/area rule when a
    cube is occluded. Reject uncertain positive labels without turning them into
    absence targets. Keep original component centroids and all old ambiguity
    checks. This is an image-only training label proposal, never an inference rule.
    """
    result = candidates(rgb)
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    hues = ((h <= 10) | (h >= 170), (h >= 35) & (h <= 85), (h >= 95) & (h <= 135))
    for row, hue in zip(result, hues):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (hue & (s >= 100) & (v >= 45)).astype(np.uint8), connectivity=8)
        core_area = 0
        if row['area']:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            core_area = int(((labels == component) & (s >= 220) & (v >= 80)).sum())
        # Evidence for this safeguard is red robot lettering. Preserve the green
        # and blue rules, including their small genuinely visible fragments.
        core_valid = row['color'] != 'red' or core_area >= 5
        row.update(original_accepted=row['accepted'], core_area=core_area,
                   accepted=bool(row['accepted'] and core_valid),
                   visibility_supervision_valid=bool(row['area'] < 12 or
                       (row['dominance'] >= .7 and core_valid)))
    return result
