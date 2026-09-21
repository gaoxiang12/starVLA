import numpy as np

from starVLA.dataloader.rgb_object_supervision_core import candidates_with_core


def test_muted_red_logo_is_unknown_not_absent():
    rgb = np.full((240, 320, 3), 180, np.uint8)
    rgb[100:110, 100:110] = [95, 37, 39]
    red = candidates_with_core(rgb)[0]
    assert red['original_accepted']
    assert not red['accepted'] and not red['visibility_supervision_valid']
    assert red['area'] >= 12


def test_occluded_saturated_red_fragment_keeps_centroid():
    rgb = np.full((240, 320, 3), 180, np.uint8)
    rgb[100:104, 100:104] = [170, 10, 10]
    red = candidates_with_core(rgb)[0]
    assert red['accepted'] and red['visibility_supervision_valid']
    assert red['center_xy'] == [101.5, 101.5]


def test_other_colors_absence_and_clipping_keep_original_semantics():
    rgb = np.full((240, 320, 3), 180, np.uint8)
    rgb[100:110, 100:110] = [37, 95, 39]
    red, green, blue = candidates_with_core(rgb)
    assert green['accepted'] and green['visibility_supervision_valid']
    assert green['core_area'] == 0
    assert red['area'] == blue['area'] == 0
    assert red['visibility_supervision_valid'] and blue['visibility_supervision_valid']
    rgb[100:110, :10] = [255, 0, 0]
    red = candidates_with_core(rgb)[0]
    assert not red['accepted'] and red['visibility_supervision_valid']
