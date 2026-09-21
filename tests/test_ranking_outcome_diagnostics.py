import pytest

from examples.Robotwin.eval_files.ranking_outcome_diagnostics import rgb_arrangement


def test_wrong_order_preserves_color_adjacent_distance_checks():
    outcome = rgb_arrangement([[.08, -.15, .76], [-.08, -.15, .76], [0, -.15, .76]], True, True)
    assert outcome['final_left_to_right_colors'] == ['green', 'blue', 'red']
    assert outcome['adjacent_x_within_tolerance'] is False  # RGB-adjacent red/green gap is .16.
    assert outcome['adjacent_y_within_tolerance'] is True
    assert outcome['correct_color_order'] is False
    assert outcome['all_success_predicates'] is False


def test_correct_arrangement_still_requires_both_grippers_open():
    positions = [[-.08, -.15, .76], [0, -.15, .76], [.08, -.15, .76]]
    assert rgb_arrangement(positions, True, True)['all_success_predicates']
    held = rgb_arrangement(positions, True, False)
    assert held['correct_color_order'] and not held['all_success_predicates']


def test_y_alignment_failure_is_reported_independently():
    outcome = rgb_arrangement([[0, 0, 0], [.08, .04, 0], [.16, .04, 0]], True, True)
    assert outcome['correct_color_order'] and outcome['adjacent_x_within_tolerance']
    assert not outcome['adjacent_y_within_tolerance']


def test_invalid_geometry_cannot_be_reported_as_a_task_result():
    with pytest.raises(ValueError):
        rgb_arrangement([[float('nan'), 0, 0]] * 3, True, True)
