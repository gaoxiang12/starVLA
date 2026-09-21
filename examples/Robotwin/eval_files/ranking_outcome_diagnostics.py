"""Read-only terminal diagnostics; never supplied to the policy."""
import numpy as np


def rgb_arrangement(positions, left_open, right_open):
    """Report the separate predicates used by blocks_ranking_rgb.check_success."""
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (3, 3) or not np.isfinite(positions).all():
        raise ValueError('Expected finite red/green/blue positions [3,3]')
    adjacent = np.abs(np.diff(positions[:, :2], axis=0))
    ordered = bool(positions[0, 0] < positions[1, 0] < positions[2, 0])
    close_x = bool((adjacent[:, 0] < .13).all())
    close_y = bool((adjacent[:, 1] < .03).all())
    return dict(final_block_positions_m=positions.tolist(),
                final_left_to_right_colors=[('red', 'green', 'blue')[i]
                                            for i in np.argsort(positions[:, 0], kind='stable')],
                correct_color_order=ordered, adjacent_x_within_tolerance=close_x,
                adjacent_y_within_tolerance=close_y,
                adjacent_xy_separation_m=adjacent.tolist(),
                left_gripper_open=bool(left_open), right_gripper_open=bool(right_open),
                all_success_predicates=bool(ordered and close_x and close_y and left_open and right_open))
