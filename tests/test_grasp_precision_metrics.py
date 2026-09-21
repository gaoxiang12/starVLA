import pytest

from examples.Robotwin.audits.measure_grasp_precision import bin_mm, summarize


def test_thresholds_and_no_closure_are_distinct():
    assert [bin_mm(v) for v in [0,10,10.01,20,20.01,30,30.01,None]] == [
        'le_10mm','le_10mm','10_to_20mm','10_to_20mm','20_to_30mm','20_to_30mm','gt_30mm','no_closure']
    for value in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            bin_mm(value)


def test_no_closure_failure_stays_in_episode_denominator():
    rows = [dict(seed=1,preclosure_bin='le_10mm',scored_result=dict(success=True,first_attempt_success=True)),
            dict(seed=2,preclosure_bin='no_closure',scored_result=dict(success=False,first_attempt_success=False))]
    result = summarize(rows, planned=20)
    assert result['completed_episodes'] == 2 and result['pending_episodes'] == 18
    assert result['successes'] == 1 and result['preclosure_episode_bins']['no_closure'] == 1
    with pytest.raises(AssertionError):
        summarize(rows+[rows[0]])
