import pytest

from examples.Robotwin.audits.export_grasp_precision_pairs import frame_sim_times


def test_completed_action_has_a_final_observation():
    result=dict(state='complete',frames=4,termination='success',result=dict(elapsed_sim_s=2.5))
    assert frame_sim_times(result,[dict(sim_s=t) for t in (0,1,2)])==[0,1,2,2.5]


def test_interrupted_action_does_not_invent_a_final_video_frame():
    result=dict(state='complete',frames=3,termination='simulation_time_budget',result=dict(elapsed_sim_s=2.5))
    assert frame_sim_times(result,[dict(sim_s=t) for t in (0,1,2)])==[0,1,2]
    result['termination']='success'
    with pytest.raises(AssertionError):
        frame_sim_times(result,[dict(sim_s=t) for t in (0,1,2)])
