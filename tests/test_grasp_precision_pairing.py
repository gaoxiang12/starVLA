from examples.Robotwin.audits.compare_grasp_precision_campaigns import transitions


def test_paired_gains_losses_and_unchanged_outcomes_keep_their_scene_ids():
    rows=[]
    for seed,before,after in [(1,False,True),(2,True,False),(3,True,True),(4,False,False)]:
        rows.append(dict(seed=seed,reference=dict(scored_result=dict(success=before)),
                         candidate=dict(scored_result=dict(success=after))))
    assert transitions(rows,'success')==dict(gained=[1],lost=[2],both_success=[3],both_failure=[4])
