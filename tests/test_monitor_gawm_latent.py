from scripts.monitor_gawm_latent import assess


def rows(n=100):
    return [dict(step=(i+1)*20,latent_loss=.2,l1_action_loss=.05,delta_target_rms=.8,
                 delta_copy_mse=.64,delta_to_copy_ratio=.32,delta_scale=.8,
                 visual_content_rms=1.,visual_tokens_rms=1.01,latent_mse_over_delta_scale_sq=.32) for i in range(n)]


def test_healthy_and_magnitude_change_with_stable_relative_error():
    data=rows()
    for r in data[50:]:r.update(latent_loss=1.2,delta_scale=2.)
    assert assess(data)['health']=='healthy'


def test_detects_scale_drift_and_nonfinite_values():
    data=rows()
    for r in data[-3:]:r['visual_content_rms']=1.2
    assert 'content_scale_outside_0.95_1.05' in assess(data)['alerts']
    data[-1]['latent_loss']=float('nan')
    assert 'nonfinite_metric' in assess(data)['alerts']


def test_detects_joint_absolute_and_relative_growth():
    data=rows()
    for r in data[50:]:r.update(latent_loss=1.5,latent_mse_over_delta_scale_sq=1.4)
    assert 'absolute_and_scale_normalized_loss_rising' in assess(data)['alerts']


def test_early_single_spike_does_not_equal_sustained_divergence():
    data=rows(10);data[-1]['latent_loss']=100.
    assert assess(data)['health']=='healthy'
    assert assess([])['health']=='waiting_for_metrics'
    del data[-1]['visual_content_rms']
    assert 'missing_scale_metrics' in assess(data)['alerts']


def test_all_recent_loss_values_nonfinite_still_emits_warning():
    data=rows()
    for r in data[50:]:r['latent_loss']=float('nan')
    result=assess(data)
    assert result['health']=='warning' and 'nonfinite_metric' in result['alerts']
