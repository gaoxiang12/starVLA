import numpy as np
import torch
from PIL import Image

from starVLA.model.modules.spatial_focus import SpatialFocus, native_crops
from starVLA.model.modules.action_model.ACT_ActionHeader import TurboStyleACTActionHead
from examples.Robotwin.audits.build_rgb_focus_labels import tcp_positions, project, gripper_events


def test_tcp_offset_is_local_not_world_x():
    p = np.array([[1.,2.,3.,2**-.5,0,0,2**-.5]])
    np.testing.assert_allclose(tcp_positions(p),[[1,2.12,3]],atol=1e-7)


def test_projection_masks_behind_and_outside():
    points=np.array([[0,0,1],[0,0,-1],[5,0,1.]])
    k=np.repeat(np.array([[[100,0,160],[0,100,120],[0,0,1.]]]),3,0)
    e=np.repeat(np.eye(4)[None,:3],3,0)
    xy,mask=project(points,k,e)
    assert mask.tolist()==[True,False,False]
    np.testing.assert_allclose(xy[0],[(160.5)/320,120.5/240])


def test_events_use_robotwin_gripper_order():
    a=np.zeros((30,14));a[:,[6,13]]=1;a[10:20,13]=0
    assert gripper_events(a)==[(10,1,0),(20,1,1)]


def test_native_crop_coordinates_and_content():
    source=np.zeros((240,320,3),np.uint8);source[:,:,0]=np.arange(320,dtype=np.uint16)%256
    crops,boxes=native_crops([[Image.fromarray(source)]],torch.tensor([[[.5,.5]]]),.4)
    assert crops[0][0][0].size==(128,96)
    assert np.asarray(crops[0][0][0])[0,0,0]==96
    torch.testing.assert_close(boxes,torch.tensor([[[.3,.3,.4,.4]]]))


def test_spatial_supervision_masks_nan_and_backpropagates():
    m=SpatialFocus(8,16,4,6,3,{})
    patches=torch.randn(2,3,16,8,requires_grad=True)
    memory,logits,xy=m.locate(patches,torch.randn(2,4),torch.randn(2,6))
    examples=[dict(spatial_target_xy=np.array([[.3,.4],[np.nan,np.nan],[.8,.2]]),spatial_target_valid=[True,False,True],view_valid_mask=[True,True,True]) for _ in range(2)]
    loss,*_=m.supervision(logits,xy,examples)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(patches.grad).all()
    assert patches.grad.abs().sum()>0
    for ex in examples: ex['spatial_target_valid']=[False]*3
    loss,*_=m.supervision(logits,xy,examples)
    assert loss.item()==0


def test_inference_crop_does_not_use_targets_or_random_jitter():
    m=SpatialFocus(8,16,4,6,3,{})
    m.eval()
    predicted=torch.rand(2,3,2)
    centers,p=m.crop_centers(predicted,torch.zeros_like(predicted),torch.ones(2,3,dtype=torch.bool),0)
    torch.testing.assert_close(centers,predicted)
    assert p==0
    m.train()
    centers,p=m.crop_centers(predicted,torch.zeros_like(predicted),torch.ones(2,3,dtype=torch.bool),3000)
    assert p==0


def test_masked_extra_memory_cannot_change_actions():
    focus=SpatialFocus(8,16,4,6,3,{})
    query=torch.randn(2,3,16)
    memory=torch.randn(2,7,16)*1000
    valid=torch.ones(2,7,dtype=torch.bool)
    torch.testing.assert_close(focus.refine_queries(query,{'dense':(memory,valid)}),query,atol=0,rtol=0)
    with torch.no_grad():focus.gates.fill_(1)
    valid.zero_()
    torch.testing.assert_close(focus.refine_queries(query,{'dense':(memory,valid)}),query,atol=0,rtol=0)


def test_dense_and_crop_memory_receive_action_gradients():
    focus=SpatialFocus(8,16,4,6,3,{'crop_dropout':0})
    with torch.no_grad():focus.gates.fill_(.1)
    patches=torch.randn(2,3,16,8,requires_grad=True)
    mem,_,_=focus.locate(patches,torch.randn(2,4),torch.randn(2,6))
    mask=torch.ones(2,3,dtype=torch.bool)
    dense,valid=focus.pack_dense(mem,mask)
    local,local_valid=focus.pack_local(patches,torch.tensor([[[.2,.2,.4,.4]]]*3).reshape(1,3,4).expand(2,-1,-1),mask)
    head=TurboStyleACTActionHead(token_dim=8,hidden_dim=16,action_dim=3,horizon=2,num_frames=3,num_visual_tokens=4,num_heads=2,num_layers=1,dim_feedforward=32,dropout=0)
    q=head.decode_action_queries(torch.randn(2,1,4,8))
    q=focus.refine_queries(q,{'dense':(dense,valid),'local':(local,local_valid)})
    head.predict_action(q).square().mean().backward()
    assert patches.grad.abs().sum()>0
    assert focus.position[0].weight.grad.abs().sum()>0


def test_event_sampling_keeps_full_trajectory_coverage_and_validation_uniform():
    from types import SimpleNamespace
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
    child=SimpleNamespace(trajectory_ids=np.array([0]),trajectory_lengths=np.array([101]),minimum_action_offset=1,data_cfg={'event_sampling_probability':.35},_spatial_episode=lambda _: {'event_anchors':np.arange(10)})
    mix=LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    mix.datasets=[child];mix._dataset_sampling_weights=np.array([1.]);mix._trajectory_sampling_weights=[np.array([1.])]
    mix.epoch,mix.seed,mix.mode=0,42,'train'
    anchors=np.array([mix.sample_step(i)[2] for i in range(5000)])
    assert len(set(anchors))==100 and (anchors<10).mean()>.3
    mix.mode='val'
    a=[mix.sample_step(i)[2] for i in range(200)]
    child.data_cfg['event_sampling_probability']=0
    assert a==[mix.sample_step(i)[2] for i in range(200)]


def test_bgr_correction_applies_to_current_native_and_future_consistently():
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
    dataset=LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    dataset.data_cfg={'video_channel_order':'bgr','preserve_native_images':True,'future_obs_frames':True}
    dataset._modality_keys={'video':['video.test'],'language':['language'],'action':['action.test']}
    dataset.tag='aloha'
    frames=np.zeros((3,240,320,3),np.uint8);frames[...,2]=255
    sample=dataset._pack_sample({'video.test':frames,'language':['task'],'action.test':np.ones((16,14))})
    for im in (sample['image'][0],sample['native_images'][0],sample['future_images'][0][0],sample['future_images'][1][0]):
        assert np.asarray(im)[0,0].tolist()==[255,0,0]
def test_camera_conditioning_changes_only_selected_camera_heatmap():
    from starVLA.model.modules.spatial_focus import SpatialFocus
    import torch

    torch.manual_seed(42)
    model = SpatialFocus(16, 16, 4, 8, 3, {})
    patches = torch.randn(2, 3, 196, 16)
    state, task = torch.randn(2, 4), torch.randn(2, 8)
    original_view = model.view.weight.detach().clone()
    delta = torch.randn(16) * 3
    for enabled in (False, True):
        model.cfg['view_conditioned_query'] = enabled
        with torch.no_grad():
            model.view.weight.copy_(original_view)
        before = model.locate(patches, state, task)[1].softmax(-1)
        with torch.no_grad():
            model.view.weight[0].add_(delta)
        after = model.locate(patches, state, task)[1].softmax(-1)
        torch.testing.assert_close(before[:, 1:], after[:, 1:], rtol=0, atol=0)
        if enabled:
            assert (before[:, 0] - after[:, 0]).abs().max() > 1e-3
        else:
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-7)


def test_zero_projection_preserves_policy_then_learns_feature_specific_residuals():
    torch.manual_seed(7)
    model = SpatialFocus(16, 16, 4, 8, 3, {'residual_mode': 'zero_projection'})
    queries = torch.randn(2, 4, 16)
    memory = torch.randn(2, 12, 16)
    valid = torch.ones(2, 12, dtype=torch.bool)
    memories = {'dense': (memory, valid)}
    target = queries + torch.randn_like(queries) * .1
    output = model.refine_queries(queries, memories)
    torch.testing.assert_close(output, queries, rtol=0, atol=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    (output - target).square().mean().backward()
    assert model.attentions[0].out_proj.weight.grad.abs().sum() > 0
    assert not model.gates.requires_grad
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    changed = model.refine_queries(queries, memories)
    assert (changed - queries).abs().max() > 1e-4
    (changed - target).square().mean().backward()
    assert model.attentions[0].in_proj_weight.grad.abs().sum() > 0
    empty = {'dense': (memory, torch.zeros_like(valid))}
    torch.testing.assert_close(model.refine_queries(queries, empty), queries, rtol=0, atol=0)
