"""Weak robust curvature of adjacent observations, with externally supplied masks."""
import torch
from torch.nn import functional as F


def temporal_curvature_loss(sequence, times, valid, event_weight, reference_dt=.05):
    if sequence.shape[1] != 3 or times.shape != sequence.shape[:2]:
        raise ValueError('Expected three neighboring observations with per-sample timestamps')
    if valid.shape != (sequence.shape[0],) or event_weight.shape != valid.shape:
        raise ValueError('Temporal validity and event weights must be per sample')
    dt=times[:,1:].float()-times[:,:-1].float()
    if not torch.isfinite(times).all() or ((dt<=0)&valid[:,None]).any():
        raise ValueError('Valid neighbors require finite increasing timestamps')
    if not torch.isfinite(event_weight).all() or ((event_weight<.25)|(event_weight>1)).any():
        raise ValueError('External event weights must be in [.25, 1]')
    dt=torch.where(valid[:,None],dt,torch.full_like(dt,reference_dt))
    x=sequence.float()
    shape=(sequence.shape[0],2)+((1,)*(x.ndim-2))
    velocity=(x[:,1:]-x[:,:-1])/dt.reshape(shape)
    span=dt.sum(1).reshape((len(x),)+((1,)*(x.ndim-2)))
    curvature=2*(velocity[:,1]-velocity[:,0])/span*reference_dt**2
    per_sample=F.smooth_l1_loss(curvature,torch.zeros_like(curvature),beta=.1,reduction='none').flatten(1).mean(1)
    # Normalize by valid sample count, not event weights; downweighting must have effect.
    return (per_sample*valid.float()*event_weight.detach()).sum()/valid.float().sum().clamp_min(1.)
