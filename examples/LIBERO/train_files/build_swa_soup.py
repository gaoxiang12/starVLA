#!/usr/bin/env python
"""Exp11: build the SWA (stochastic weight averaging) model soup.

Averages the four dense snapshots collected during the SWA continuation run
(run_lewm_oft_swa_train.sh), i.e. steps_1000..4000 which correspond to
baseline_steps_8000 + {1000,2000,3000,4000}. Uniform weight-space average
of same-basin checkpoints -> a flatter minimum than any single checkpoint.

Verified result (libero_goal, 500ep x 3 rounds = 1500 episodes):
    baseline steps_8000 = 76.0%  (1000ep)
    swa4 soup           = 79.6%  (79.8/79.2/79.8) -> +3.6%, z=2.13, p~=0.03

Usage:
    python examples/LIBERO/train_files/build_swa_soup.py \
        --swa_dir playground/Checkpoints/lewm_oft_libero_swa/checkpoints \
        --steps 1000 2000 3000 4000 \
        --out playground/Checkpoints/_swa4_eval/checkpoints/swa4_pytorch_model.pt
"""
import argparse
import collections
import os

import torch


def build_soup(paths, out_path):
    sds = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    keys = list(sds[0].keys())
    for s in sds:
        assert list(s.keys()) == keys, "checkpoint key mismatch"
    soup = collections.OrderedDict()
    n = len(sds)
    for k in keys:
        if sds[0][k].is_floating_point():
            acc = sum(s[k].float() for s in sds) / n
            soup[k] = acc.to(sds[0][k].dtype)
        else:
            # integer buffers (e.g. position ids): copy from the last snapshot.
            soup[k] = sds[-1][k].clone()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(soup, out_path)
    print(f"saved SWA soup ({n} checkpoints averaged) -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swa_dir", required=True, help="dir with steps_*_pytorch_model.pt snapshots")
    ap.add_argument("--steps", type=int, nargs="+", default=[1000, 2000, 3000, 4000])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paths = [os.path.join(args.swa_dir, f"steps_{s}_pytorch_model.pt") for s in args.steps]
    build_soup(paths, args.out)


if __name__ == "__main__":
    main()
