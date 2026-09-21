"""Fixed held-out native-image localization audit for the learned object branch."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.dataloader.rgb_object_supervision import COLORS, candidates
from starVLA.model.modules.rgb_object_readout import RGBObjectReadout


def error_summary(rows):
    valid = [r for r in rows if r['target_valid']]
    if not valid:
        return dict(valid_targets=0)
    errors = np.asarray([r['pixel_error'] for r in valid])
    peak = np.asarray([r['peak_pixel_error'] for r in valid])
    return dict(valid_targets=len(valid), mean_pixel_error=float(errors.mean()),
        median_pixel_error=float(np.median(errors)), p90_pixel_error=float(np.percentile(errors, 90)),
        within_5px=float((errors <= 5).mean()), within_10px=float((errors <= 10).mean()),
        mean_peak_pixel_error=float(peak.mean()),
        image_center_baseline_error=float(np.mean([r['image_center_error'] for r in valid])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchors-per-episode', type=int, default=8)
    parser.add_argument('--temperature-probe', nargs='*', type=float, default=[])
    args = parser.parse_args()
    if args.output.exists() or args.anchors_per_episode < 2:
        raise ValueError('Use a new output path and at least two anchors per episode')
    if any(not np.isfinite(t) or t <= 0 for t in args.temperature_probe):
        raise ValueError('Diagnostic temperatures must be finite and positive')
    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.config)
    split_path = Path(cfg.datasets.vla_data.episode_split_manifest)
    split = json.loads(split_path.read_text())
    cfg.datasets.vla_data.episode_split = 'validation'
    data = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    child, = data.datasets
    child.transforms.eval()
    assert set(map(int, child.trajectory_ids)) == set(split['validation_episode_ids'])
    assert set(map(int, child.trajectory_ids)).isdisjoint(split['train_episode_ids'])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    prefix = 'spatial_focus.object_readout.'
    state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    model = RGBObjectReadout(state['identity.weight'].shape[1]).eval()
    model.load_state_dict(state, strict=True)
    del state
    rows, frame_records = [], []
    examples, keys = [], []

    def flush():
        if not examples:
            return
        head_valid = torch.as_tensor([ex['view_valid_mask'][0] for ex in examples], dtype=torch.bool)
        images, arrays = model.head_images(examples, head_valid, 'cpu')
        with torch.inference_mode():
            _, prediction = model.encode(images, head_valid)
        xy = prediction['xy'].numpy() * [320, 240] - .5
        probabilities = prediction['logits'].softmax(-1)
        peak_xy = prediction['coordinates'][prediction['logits'].argmax(-1)].numpy() * [320, 240] - .5
        visibility = prediction['visibility'].sigmoid().numpy()
        temperature_xy = {str(t): ((prediction['logits']/t).softmax(-1) @ prediction['coordinates']).numpy()
                          * [320, 240] - .5 for t in args.temperature_probe}
        for index, (key, image) in enumerate(zip(keys, arrays)):
            proposals = candidates(image)
            frame_records.append(dict(**key, native_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
            for color_index, proposal in enumerate(proposals):
                valid = bool(proposal['accepted'] and head_valid[index])
                target = np.asarray(proposal['center_xy']) if valid else None
                rows.append(dict(**key, color=COLORS[color_index], target_valid=valid,
                    predicted_xy=xy[index, color_index].tolist(), target_xy=target.tolist() if valid else None,
                    pixel_error=float(np.linalg.norm(xy[index, color_index]-target)) if valid else None,
                    peak_pixel_error=float(np.linalg.norm(peak_xy[index, color_index]-target)) if valid else None,
                    image_center_error=float(np.linalg.norm(np.array([159.5, 119.5])-target)) if valid else None,
                    predicted_visible=float(visibility[index, color_index]),
                    peak_probability=float(probabilities[index, color_index].max()),
                    temperature_pixel_errors={t: float(np.linalg.norm(points[index, color_index]-target))
                                               if valid else None for t, points in temperature_xy.items()},
                    candidate_area=proposal['area'], candidate_dominance=proposal['dominance'],
                    candidate_edge_clipped=proposal['touches_image_edge']))
        examples.clear(); keys.clear()

    for episode, length in sorted(zip(map(int, child.trajectory_ids), map(int, child.trajectory_lengths))):
        for anchor in np.unique(np.linspace(0, length-1, args.anchors_per_episode, dtype=int)):
            sample = child._pack_sample(child.transforms(child.get_step_data(episode, int(anchor))))
            # Only current native images and view validity enter this audit's network.
            examples.append(dict(native_images=sample['native_images'], view_valid_mask=sample['view_valid_mask']))
            keys.append(dict(episode=episode, anchor=int(anchor), progress=float(anchor/(length-1))))
            if len(examples) == 4:
                flush()
    flush()
    with args.checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:
        from examples.Robotwin.audits.run_rgb_contact_refinement import digest as file_digest
        digest = file_digest(args.checkpoint)
    report = dict(state='complete', checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
        device='cpu', module='RGBObjectReadout only; same weights and native current RGB',
        validation_episode_ids=sorted(map(int, child.trajectory_ids)), frames=len(frame_records),
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        input_manifest_sha256=hashlib.sha256(json.dumps(frame_records, sort_keys=True).encode()).hexdigest(),
        summary=error_summary(rows), by_color={color:error_summary([r for r in rows if r['color']==color]) for color in COLORS},
        temperature_probe={str(t): dict(mean_pixel_error=float(np.mean([
            r['temperature_pixel_errors'][str(t)] for r in rows if r['target_valid']])) )
            for t in args.temperature_probe},
        by_progress={str(i):error_summary([r for r in rows if min(3, int(r['progress']*4)) == i]) for i in range(4)},
        rows=rows, frame_records=frame_records,
        note='Held-out scene-safe observations, but targets are heuristic visible color-component centers, not '
             'independent ground truth, amodal centers, TCP targets, or task success. Clipped/ambiguous targets '
             'are excluded. CPU branch-only inference is not an exact replay of CUDA policy actions. '
             'Temperature probes only recompute heatmap coordinates; no policy weights, pooling, or actions are changed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(frames=report['frames'], summary=report['summary'], by_color=report['by_color'],
                         temperature_probe=report['temperature_probe']), indent=2))


if __name__ == '__main__':
    main()
