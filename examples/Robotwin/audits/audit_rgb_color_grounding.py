"""Counterfactual first-target grounding at a saved live episode start."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.task_language import resolve_task_language


def color_centers(image):
    centers = []
    pixels = image.astype(np.float32)
    for color in range(3):
        others = np.delete(pixels, color, axis=-1).max(-1)
        mask = ((pixels[..., color] > 80) & (pixels[..., color] > 1.7*others)).astype(np.uint8)
        count, _, stats, xy = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            raise ValueError('Missing a visible primary-color block')
        index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[index, cv2.CC_STAT_AREA] < 20:
            raise ValueError('Primary-color component too small for this audit')
        centers.append(xy[index].tolist())
    return np.asarray(centers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.trace, allow_pickle=False) as trace:
        metadata = json.loads(trace['metadata'].item())
        saved = {key: trace[key].copy() for key in ('image', 'native_images', 'image_history', 'state')}
    if not metadata['episode_start'] or np.any(saved['state'].reshape(-1)[12:14] < .95):
        raise ValueError('This audit assumes the episode start with both empty grippers open')
    if metadata['lang'] != resolve_task_language('', 'blocks_ranking_rgb', 'dataset_name'):
        raise ValueError('First-target identity is specific to blocks_ranking_rgb')
    centers = color_centers(saved['native_images'][0])
    wrapper = PolicyServerWrapper(str(args.checkpoint), use_bf16=False, unnorm_key=metadata['unnorm_key'])
    model = wrapper._framework
    original = model.predict_action
    captured = {}

    def predict(*positional, **kwargs):
        result = original(*positional, **kwargs)
        captured.clear()
        captured.update(result)
        return result

    model.predict_action = predict
    rows = []
    baseline = None
    for order in itertools.permutations(range(3)):
        example = dict(state=saved['state'].copy(), lang=metadata['lang'], episode_start=True)
        for key in ('image', 'native_images'):
            example[key] = [np.ascontiguousarray(image[..., list(order)]) for image in saved[key]]
        example['image_history'] = [[np.ascontiguousarray(image[..., list(order)]) for image in frame]
                                    for frame in saved['image_history']]
        result = wrapper.predict_action(examples=[example], unnorm_key=metadata['unnorm_key'],
                                        **metadata['request_options'])
        actions = np.asarray(result['actions'])
        if baseline is None:
            baseline = actions.copy()
        predicted = np.asarray(captured['spatial_predicted_xy'])[0, 0] * [320, 240]
        # The first output channel is red, so its source component is the
        # original block that becomes the first sorting target after swapping.
        expected = centers[order[0]]
        rows.append(dict(channel_order=list(order), original_color_becoming_red='rgb'[order[0]],
                         predicted_head_xy_pixels=predicted.tolist(), expected_red_center_pixels=expected.tolist(),
                         distance_to_red_center_pixels=float(np.linalg.norm(predicted-expected)),
                         nearest_original_color='rgb'[np.argmin(np.linalg.norm(centers-predicted, axis=-1))],
                         mean_action_change=float(np.abs(actions-baseline).mean()),
                         max_action_change=float(np.abs(actions-baseline).max()), actions=actions.tolist()))
    report = dict(checkpoint=str(args.checkpoint.resolve()), trace=str(args.trace.resolve()),
                  original_head_color_centers_pixels=centers.tolist(), cases=rows,
                  note='Synthetic color permutation of all policy views and history; robot state and geometry fixed. '
                       'Expected centers come from visible image components, never fed to the model. '
                       'Center distance diagnoses target identity, not exact TCP placement or task success.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
