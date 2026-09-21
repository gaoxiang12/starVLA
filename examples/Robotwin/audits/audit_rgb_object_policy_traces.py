"""Check learned object positions on a frozen snapshot of actual policy inputs."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from starVLA.dataloader.rgb_object_supervision import COLORS, candidates
from starVLA.model.modules.rgb_object_readout import RGBObjectReadout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    # Freeze the file list before reading: later policy calls belong to a later audit.
    paths = sorted(args.trace_root.rglob('*.npz'))
    if not paths:
        raise ValueError('No recorded policy inputs')
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    prefix = 'spatial_focus.object_readout.'
    state = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    model = RGBObjectReadout(state['identity.weight'].shape[1]).eval()
    model.load_state_dict(state, strict=True)
    del state
    rows, frames, panels = [], [], []
    for path in paths:
        raw = path.read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as trace:
            metadata = json.loads(str(trace['metadata']))
            assert Path(metadata['server_metadata']['ckpt_path']).resolve() == args.checkpoint.resolve()
            assert metadata['view_order'] == ['head_camera', 'left_camera', 'right_camera']
            native = trace['native_images'].copy()
        assert native.shape == (3, 240, 320, 3) and native.dtype == np.uint8
        valid = torch.ones(1, dtype=torch.bool)
        tensor, arrays = model.head_images([dict(native_images=native)], valid, 'cpu')
        with torch.inference_mode():
            _, pred = model.encode(tensor, valid)
        xy = pred['xy'][0].numpy() * [320, 240] - .5
        assert np.isfinite(xy).all()
        key = dict(trial=path.parent.name, step=int(path.stem.removeprefix('step_')))
        frames.append(dict(**key, path=str(path.resolve()), trace_sha256=hashlib.sha256(raw).hexdigest(),
                           native_head_sha256=hashlib.sha256(arrays[0].tobytes()).hexdigest()))
        panel = Image.new('RGB', (320, 280), '#202020')
        panel.paste(Image.fromarray(arrays[0]), (0, 40))
        draw = ImageDraw.Draw(panel)
        draw.text((5, 2), f"{key['trial']} action step {key['step']}", fill='white')
        draw.text((5, 17), 'color +: prediction / white circle: label', fill='white')
        for i, proposal in enumerate(candidates(arrays[0])):
            accepted = bool(proposal['accepted'])
            target = np.asarray(proposal['center_xy']) if accepted else None
            rows.append(dict(**key, color=COLORS[i], target_valid=accepted,
                predicted_xy=xy[i].tolist(), target_xy=target.tolist() if accepted else None,
                pixel_error=float(np.linalg.norm(xy[i]-target)) if accepted else None,
                candidate_area=proposal['area'], candidate_dominance=proposal['dominance'],
                predicted_visible=float(pred['visibility'][0, i].sigmoid())))
            if accepted:
                x, y = target; y += 40
                draw.ellipse((x-5, y-5, x+5, y+5), outline='white')
            x, y = xy[i]; y += 40
            color = ['#ff4040', '#00ff00', '#4488ff'][i]
            draw.line((x-5, y, x+5, y), fill=color, width=1)
            draw.line((x, y-5, x, y+5), fill=color, width=1)
        panels.append(panel)
    def summarize(group):
        accepted = [r for r in group if r['target_valid']]
        errors = np.asarray([r['pixel_error'] for r in accepted])
        return dict(targets=len(group), accepted_targets=len(accepted),
                    mean_pixel_error=float(errors.mean()) if len(errors) else None,
                    over_10px=int((errors > 10).sum()))
    report = dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        input_manifest_sha256=hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest(),
        frames=frames, rows=rows, summary=summarize(rows),
        by_area={f'{lo}_{hi}': summarize([r for r in rows if lo <= r['candidate_area'] < hi])
                 for lo, hi in [(0, 12), (12, 30), (30, 60), (60, 120), (120, 100000)]},
        note='Snapshot of available sparse policy traces, not a complete or unbiased rollout sample. '
             'CPU float32 branch-only predictions, not exact CUDA policy/action replay. '
             'Heuristic visible color centers are diagnostic targets only; they never enter policy inference. '
             'Not independent object ground truth, amodal centers, task success, or causal ablation.')
    args.output.mkdir(parents=True)
    for start in range(0, len(panels), 6):
        sheet = Image.new('RGB', (960, 560), '#202020')
        for i, panel in enumerate(panels[start:start+6]):
            sheet.paste(panel, ((i % 3)*320, (i // 3)*280))
        sheet.save(args.output / f'contact_sheet_{start//6:02d}.png')
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(frames=len(frames), summary=report['summary'], by_area=report['by_area']), indent=2))


if __name__ == '__main__':
    main()
