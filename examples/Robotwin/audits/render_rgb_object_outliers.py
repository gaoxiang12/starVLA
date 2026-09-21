"""Render existing localization errors using hash-verified validation loader images."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    torch.set_num_threads(4)
    report = json.loads(args.report.read_text())
    rows = sorted((r for r in report['rows'] if r['target_valid']),
                  key=lambda r: r['pixel_error'], reverse=True)[:6]
    hashes = {(r['episode'], r['anchor']): r['native_rgb_sha256']
              for r in report['frame_records']}
    cfg = OmegaConf.load(args.config)
    cfg.datasets.vla_data.episode_split = 'validation'
    data = get_vla_dataset(cfg.datasets.vla_data, mode='validation')
    child, = data.datasets
    child.transforms.eval()
    sheet = Image.new('RGB', (960, 560), '#202020')
    records = []
    args.output.mkdir(parents=True)
    for i, row in enumerate(rows):
        ep, anchor = row['episode'], row['anchor']
        assert ep in set(map(int, child.trajectory_ids))
        sample = child._pack_sample(child.transforms(child.get_step_data(ep, anchor)))
        native = np.asarray(sample['native_images'][0])
        digest = hashlib.sha256(native.tobytes()).hexdigest()
        assert digest == hashes[ep, anchor], (ep, anchor, 'native image mismatch')
        filename = f"episode{ep}_anchor{anchor}_{row['color']}.png"
        Image.fromarray(native).save(args.output / filename)
        panel = Image.new('RGB', (320, 280), '#202020')
        panel.paste(Image.fromarray(native), (0, 40))
        draw = ImageDraw.Draw(panel)
        draw.text((5, 2), f"ep{ep} frame{anchor} {row['color']} {row['pixel_error']:.2f}px", fill='white')
        draw.text((5, 17), 'yellow +: prediction / white circle: label', fill='white')
        x, y = row['target_xy']; y += 40
        draw.ellipse((x-5, y-5, x+5, y+5), outline='white', width=1)
        x, y = row['predicted_xy']; y += 40
        draw.line((x-5, y, x+5, y), fill='yellow', width=1)
        draw.line((x, y-5, x, y+5), fill='yellow', width=1)
        sheet.paste(panel, ((i % 3)*320, (i // 3)*280))
        records.append(dict(**row, native_rgb_sha256=digest, image=filename))
    sheet.save(args.output / 'contact_sheet.png')
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        source_report=str(args.report.resolve()),
        source_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        checkpoint_sha256=report['checkpoint_sha256'], rows=records,
        note='Six largest accepted-label errors. Exact source native RGB hashes verified; '
             'labels are visible color centroids, not independent ground truth.'), indent=2)+'\n')
    print(args.output / 'contact_sheet.png')


if __name__ == '__main__':
    main()
