"""Compare B's reader against A on every task, including episode ends."""
import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from examples.LiLaWAM.train_official_robotwin import load_dataset_module
from examples.LiLaWAM.gawm_official import GAWMOfficialDataset, OfficialNormalizer
from examples.LiLaWAM.official_robotwin_data import write_json, digest
from starVLA.model.framework.WM4A.LiLaWAM import upstream_models
from starVLA.task_language import resolve_task_language


def audit(config, source, output):
    module = load_dataset_module(source)
    cfg = OmegaConf.load(config)
    dataset = module.create_dataset(cfg, val=False)
    candidate = GAWMOfficialDataset(dataset, module)
    norm = OfficialNormalizer(source / 'utils/stat-500-all.json')
    reference_norm = upstream_models(source).VLAWrapper._normalize_tensor
    seen, rows = set(), []
    inventory = hashlib.sha256()
    for ep in dataset.episode_metadata:
        inventory.update(json.dumps(ep, sort_keys=True).encode())
        task = ep['task_name']
        if task in seen:
            continue
        seen.add(task)
        for anchor in (0, ep['length'] // 2, ep['length'] - 1):
            idx = int(dataset.valid_indices.searchsorted(ep['global_start'] + anchor))
            a, b = dataset[idx], candidate[idx]
            for key in ('state', 'action_sequence', 'state_mask', 'action_mask'):
                if not torch.equal(a[key], b[key]):
                    raise AssertionError((task, anchor, key))
            assert torch.equal(a['pixel_values'], b['pixel_values'][0]), (task, anchor, 'current')
            assert torch.equal(a['future_pixel_values'], b['pixel_values'][2]), (task, anchor, 'future32')
            for field, key in (('state', 'state'), ('action_sequence', 'action')):
                for dtype in (torch.float32, torch.bfloat16):
                    x = b[field].to(dtype)
                    lo = torch.tensor(norm.stats[key]['min'])
                    hi = torch.tensor(norm.stats[key]['max'])
                    assert torch.equal(norm.normalize(x, key), reference_norm(None, x, lo, hi))
            if anchor == ep['length'] - 1:
                assert torch.equal(b['pixel_values'][0], b['pixel_values'][1])
                assert torch.equal(b['pixel_values'][1], b['pixel_values'][2])
            rows.append(dict(task=task, anchor=anchor, episode_length=ep['length'],
                             canonical_text=resolve_task_language(None, task, 'dataset_name')))
    assert len(seen) == 50 and len(candidate) == 6120962 and len(candidate.all_episodes) == 27071
    result = dict(status='passed', tasks=len(seen), episodes=len(candidate.all_episodes), frames=len(candidate),
        samples=len(rows), raw_state_action_and_current_future32_pixels='bitwise_equal',
        normalization_fp32_bf16='bitwise_equal_to_upstream', tail='same_endpoint_clamp',
        inventory_sha256=inventory.hexdigest(), loader_sha256=digest(source/'dataloader/dataset.py'),
        stats_sha256=digest(source/'utils/stat-500-all.json'), rows=rows)
    write_json(output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.config, args.source, args.output)
