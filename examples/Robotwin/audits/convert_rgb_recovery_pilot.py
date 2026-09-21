"""Convert the verified training-source recovery clip into an isolated dataset."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from examples.Robotwin.data_preparation import convert_extracted
from examples.Robotwin.audits.build_rgb_focus_labels import build
from starVLA.task_language import canonical_task_text

ROOT = Path(__file__).resolve().parents[3]
CAMP = ROOT / 'playground/Checkpoints/gawm_rgb_recovery_tabletop_pilot_20260908'
OUTPUT = Path('/data/gaoxiang/RoboTwinRecoveryPilot_20260908/RoboTwinGenerated/Clean/blocks_ranking_rgb')


def main():
    result = json.loads((CAMP / 'cases/source_000003/result.json').read_text())
    assert result['state'] == 'raw_clip_verified' and result['expert_assisted_success']
    raw_file = Path(result['hdf5'])
    assert hashlib.sha256(raw_file.read_bytes()).hexdigest() == result['hdf5_sha256']
    sources = json.loads((ROOT / 'examples/Robotwin/audits/rgb_recovery_train_sources_20260908.json').read_text())
    split = json.loads(Path(sources['split']).read_text())
    assert result['source_episode'] in split['train_episode_ids']
    assert result['source_episode'] not in split['validation_episode_ids']
    assert not OUTPUT.exists() and not OUTPUT.with_name('.blocks_ranking_rgb.tmp-convert').exists()
    raw_root = raw_file.parent.parent
    instructions = raw_root / 'instructions/episode0.json'
    instructions.parent.mkdir(exist_ok=True)
    instruction = {'seen': [canonical_task_text('blocks_ranking_rgb')]}
    if instructions.exists():
        assert json.loads(instructions.read_text()) == instruction
    else:
        instructions.write_text(json.dumps(instruction) + '\n')
    convert_extracted(raw_root, OUTPUT, ROOT / 'examples/Robotwin/train_files/modality.json',
                      task_name='blocks_ranking_rgb')
    audit_path = OUTPUT / 'meta/audit/conversion_audit.json'
    audit = json.loads(audit_path.read_text())
    audit.pop('control_hz')
    audit.update(state_semantics='joint drive targets, not measured joint positions',
                 action_semantics='same-record absolute joint drive targets; continuous_next loader applies +1',
                 video_storage_fps=30, physics_timestep_s=.004, save_freq_physics_steps=15,
                 time_note='Recorded-frame offsets follow original dataset convention; phase boundaries may duplicate frames. Storage FPS is not a measured physical control rate.',
                 source_episode=result['source_episode'], source_seed=result['scene_seed'],
                 source_hdf5_sha256=result['hdf5_sha256'],
                 policy_prefix_included_as_targets=False, training_enabled=False,
                 normalization_note='Dataset statistics describe this clip only; smoke/training must retain source model normalization statistics.',
                 split_note='Local episode 0 derives from original TRAIN episode 3. No local validation set; use original scene-safe validation separately.')
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    labels = CAMP / 'converted_labels'
    assert not labels.exists()
    build(SimpleNamespace(dataset=OUTPUT, raw=raw_file.parent, output=labels))
    provenance = dict(state='converted_labels_ready', dataset=str(OUTPUT), labels=str(labels),
                      train_episode_ids=[0], validation_episode_ids=[], excluded_episode_ids=[],
                      source_mapping=[dict(episode_index=0, source_episode=result['source_episode'],
                                           scene_seed=result['scene_seed'])],
                      training_enabled=False, loader_audit_complete=False,
                      note='Train-only recovery supplement, not a standalone validation split or unified pretraining mixture.')
    (CAMP / 'converted_dataset.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
