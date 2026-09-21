# VLA 2–5 candidate preparation

This directory contains **candidate-only** integration artifacts for the
datasets selected in `vla_hf_dataset_value_audit_20260825.md`. Nothing here is
auto-discovered by the production data registry, and none of these datasets is
part of the formal unified mixture yet.

The preparation contract is:

1. preserve the source v3 metadata and a self-contained Parquet/video sample;
2. convert the selected source to the loader-supported LeRobot layout;
3. generate `episodes.jsonl`, `tasks.jsonl`, `modality.json`, `stats.json`, and
   `stats_gr00t.json` from filtered training episodes;
4. run schema, task-language, video, finite-value, normalization, mask, and
   single-dataset smoke audits;
5. only then move a candidate config into `train_files/data_registry/` and add
   its head/weight to the production YAML.

`candidate_specs.yaml` is the source of truth for provisional robot/action/state
semantics. The JSON files are loader-shaped modality drafts. Their filenames
intentionally do not equal `modality.json`, so they cannot be mistaken for
accepted dataset metadata.

Important decisions:

- Molmo YAM and ABC YAM remain separate action specs until units, joint order,
  gripper convention, and command semantics are proven identical on a
  representative sample. Equal 14-D shapes are not sufficient.
- ABC's 14-D raw joint representation is the default action-head candidate.
  Its 20-D Cartesian representation remains available for a later, separate
  Cartesian head.
- HIW uses the 23-D WBC state/action pair. Its 29-D joint state is retained as
  auxiliary source data and must not be concatenated into the 23-D WBC head.
- Trigger/squeeze fields in HIW are not declared as binary grippers until their
  range and control semantics are verified.

Local staging and audit outputs live under
`/data/gaoxiang/vla_dataset_staging/`; large full downloads live directly under
`/data/gaoxiang/`.
