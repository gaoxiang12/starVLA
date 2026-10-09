from pathlib import Path

from omegaconf import OmegaConf
import pytest

from scripts.feature_ablation_spec import (
    ALLOWED_CHANGES, VARIANTS, asset_contract, baseline_budget, make_config,
    sha256_file, validate_config,
)
from starVLA.training.recipe import apply_training_recipe


ROOT = Path(__file__).resolve().parents[1]
BASELINES = [
    (ROOT / "examples/LIBERO/train_files/starvla_gawm_l_libero_fixed_dino_c_160k.yaml",
     80000, 160000, 6419, 77028, 256760, 5, 2),
    (ROOT / "examples/Robotwin/train_files/starvla_gawm_l_robotwin_fixed_dino_temporal_c_12plus4.yaml",
     765120, 765120, 47820, 573840, 1912800, 4, 4),
]


@pytest.mark.parametrize("path,stop,complete,steps,stage1,period,batch,workers", BASELINES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_real_configs_change_only_layers_and_bookkeeping(
        tmp_path, path, stop, complete, steps, stage1, period, batch, workers, variant):
    original_bytes = path.read_bytes()
    baseline = OmegaConf.load(path)
    candidate = make_config(path, tmp_path / variant, variant, 32, max_steps=stop,
                            eval_steps=[stop])
    changes = validate_config(baseline, candidate)
    assert set(changes) <= ALLOWED_CHANGES
    assert path.read_bytes() == original_bytes
    assert candidate.framework.world_model.feat_layers == VARIANTS[variant]
    before, after = apply_training_recipe(baseline), apply_training_recipe(candidate)
    assert after.datasets == before.datasets
    assert after.datasets.vla_data.per_device_batch_size == batch
    assert after.datasets.vla_data.num_workers == workers
    assert after.framework.action_model == before.framework.action_model
    assert after.framework.lang_cond == before.framework.lang_cond
    budget = baseline_budget(candidate)
    assert budget["max_steps"] == stop
    assert baseline_budget(path)["max_steps"] == complete
    assert budget["steps_per_epoch"] == steps
    assert budget["stage1_steps"] == stage1
    assert budget["lr_scheduler_total_steps"] == period
    assert budget["stage_epochs"] == baseline_budget(path)["stage_epochs"]


@pytest.mark.parametrize("field,value", [
    ("framework.world_model.future_objective", "adapter_latent"),
    ("framework.world_model.residual_predictor_depth", 8),
    ("framework.world_model.dense_temporal_smoothness_weight", 0.3),
    ("framework.world_model.visual_tokens_per_view", 16),
    ("framework.lang_cond.type", "text"),
    ("datasets.vla_data.num_workers", 1),
    ("training_overrides.datasets.vla_data.per_device_batch_size", 10),
    ("training_overrides.trainer.stage_epochs", [12, 4]),
    ("training_overrides.trainer.scheduler_epochs", 20),
    # This raw field is replaced by the C recipe, but must still be rejected.
    ("trainer.learning_rate.base", 0.001),
    ("seed", 43),
    ("unknown", {}),
])
def test_guard_rejects_protected_fields_even_when_recipe_overrides_them(tmp_path, field, value):
    path = BASELINES[0][0]
    candidate = make_config(path, tmp_path / "late", "late_layers", 32, max_steps=80000)
    OmegaConf.update(candidate, field, value, force_add=True)
    with pytest.raises(ValueError, match="protected fields"):
        validate_config(path, candidate)


def test_relocated_vtt_requires_identical_bytes(tmp_path):
    base = OmegaConf.load(BASELINES[0][0])
    original = tmp_path / "original.json"
    original.write_text('{"vectors":{"franka:test":[1,2,3]}}')
    base.framework.lang_cond.task_vectors_path = str(original)
    relocated = tmp_path / "copied.json"
    relocated.write_bytes(original.read_bytes())
    config = make_config(base, tmp_path / "run", "last_layer", 32,
                         task_vectors_path=relocated)
    assert sha256_file(original) == sha256_file(relocated)
    assert "framework.lang_cond.task_vectors_path" in validate_config(base, config)
    relocated.write_text('{"vectors":{"franka:test":[1,2,4]}}')
    with pytest.raises(ValueError, match="reference SHA-256"):
        validate_config(base, config)


def test_invalid_budget_world_size_and_variant_are_rejected(tmp_path):
    base = BASELINES[0][0]
    with pytest.raises(ValueError, match="32 ranks"):
        make_config(base, tmp_path / "arm", "baseline", 40)
    with pytest.raises(ValueError, match="complete reference"):
        make_config(base, tmp_path / "arm", "baseline", 32, max_steps=160001)
    with pytest.raises(ValueError, match="Unknown feature"):
        make_config(base, tmp_path / "arm", "unknown", 32)
    with pytest.raises(ValueError, match="cannot exceed"):
        make_config(base, tmp_path / "arm", "baseline", 32, max_steps=80000, eval_steps=[80001])
    with pytest.raises(ValueError, match="positive integer"):
        make_config(base, tmp_path / "arm", "baseline", 32, max_steps=True)


def test_guard_rejects_wrong_layer_capacity_and_invalid_milestones(tmp_path):
    base = BASELINES[0][0]
    candidate = make_config(base, tmp_path / "arm", "baseline", 32, max_steps=80000)
    candidate.framework.world_model.feat_layers = [-1]
    with pytest.raises(ValueError, match="capacity-matched"):
        validate_config(base, candidate)
    candidate.framework.world_model.feat_layers = VARIANTS["baseline"]
    candidate.training_overrides.trainer.milestone_steps = [80000, 40000]
    with pytest.raises(ValueError, match="sorted unique"):
        validate_config(base, candidate)


def test_resolved_input_preserves_nominal_scheduler_and_stage_budget(tmp_path):
    resolved = apply_training_recipe(OmegaConf.load(BASELINES[0][0]))
    before = baseline_budget(resolved)
    candidate = make_config(resolved, tmp_path / "resolved", "wide_layers", 32, max_steps=80000)
    assert candidate.recipe_resolved
    assert candidate.trainer.max_train_steps == 80000
    assert baseline_budget(candidate)["lr_scheduler_total_steps"] == before["lr_scheduler_total_steps"]
    assert baseline_budget(candidate)["stage1_steps"] == before["stage1_steps"]
    validate_config(resolved, candidate)


@pytest.mark.parametrize("path", [row[0] for row in BASELINES])
def test_asset_contract_has_fixed_teacher_dimensions_and_small_asset_hashes(path, tmp_path):
    cfg = OmegaConf.load(path)
    encoder = tmp_path / "encoder"
    encoder.mkdir()
    (encoder / "config.json").write_text(
        '{"num_hidden_layers":24,"hidden_size":1024,"num_register_tokens":4,"patch_size":16}')
    cfg.framework.world_model.vision_encoder_path = str(encoder)
    vtt = tmp_path / "vtt.json"
    vtt.write_text('{"format_version":1,"split":"train","vectors":{}}')
    cfg.framework.lang_cond.task_vectors_path = str(vtt)
    data = cfg.datasets.vla_data
    if data.get("episode_exclusions_file"):
        exclusions = tmp_path / "exclusions.json"
        exclusions.write_text('{"excluded_episodes":{}}')
        data.episode_exclusions_file = str(exclusions)
    if data.dataset_py == "robotwin_official_hdf5":
        data.data_root_dir = str(tmp_path / "dataset")
        metadata = Path(data.data_root_dir) / "training_metadata"
        metadata.mkdir(parents=True)
        (metadata / "stat-500-all.json").write_text('{"action":{"mean":[0]}}')
    contract = asset_contract(cfg)
    expected_patches = 256 if "LIBERO" in str(path) else 300
    assert contract["teacher"]["feature_dim"] == 1024
    assert contract["teacher"]["patches_per_view"] == expected_patches
    assert contract["interface"]["prefix_tokens"] == 5
    assert contract["interface"]["tokens_per_view"] == 64
    assert len(contract["assets"]["vtt"]["sha256"]) == 64
