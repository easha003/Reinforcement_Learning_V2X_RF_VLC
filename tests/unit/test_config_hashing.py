from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import (
    HASH_EXCLUDED_PATHS,
    canonical_data,
    canonical_json,
    config_hash,
    hashed_data,
    headline_config_layers,
    load_config,
    load_headline_config,
)
from hybrid_v2x_rl.config.hashing import HASH_SCOPES, scope_hash, scoped_data
from hybrid_v2x_rl.core.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_serialization_ignores_mapping_insertion_order() -> None:
    first = {"outer": {"b": 2, "a": 1}, "value": -0.0}
    second = {"value": 0.0, "outer": {"a": 1, "b": 2}}

    assert canonical_json(first) == canonical_json(second)
    assert config_hash(first) == config_hash(second)
    assert json.loads(canonical_json(first)) == {
        "outer": {"a": 1, "b": 2},
        "value": 0.0,
    }


def test_hash_is_reproducible_and_sensitive_to_resolved_values() -> None:
    config = load_headline_config(PROJECT_ROOT)
    same_config = load_headline_config(PROJECT_ROOT)
    changed_project = config.project.model_copy(
        update={"description": "a deliberately changed description"}
    )
    changed = config.model_copy(update={"project": changed_project})

    digest = config_hash(config)
    assert digest == config_hash(same_config)
    assert len(digest) == 64
    assert digest != config_hash(changed)


def test_hash_is_portable_across_project_locations(tmp_path: Path) -> None:
    original = load_headline_config(PROJECT_ROOT)
    relocated = load_config(
        headline_config_layers(PROJECT_ROOT),
        project_root=tmp_path,
    )

    assert original.paths.project_root != relocated.paths.project_root
    assert config_hash(original) == config_hash(relocated)
    assert "${PROJECT_ROOT}/artifacts" in canonical_json(original)
    assert str(PROJECT_ROOT) not in canonical_json(original)


def test_split_membership_does_not_change_the_hash() -> None:
    """Adding a held-out trace moves no vehicle and must not restamp artifacts.

    Split membership is read at training and evaluation time only.  If it
    entered the digest, extending the test split would strand every already
    published trace under a hash nothing could reproduce.
    """

    config = load_headline_config(PROJECT_ROOT)
    extended_splits = config.environment.splits.model_copy(
        update={"test": (*config.environment.splits.test, "synthetic-d30-test-003")}
    )
    extended_environment = config.environment.model_copy(update={"splits": extended_splits})
    extended = config.model_copy(update={"environment": extended_environment})

    assert extended.environment.splits.test != config.environment.splits.test
    assert config_hash(extended) == config_hash(config)
    # The full serialization still carries the change; only the digest ignores it.
    assert canonical_json(extended) != canonical_json(config)
    assert "synthetic-d30-test-003" in canonical_json(extended)


def test_other_environment_fields_still_change_the_hash() -> None:
    """The exclusion is one field, not the whole environment section."""

    config = load_headline_config(PROJECT_ROOT)
    changed_environment = config.environment.model_copy(
        update={"matched_random_tapes": not config.environment.matched_random_tapes}
    )
    changed = config.model_copy(update={"environment": changed_environment})

    assert config_hash(changed) != config_hash(config)


def test_hashed_data_omits_only_the_declared_paths() -> None:
    config = load_headline_config(PROJECT_ROOT)
    digested = hashed_data(config)
    complete = canonical_data(config)

    assert HASH_EXCLUDED_PATHS == (("environment", "splits"),)
    assert "splits" not in digested["environment"]
    assert "splits" in complete["environment"]
    assert set(digested) == set(complete)
    assert set(digested["environment"]) == set(complete["environment"]) - {"splits"}


def test_excluded_path_absent_from_an_unrelated_mapping_is_a_noop() -> None:
    """Hashing a mapping shaped differently must not fail or silently prune."""

    assert hashed_data({"environment": 3}) == {"environment": 3}
    assert hashed_data({"other": {"splits": [1]}}) == {"other": {"splits": [1]}}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_canonical_serialization_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ConfigurationError, match="non-finite"):
        canonical_json({"value": value})


# -- scoped digests: what an artifact can actually depend on -------------------


def mobility_relevant_edit(data: dict) -> dict:
    edited = copy.deepcopy(data)
    edited["mobility"]["grid"]["lane_width_m"] = 3.6
    return edited


def test_an_observation_edit_moves_the_run_digest_but_not_the_mobility_one() -> None:
    """A receiver field cannot move a vehicle.

    Under a single global digest it would still strand every trace and force
    hours of recomputation to reproduce identical bytes.
    """

    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited["observation"]["features"] = [*base["observation"]["features"], "distance_to_junction_m"]

    assert config_hash(edited) != config_hash(base)
    assert scope_hash(edited, "mobility") == scope_hash(base, "mobility")


def test_a_channel_edit_does_not_strand_the_mobility_campaign() -> None:
    """The concrete reason this exists: §7.2's collision model is undefined,
    so implementing M3 will add configuration the generator never reads."""

    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited["rf"]["collision_model_placeholder"] = 1.0

    assert config_hash(edited) != config_hash(base)
    assert scope_hash(edited, "mobility") == scope_hash(base, "mobility")


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("mobility", "grid", None),
        ("geometry", "min_separation_m", 6.0),
        ("environment", "episode_duration_s", 45.0),
    ],
)
def test_anything_the_generator_reads_still_moves_the_mobility_digest(
    section: str, field: str, value: object
) -> None:
    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    if value is None:
        edited[section][field]["lane_width_m"] = 3.6
    else:
        edited[section][field] = value

    assert scope_hash(edited, "mobility") != scope_hash(base, "mobility")


def test_the_seed_the_generator_reads_still_moves_the_mobility_digest() -> None:
    """``training`` stays in scope wholesale because ``root_seed`` lives there.

    Carving out a subtree is where the exclusion list's guarantee would start
    to erode, so the conservative choice is kept.
    """

    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited["training"]["root_seed"] = base["training"]["root_seed"] + 1

    assert scope_hash(edited, "mobility") != scope_hash(base, "mobility")


def test_optimizer_changes_do_not_move_current_artifact_compatibility_scopes() -> None:
    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited["training"]["entropy_coefficient"] = 0.05
    for multiplier in edited["training"]["density_multipliers"]:
        multiplier["learning_rate"] = 5.0

    assert config_hash(edited) != config_hash(base)
    assert scope_hash(edited, "mobility_trace") == scope_hash(base, "mobility_trace")
    assert scope_hash(edited, "policy_environment") == scope_hash(
        base, "policy_environment"
    )


def test_current_artifact_scopes_keep_the_root_seed() -> None:
    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited["training"]["root_seed"] += 1

    for scope in ("mobility_trace", "policy_environment"):
        assert scope_hash(edited, scope) != scope_hash(base, scope)


def test_policy_environment_scope_is_stricter_than_trace_replay_scope() -> None:
    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited["rf"]["collision_model_placeholder"] = 1.0

    assert scope_hash(edited, "mobility_trace") == scope_hash(base, "mobility_trace")
    assert scope_hash(edited, "policy_environment") != scope_hash(
        base, "policy_environment"
    )


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("mobility", "speed_limit_mps", 12.0),
        ("geometry", "min_separation_m", 6.0),
    ],
)
def test_physical_trace_edits_move_both_current_compatibility_scopes(
    section: str, field: str, value: object
) -> None:
    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited[section][field] = value

    for scope in ("mobility_trace", "policy_environment"):
        assert scope_hash(edited, scope) != scope_hash(base, scope)


def test_a_scope_digest_still_honours_the_global_exclusions() -> None:
    base = canonical_data(load_headline_config(PROJECT_ROOT))
    edited = copy.deepcopy(base)
    edited["environment"]["splits"] = {"train": ["anything"], "validation": [], "test": []}

    assert scope_hash(edited, "mobility") == scope_hash(base, "mobility")


def test_an_unknown_scope_is_refused_rather_than_silently_global() -> None:
    config = load_headline_config(PROJECT_ROOT)
    with pytest.raises(ConfigurationError, match="unknown configuration hash scope"):
        scope_hash(config, "channel")


def test_every_declared_scope_is_narrower_than_the_run_digest() -> None:
    config = load_headline_config(PROJECT_ROOT)
    for scope in HASH_SCOPES:
        assert scoped_data(config, scope) != hashed_data(config), (
            f"scope {scope!r} excludes nothing and is therefore pointless"
        )
