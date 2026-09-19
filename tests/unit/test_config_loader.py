from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hybrid_v2x_rl.config import (
    deep_merge,
    headline_config_layers,
    load_config,
    load_headline_config,
    load_yaml_file,
)
from hybrid_v2x_rl.core.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_deep_merge_is_recursive_ordered_and_nonmutating() -> None:
    base = {
        "project": {"name": "hybrid-rf-vlc-rl", "description": "base"},
        "values": [1, 2],
    }
    override = {
        "project": {"description": "override"},
        "values": [3],
        "new": True,
    }

    merged = deep_merge(base, override)

    assert merged == {
        "project": {"name": "hybrid-rf-vlc-rl", "description": "override"},
        "values": [3],
        "new": True,
    }
    assert base["project"]["description"] == "base"
    assert base["values"] == [1, 2]


def test_headline_loader_returns_frozen_resolved_configuration() -> None:
    config = load_headline_config(PROJECT_ROOT)

    assert config.schema_version == "1.0"
    assert config.service.payload_bytes == 300
    assert config.service.deadline_s == pytest.approx(0.003)
    assert config.service.miss_budget == pytest.approx(1e-4)
    assert config.mobility.target_densities_veh_per_lane_km == (
        10.0,
        20.0,
        30.0,
    )
    assert config.paths.project_root == PROJECT_ROOT.resolve()
    assert config.rf.calibration_artifact.is_absolute()
    assert config.vlc.pattern_artifact.is_absolute()
    with pytest.raises(ValidationError, match="Instance is frozen"):
        config.service.payload_bytes = 301  # type: ignore[misc]


def test_explicit_project_root_controls_all_relative_paths(tmp_path: Path) -> None:
    config = load_config(
        headline_config_layers(PROJECT_ROOT),
        project_root=tmp_path,
    )

    assert config.paths.project_root == tmp_path.resolve()
    assert config.paths.artifact_root == (tmp_path / "artifacts").resolve()
    assert (
        config.rf.calibration_artifact == (tmp_path / "artifacts/calibration/rf/mode2-v1").resolve()
    )


def test_later_yaml_layer_overrides_nested_value(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text(
        "project:\n  description: layered override\n",
        encoding="utf-8",
    )

    config = load_config(
        (*headline_config_layers(PROJECT_ROOT), override),
        project_root=PROJECT_ROOT,
    )

    assert config.project.name == "hybrid-rf-vlc-rl"
    assert config.project.description == "layered override"


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "service:\n  deadline_s: 0.001\n  deadline_s: 0.003\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="duplicate key"):
        load_yaml_file(duplicate)


def test_nonmapping_yaml_layer_is_rejected(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="top-level mapping"):
        load_yaml_file(sequence)


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    override = tmp_path / "unknown.yaml"
    override.write_text("service:\n  deadine_s: 0.001\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="deadine_s"):
        load_config(
            (*headline_config_layers(PROJECT_ROOT), override),
            project_root=PROJECT_ROOT,
        )


def test_secondary_ultra_reliability_layers_form_valid_configuration() -> None:
    config_root = PROJECT_ROOT / "configs"
    layers = list(headline_config_layers(PROJECT_ROOT))
    layers[2] = config_root / "service" / "ev2x_300B_3ms_1e-5.yaml"
    layers.insert(
        -1,
        config_root / "training" / "primal_dual_ppo_ultra.yaml",
    )
    layers[-1] = config_root / "evaluation" / "ultra_reliability.yaml"

    config = load_config(layers, project_root=PROJECT_ROOT)

    assert config.project.experiment == "ev2x_300B_3ms_1e-5"
    assert config.service.deadline_s == pytest.approx(0.003)
    assert config.service.miss_budget == pytest.approx(1e-5)
    assert config.training.curriculum[-1].miss_budget == pytest.approx(1e-5)
    assert config.evaluation.min_packets_per_policy_density == 10_000_000


def test_concentrated_receiver_layer_narrows_the_frontier_field_of_view() -> None:
    """The 10^-5 frontier runs a 30 deg semi-angle; the headline stays at 60.

    ``receiver_fov_deg`` is the receiver semi-angle.  Narrowing it raises
    concentrator gain as ``n^2/sin^2(psi_c)``, worth about 10 dB of electrical
    SNR, which is the headroom the ultra-reliability target needs.  Pinning
    both values here keeps the layer from silently going unused and keeps the
    headline from drifting onto the optimistic front end.
    """

    config_root = PROJECT_ROOT / "configs"
    layers = list(headline_config_layers(PROJECT_ROOT))
    layers[2] = config_root / "service" / "ev2x_300B_3ms_1e-5.yaml"
    layers.insert(
        layers.index(config_root / "channel" / "vlc_vehicle.yaml") + 1,
        config_root / "channel" / "vlc_vehicle_concentrated.yaml",
    )
    layers.insert(-1, config_root / "training" / "primal_dual_ppo_ultra.yaml")
    layers[-1] = config_root / "evaluation" / "ultra_reliability.yaml"

    frontier = load_config(layers, project_root=PROJECT_ROOT)
    headline = load_headline_config(PROJECT_ROOT)

    assert frontier.vlc.receiver_fov_deg == pytest.approx(30.0)
    assert headline.vlc.receiver_fov_deg == pytest.approx(60.0)
    assert frontier.service.miss_budget == pytest.approx(1e-5)
    # The override touches the receiver only; the rest of the optical front
    # end must be identical, or the sensitivity would confound two changes.
    assert frontier.vlc.electrical_bandwidth_hz == headline.vlc.electrical_bandwidth_hz
    assert frontier.vlc.modulation == headline.vlc.modulation
    assert frontier.vlc.timing == headline.vlc.timing


def test_shortened_deadline_is_rejected_by_the_channel_layers(tmp_path: Path) -> None:
    """Channel timing targets 3 ms; a tighter deadline must fail, not silently pass.

    The 1-ms profiles were removed, so nothing should be able to shorten the
    deadline without also replacing the channel layers.  A non-headline
    experiment name is used so the PHY invariant is reached instead of the
    frozen headline contract.
    """

    override = tmp_path / "tightened.yaml"
    override.write_text(
        "project:\n  experiment: tightened-deadline\nservice:\n  deadline_s: 0.001\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="VLC airtime"):
        load_config(
            (*headline_config_layers(PROJECT_ROOT), override),
            project_root=PROJECT_ROOT,
        )
