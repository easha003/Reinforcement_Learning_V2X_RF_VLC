"""Bounded propagation-tail declaration and decomposition invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.propagation_tail_diagnostic import (
    EXPECTED_DIMENSIONS,
    PROPAGATION_TAIL_RESULT_SCHEMA,
    PropagationTailKey,
    PropagationTailTally,
    load_propagation_tail_declaration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION = PROJECT_ROOT / "configs/evaluation/propagation_tail_decomposition.yaml"


def test_declaration_freezes_the_failed_headline_slice() -> None:
    declaration = load_propagation_tail_declaration(
        DECLARATION,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )

    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.densities == (20.0, 30.0)
    assert declaration.receive_profile_name == (
        "rx2-mrc__low-correlation-hardware-bound__short-cable-loss"
    )
    assert declaration.optical_configuration_names == (
        "wide-60deg",
        "concentrated-30deg",
    )
    assert declaration.dimensions == EXPECTED_DIMENSIONS
    assert declaration.output_path.name == "phase8_propagation_tail_decomposition.json"


def test_group_decomposition_closes_and_orders_by_risk_contribution() -> None:
    tally = PropagationTailTally()
    tally.begin_frame()
    low = PropagationTailKey("DUP-4", "actor-usable", "los", "available")
    high = PropagationTailKey(
        "RF-4",
        "contract-fallback",
        "nlos",
        "unavailable:occluded",
    )
    tally.observe(low, selected_risk=1e-8, rf_failure=0.01, vlc_failure=1e-4)
    tally.observe(high, selected_risk=1e-2, rf_failure=0.3, vlc_failure=1.0)

    payload = tally.as_dict(thresholds=(1e-6, 1e-4))

    assert payload["frames"] == 1
    assert payload["transitions"] == 2
    assert payload["mean_selected_risk"] == pytest.approx(0.005000005)
    groups = payload["groups_by_risk_contribution"]
    assert isinstance(groups, list)
    assert groups[0]["lower_bound_action"] == "RF-4"
    assert sum(row["transitions"] for row in groups) == 2
    assert sum(row["fraction_of_total_risk_sum"] for row in groups) == pytest.approx(1.0)
    tails = payload["risk_tails"]
    assert isinstance(tails, list)
    assert tails[0]["transitions_above"] == 1
    assert tails[0]["fraction_of_total_risk_sum_above"] == pytest.approx(1e-2 / (1e-2 + 1e-8))


def test_result_schema_name_is_versioned() -> None:
    assert PROPAGATION_TAIL_RESULT_SCHEMA.endswith(".v1")
