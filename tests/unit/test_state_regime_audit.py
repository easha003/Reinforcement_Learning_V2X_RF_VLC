"""Causal regime definitions and RF-load counterfactuals."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand, headline_parameters
from hybrid_v2x_rl.core.policy_actions import ACTION_CONTRACT_VERSION, ActionResourceMap
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolError, RFPoolModel
from hybrid_v2x_rl.mean_field.state_regime_audit import (
    REGIME_NAMES,
    RegimeThresholds,
    ThresholdReservoir,
    classify_regimes,
)

COLUMNS = (
    "rf_channel_busy_ratio",
    "neighbor_count",
    "optical_fov_margin",
    "predicted_blockage_probability",
    "predictor_confidence",
    "track_age",
)


def _thresholds() -> RegimeThresholds:
    return RegimeThresholds(
        cbr_low=0.2,
        cbr_high=0.8,
        neighbor_low=20.0,
        neighbor_high=80.0,
        blockage_low=0.2,
        blockage_high=0.8,
        confidence_low=0.4,
        track_age_high=0.2,
        fit_rows_seen=100,
        fit_rows_retained=100,
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    (
        ((0.1, 10.0, 0.5, 0.1, 0.9, 0.05), "easy_state"),
        ((0.5, 50.0, 0.5, 0.5, 0.9, 0.05), "moderate_rf_conditions"),
        ((0.3, 30.0, -0.1, 0.9, 0.9, 0.05), "poor_vlc_usable_rf"),
        ((0.5, 50.0, 0.1, 0.5, 0.2, 0.05), "uncertain_mixed_state"),
        (
            (0.9, 90.0, 0.5, 0.1, 0.9, 0.05),
            "heavy_rf_contention_optical_permitted",
        ),
    ),
)
def test_each_declared_regime_has_an_operational_causal_definition(
    row: tuple[float, ...],
    expected: str,
) -> None:
    labels = classify_regimes(row, COLUMNS, _thresholds())

    assert expected in labels
    assert set(labels).issubset(REGIME_NAMES)


def test_threshold_reservoir_fits_only_retained_causal_rows() -> None:
    reservoir = ThresholdReservoir(maximum_rows=5, seed=7)
    for index in range(20):
        fraction = index / 20.0
        reservoir.observe(
            (fraction, float(index), 0.5, fraction, 1.0 - fraction, fraction / 10.0),
            COLUMNS,
        )

    thresholds = reservoir.fit()

    assert thresholds.fit_rows_seen == 20
    assert thresholds.fit_rows_retained == 5
    assert thresholds.cbr_low <= thresholds.cbr_high
    assert thresholds.neighbor_low <= thresholds.neighbor_high
    assert thresholds.blockage_low <= thresholds.blockage_high


def test_quantile_ties_do_not_label_the_central_mass_uncertain() -> None:
    labels = classify_regimes(
        (0.5, 50.0, 0.5, 0.5, 0.4, 0.2),
        COLUMNS,
        _thresholds(),
    )

    assert "uncertain_mixed_state" not in labels


def test_counterfactual_aggregate_rf_risk_responds_to_pressure() -> None:
    model = RFPoolModel(
        parameters=headline_parameters(SensitivityBand.NOMINAL),
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=0.0005,
    )

    light = model.counterfactual_attempt_failure_probability(
        active_pairs=100,
        offered_rf_attempts=1,
        decoding_failure_probability=0.01,
    )
    heavy = model.counterfactual_attempt_failure_probability(
        active_pairs=100,
        offered_rf_attempts=400,
        decoding_failure_probability=0.01,
    )

    assert 0.0 <= light < heavy <= 1.0
    with pytest.raises(RFPoolError, match="positive offered"):
        model.counterfactual_attempt_failure_probability(
            active_pairs=100,
            offered_rf_attempts=0,
            decoding_failure_probability=0.01,
        )


def test_resource_map_fixture_covers_the_frozen_nine_actions() -> None:
    resources = ActionResourceMap(
        contract_version=ACTION_CONTRACT_VERSION,
        rf_activation_cost=1.0,
        vlc_activation_cost=1.0,
    )

    costs = np.asarray([resources.activation_cost(index) for index in range(9)])

    assert costs.shape == (9,)
    assert costs[0] == pytest.approx(1.0)
    assert costs[-1] == pytest.approx(5.0)
