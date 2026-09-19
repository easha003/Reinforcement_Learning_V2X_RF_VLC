"""Tests for centralized unit conversion and validation."""

import numpy as np
import pytest

from hybrid_v2x_rl.core.units import (
    angular_difference_rad,
    db_to_linear,
    dbm_to_w,
    linear_to_db,
    validate_probability,
    w_to_dbm,
    wrap_angle_positive_rad,
    wrap_angle_rad,
)


@pytest.mark.parametrize(
    ("power_dbm", "expected_w"),
    [(0.0, 1e-3), (30.0, 1.0), (-30.0, 1e-6)],
)
def test_dbm_to_w_reference_values(power_dbm: float, expected_w: float) -> None:
    assert dbm_to_w(power_dbm) == pytest.approx(expected_w)


def test_dbm_watt_round_trip_supports_arrays() -> None:
    source = np.array([-50.0, -3.5, 0.0, 23.0, 40.0])
    converted = dbm_to_w(source)
    assert isinstance(converted, np.ndarray)
    np.testing.assert_allclose(w_to_dbm(converted), source, atol=1e-12)


def test_db_linear_reference_values_and_round_trip() -> None:
    source = np.array([-20.0, 0.0, 3.0, 10.0])
    np.testing.assert_allclose(
        db_to_linear(source),
        np.array([0.01, 1.0, 10.0**0.3, 10.0]),
    )
    np.testing.assert_allclose(linear_to_db(db_to_linear(source)), source)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_logarithmic_conversions_reject_invalid_linear_input(
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        w_to_dbm(invalid)
    with pytest.raises(ValueError):
        linear_to_db(invalid)


def test_numeric_conversions_reject_booleans_and_text() -> None:
    with pytest.raises(TypeError):
        dbm_to_w(True)
    with pytest.raises(TypeError):
        db_to_linear("3")


def test_angle_wrapping_uses_documented_half_open_intervals() -> None:
    np.testing.assert_allclose(
        wrap_angle_rad(np.array([-3.0 * np.pi, -np.pi, 0.0, np.pi, 3.0 * np.pi])),
        np.array([-np.pi, -np.pi, 0.0, -np.pi, -np.pi]),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        wrap_angle_positive_rad(np.array([-2.0 * np.pi, -np.pi, 2.0 * np.pi])),
        np.array([0.0, np.pi, 0.0]),
        atol=1e-15,
    )


def test_angular_difference_crosses_branch_cut_on_short_path() -> None:
    result = angular_difference_rad(
        np.deg2rad(-179.0),
        np.deg2rad(179.0),
    )
    assert result == pytest.approx(np.deg2rad(2.0))


def test_probability_validation_accepts_boundaries_and_rejects_bad_data() -> None:
    np.testing.assert_array_equal(
        validate_probability(np.array([0.0, 0.2, 1.0])),
        np.array([0.0, 0.2, 1.0]),
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_probability(1.00001, name="miss_probability")
    with pytest.raises(ValueError, match="finite"):
        validate_probability(np.nan)
