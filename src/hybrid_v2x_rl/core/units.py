"""Canonical unit conversions and boundary validation.

Internal distances are metres, times are seconds, angles are radians,
frequencies are hertz, and powers are watts.  Decibel helpers are centralized
here so channel modules cannot drift onto subtly different conventions.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NumericInput: TypeAlias = npt.ArrayLike
NumericOutput: TypeAlias = float | npt.NDArray[np.float64]


def _finite_float_array(value: NumericInput, *, name: str) -> npt.NDArray[np.float64]:
    """Convert numeric input to float64 while rejecting booleans and non-finite data."""

    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise TypeError(f"{name} must contain real numeric values")
    result = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scalar_or_array(value: npt.NDArray[np.float64]) -> NumericOutput:
    """Return Python floats for scalar input and float64 arrays otherwise."""

    if value.ndim == 0:
        return float(value)
    return value


def _positive_finite_result(value: npt.NDArray[np.float64], *, conversion: str) -> None:
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(f"{conversion} produced a value outside float64 range")


def dbm_to_w(power_dbm: NumericInput) -> NumericOutput:
    """Convert dBm to watts.

    The conversion is ``P[W] = 10 ** (P[dBm] / 10) / 1000``.
    """

    values = _finite_float_array(power_dbm, name="power_dbm")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.power(10.0, values / 10.0) / 1000.0
    _positive_finite_result(converted, conversion="dBm-to-watt conversion")
    return _scalar_or_array(converted)


def w_to_dbm(power_w: NumericInput) -> NumericOutput:
    """Convert strictly positive power in watts to dBm."""

    values = _finite_float_array(power_w, name="power_w")
    if np.any(values <= 0.0):
        raise ValueError("power_w must be strictly positive")
    converted = 10.0 * np.log10(values * 1000.0)
    if not np.all(np.isfinite(converted)):
        raise ValueError("watt-to-dBm conversion produced a non-finite value")
    return _scalar_or_array(converted)


def db_to_linear(value_db: NumericInput) -> NumericOutput:
    """Convert a power ratio in decibels to a linear power ratio."""

    values = _finite_float_array(value_db, name="value_db")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.power(10.0, values / 10.0)
    _positive_finite_result(converted, conversion="dB-to-linear conversion")
    return _scalar_or_array(converted)


def linear_to_db(value_linear: NumericInput) -> NumericOutput:
    """Convert a strictly positive linear power ratio to decibels."""

    values = _finite_float_array(value_linear, name="value_linear")
    if np.any(values <= 0.0):
        raise ValueError("value_linear must be strictly positive")
    converted = 10.0 * np.log10(values)
    if not np.all(np.isfinite(converted)):
        raise ValueError("linear-to-dB conversion produced a non-finite value")
    return _scalar_or_array(converted)


def wrap_angle_rad(angle_rad: NumericInput) -> NumericOutput:
    """Wrap radians to the canonical half-open interval ``[-pi, pi)``."""

    values = _finite_float_array(angle_rad, name="angle_rad")
    wrapped = np.remainder(values + np.pi, 2.0 * np.pi) - np.pi
    return _scalar_or_array(wrapped)


def wrap_angle(angle_rad: NumericInput) -> NumericOutput:
    """Alias for :func:`wrap_angle_rad`."""

    return wrap_angle_rad(angle_rad)


def wrap_angle_positive_rad(angle_rad: NumericInput) -> NumericOutput:
    """Wrap radians to the half-open interval ``[0, 2*pi)``."""

    values = _finite_float_array(angle_rad, name="angle_rad")
    wrapped = np.remainder(values, 2.0 * np.pi)
    return _scalar_or_array(wrapped)


def angular_difference_rad(
    minuend_rad: NumericInput, subtrahend_rad: NumericInput
) -> NumericOutput:
    """Return the signed shortest angular difference in ``[-pi, pi)``."""

    minuend = _finite_float_array(minuend_rad, name="minuend_rad")
    subtrahend = _finite_float_array(subtrahend_rad, name="subtrahend_rad")
    try:
        difference = np.subtract(minuend, subtrahend)
    except ValueError as error:
        raise ValueError("angle inputs are not broadcast-compatible") from error
    return wrap_angle_rad(difference)


def validate_probability(probability: NumericInput, *, name: str = "probability") -> NumericOutput:
    """Validate and return scalar or array probabilities in ``[0, 1]``."""

    if not isinstance(name, str) or not name:
        raise ValueError("probability name must be a non-empty string")
    values = _finite_float_array(probability, name=name)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} must be within [0, 1]")
    return _scalar_or_array(values)


__all__ = [
    "NumericInput",
    "NumericOutput",
    "angular_difference_rad",
    "db_to_linear",
    "dbm_to_w",
    "linear_to_db",
    "validate_probability",
    "w_to_dbm",
    "wrap_angle",
    "wrap_angle_positive_rad",
    "wrap_angle_rad",
]
