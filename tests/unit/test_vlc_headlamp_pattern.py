"""Headlamp radiant intensity.

The load-bearing test here is the one asserting that a missing measured pattern
*raises* rather than falling back to a cosine lobe. W17's conclusion is that
Lambertian results should be stress-tested against non-Lambertian patterns
before a V2V claim is made; a silent fallback would produce exactly the claim
W17 warns against and would look identical to a real result in every log.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.vlc.headlamp_pattern import (
    PATTERN_SCHEMA,
    HeadlampPattern,
    HeadlampPatternError,
    LambertianPattern,
    TabulatedPattern,
    load_pattern,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_artifact(root: Path, **overrides) -> Path:
    payload = {
        "schema": PATTERN_SCHEMA,
        "pattern_id": "headlamp-v1",
        "units": "W/sr",
        "source": "synthetic fixture, not a measurement",
        "horizontal_angles_rad": [-0.5, 0.0, 0.5],
        "vertical_angles_rad": [-0.2, 0.0, 0.2],
        "intensity_w_per_sr": [
            [1.0, 2.0, 1.0],
            [3.0, 10.0, 4.0],
            [1.0, 2.0, 1.0],
        ],
    }
    payload.update(overrides)
    path = root / "manifest.json"
    path.write_text(json.dumps(payload))
    return root


# -- the rule that matters ----------------------------------------------------


def test_a_missing_measured_pattern_raises_rather_than_falling_back(tmp_path: Path) -> None:
    """Implementation spec 12.1, and the reason it is a rule.

    The headline configuration declares ``measured_non_lambertian`` and names
    ``artifacts/calibration/vlc/headlamp-v1``, which does not exist. That must
    be a loud failure. A result computed from a cosine lobe while the manifest
    records a measured beam is indistinguishable from a real one afterwards.
    """

    with pytest.raises(HeadlampPatternError, match="no Lambertian"):
        load_pattern(tmp_path / "absent")


def test_the_configured_artifact_exists_and_is_a_declared_envelope() -> None:
    """Whatever the profile points at must be present and must declare itself.

    This named ``headlamp-ece-r112-v1`` until 2026-09-01, which was the
    configured pattern when the test was written and is now the superseded one.
    The .gitignore keeps that artifact out of the repository deliberately -- "so
    nobody points a profile at it by accident" -- so the assertion passed only
    on machines still carrying a stale local copy and failed on every clean
    clone. Reading the path from the configuration instead of naming one is what
    stops that recurring.

    The distinction the assertion defends is unchanged: these patterns are
    regulatory constructions, not measured lamps. A compliant beam is
    constrained by R112, not described by it, so any optical result quoted from
    one inherits that.
    """

    from hybrid_v2x_rl.config import load_headline_config

    config = load_headline_config(PROJECT_ROOT)
    configured = PROJECT_ROOT / config.vlc.pattern_artifact
    pattern = load_pattern(configured)

    assert pattern.pattern_id == configured.name
    assert "not measured" in pattern.source.lower()


def test_lambertian_is_available_only_by_name() -> None:
    """It exists as a comparison model, and nothing constructs it implicitly."""

    pattern = LambertianPattern(peak_intensity_w_per_sr=100.0,
                                half_power_semi_angle_rad=math.radians(15.0))
    assert pattern.pattern_id == "lambertian_comparison"
    assert isinstance(pattern, HeadlampPattern)


# -- photometric units --------------------------------------------------------


def test_a_photometric_pattern_is_refused(tmp_path: Path) -> None:
    """Candela weights by human visual response; a photodiode has none."""

    write_artifact(tmp_path, units="cd")
    with pytest.raises(HeadlampPatternError, match="radiometric"):
        load_pattern(tmp_path)


def test_an_unknown_schema_is_refused(tmp_path: Path) -> None:
    write_artifact(tmp_path, schema="something-else")
    with pytest.raises(HeadlampPatternError, match="schema"):
        load_pattern(tmp_path)


# -- the Lambertian comparison model -----------------------------------------


def test_lambertian_order_follows_the_half_power_angle() -> None:
    """m = -ln2 / ln(cos(psi)), so a 60 deg half-power angle gives m = 1."""

    pattern = LambertianPattern(1.0, math.radians(60.0))
    assert pattern.lambertian_order == pytest.approx(1.0, abs=1e-9)


def test_lambertian_is_half_power_at_its_half_power_angle() -> None:
    for degrees in (10.0, 30.0, 60.0):
        pattern = LambertianPattern(100.0, math.radians(degrees))
        on_axis = pattern.radiant_intensity_w_per_sr(0.0, 0.0)
        at_angle = pattern.radiant_intensity_w_per_sr(math.radians(degrees), 0.0)
        assert at_angle == pytest.approx(0.5 * on_axis, rel=1e-9)


def test_lambertian_is_rotationally_symmetric_which_is_the_idealization() -> None:
    """A real beam is asymmetric horizontally because it is aimed away from
    oncoming traffic. This model cannot represent that, by construction."""

    pattern = LambertianPattern(100.0, math.radians(20.0))
    angle = math.radians(10.0)
    horizontal = pattern.radiant_intensity_w_per_sr(angle, 0.0)
    vertical = pattern.radiant_intensity_w_per_sr(0.0, angle)
    mirrored = pattern.radiant_intensity_w_per_sr(-angle, 0.0)
    assert horizontal == pytest.approx(vertical)
    assert horizontal == pytest.approx(mirrored)


def test_lambertian_vanishes_beyond_ninety_degrees() -> None:
    pattern = LambertianPattern(100.0, math.radians(20.0))
    assert pattern.radiant_intensity_w_per_sr(math.radians(95.0), 0.0) == 0.0


def test_an_invalid_lambertian_is_refused() -> None:
    with pytest.raises(HeadlampPatternError, match="semi-angle"):
        LambertianPattern(1.0, math.radians(95.0))
    with pytest.raises(HeadlampPatternError, match="peak intensity"):
        LambertianPattern(0.0, math.radians(20.0))


# -- the tabulated pattern ----------------------------------------------------


def test_a_measured_pattern_round_trips_through_its_artifact(tmp_path: Path) -> None:
    write_artifact(tmp_path)
    pattern = load_pattern(tmp_path)
    assert pattern.pattern_id == "headlamp-v1"
    assert pattern.radiant_intensity_w_per_sr(0.0, 0.0) == pytest.approx(10.0)
    assert pattern.radiant_intensity_w_per_sr(-0.5, -0.2) == pytest.approx(1.0)


def test_a_measured_pattern_can_be_horizontally_asymmetric(tmp_path: Path) -> None:
    """The property the Lambertian comparison cannot express."""

    write_artifact(tmp_path)
    pattern = load_pattern(tmp_path)
    left = pattern.radiant_intensity_w_per_sr(-0.5, 0.0)
    right = pattern.radiant_intensity_w_per_sr(0.5, 0.0)
    assert left != right


def test_interpolation_is_bilinear(tmp_path: Path) -> None:
    write_artifact(tmp_path)
    pattern = load_pattern(tmp_path)
    # Midway between (0.0, 0.0)=10 and (0.5, 0.0)=4 on the horizontal axis.
    assert pattern.radiant_intensity_w_per_sr(0.25, 0.0) == pytest.approx(7.0)


def test_a_bearing_outside_the_measured_envelope_is_refused(tmp_path: Path) -> None:
    """Clamping would report the beam's rim intensity for a direction the lamp
    may not illuminate at all -- which is the junction geometry that decides
    the contribution."""

    write_artifact(tmp_path)
    pattern = load_pattern(tmp_path)
    with pytest.raises(HeadlampPatternError, match="outside the measured envelope"):
        pattern.radiant_intensity_w_per_sr(1.0, 0.0)
    with pytest.raises(HeadlampPatternError, match="vertical"):
        pattern.radiant_intensity_w_per_sr(0.0, 0.9)


def test_a_malformed_table_is_refused() -> None:
    with pytest.raises(HeadlampPatternError, match="ascending"):
        TabulatedPattern("bad", (0.5, -0.5), (0.0, 1.0), ((1.0, 1.0), (1.0, 1.0)), "x")
    with pytest.raises(HeadlampPatternError, match="columns"):
        TabulatedPattern("bad", (0.0, 1.0), (0.0, 1.0), ((1.0,), (1.0,)), "x")
    with pytest.raises(HeadlampPatternError, match="negative"):
        TabulatedPattern("bad", (0.0, 1.0), (0.0, 1.0), ((1.0, -1.0), (1.0, 1.0)), "x")
