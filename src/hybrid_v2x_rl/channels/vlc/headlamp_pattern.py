"""Automotive headlamp radiant intensity, from a measured pattern.

Implementation spec section 12.1, and one rule in it does the work:

    A Lambertian pattern may exist only as an explicitly named comparison
    model.

So :class:`LambertianPattern` exists here, and nothing will ever fall back to
it. Loading a configuration that declares ``measured_non_lambertian`` while its
artifact is absent **raises**, rather than substituting a cosine lobe and
carrying on. The reason is W17: automotive headlamps are asymmetric,
manufacturer- and function-specific, with a hot spot and a legally mandated
cutoff, and W17's own conclusion is that Lambertian results "should be
stress-tested against non-Lambertian patterns before claiming" a V2V finding. A
silent fallback would produce exactly the claim W17 warns against, and it would
look identical to a real result in every log.

**Angle convention, fixed here so it cannot drift.** Both angles are measured
from the lamp's optical axis in the vehicle frame:

* ``horizontal_angle_rad`` is positive toward the vehicle's right, and is the
  axis along which a real beam is asymmetric because of traffic-side aiming;
* ``vertical_angle_rad`` is positive upward, and is the axis carrying the
  cutoff.

The optical link in this project runs headlamp to rear photodiode along a road,
so it lives near the horizontal plane and samples the beam close to its
brightest region. That is convenient and it is also why the pattern matters:
the same geometry at a junction swings the bearing tens of degrees off axis,
where a measured beam and a cosine lobe disagree most.

**Units are watts per steradian**, not candela and not lux. The distinction is
not pedantic here: photometric units weight by human visual response, and a
photodiode does not have one. A pattern tabulated in candela must be converted
before it becomes an artifact, and the manifest records which was done.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from hybrid_v2x_rl.core.errors import HybridV2XError

#: Artifact schema this loader understands.
# Versioned schema for the copied calibration payloads in this repository.
PATTERN_SCHEMA = "hybrid-rf-vlc-rl.vlc.headlamp.v1"


class HeadlampPatternError(HybridV2XError):
    """A headlamp pattern could not be loaded, or was sampled out of range."""


@runtime_checkable
class HeadlampPattern(Protocol):
    """Radiant intensity as a function of bearing from the optical axis."""

    @property
    def pattern_id(self) -> str:
        """Identity carried into every manifest that depends on this pattern."""

    def radiant_intensity_w_per_sr(
        self, horizontal_angle_rad: float, vertical_angle_rad: float
    ) -> float: ...

    def covers(self, horizontal_angle_rad: float, vertical_angle_rad: float) -> bool:
        """Whether this pattern can answer for that direction at all.

        The escape hatch that keeps :meth:`radiant_intensity_w_per_sr`'s refusal
        to extrapolate strict. A caller that must handle a direction the lamp
        does not illuminate asks this first and treats a ``False`` as *no
        light*, so the exception stays reserved for a genuine bug rather than
        becoming a routine control-flow signal that someone eventually softens
        into a clamp.
        """
        ...


@dataclass(frozen=True, slots=True)
class LambertianPattern:
    """A generalized Lambertian lobe -- **a comparison model, never a default**.

    ``I(phi) = I_0 cos^m(phi)`` with ``m = -ln 2 / ln(cos(half_angle))``. Kept
    because a reviewer is entitled to see what the idealization would have
    given, and because the difference between it and a measured beam is a
    result worth reporting rather than an embarrassment to hide.

    It must be selected by name. Nothing constructs it as a fallback.
    """

    peak_intensity_w_per_sr: float
    half_power_semi_angle_rad: float
    pattern_id: str = "lambertian_comparison"

    def __post_init__(self) -> None:
        if self.peak_intensity_w_per_sr <= 0.0:
            raise HeadlampPatternError("peak intensity must be positive",
                                       context={"value": self.peak_intensity_w_per_sr})
        if not 0.0 < self.half_power_semi_angle_rad < 0.5 * math.pi:
            raise HeadlampPatternError(
                "half-power semi-angle must lie strictly between 0 and 90 degrees",
                context={"radians": self.half_power_semi_angle_rad},
            )

    @property
    def lambertian_order(self) -> float:
        return -math.log(2.0) / math.log(math.cos(self.half_power_semi_angle_rad))

    def radiant_intensity_w_per_sr(
        self, horizontal_angle_rad: float, vertical_angle_rad: float
    ) -> float:
        """Rotationally symmetric, so only the total off-axis angle matters.

        That symmetry is the idealization: a real beam is asymmetric in the
        horizontal axis by design, because it is aimed away from oncoming
        traffic.
        """

        off_axis = math.hypot(horizontal_angle_rad, vertical_angle_rad)
        if off_axis >= 0.5 * math.pi:
            return 0.0
        return float(
            self.peak_intensity_w_per_sr * math.cos(off_axis) ** self.lambertian_order
        )

    def covers(self, horizontal_angle_rad: float, vertical_angle_rad: float) -> bool:
        """Everywhere in the forward hemisphere, which is the idealization.

        A closed form has an answer for every direction, and that is exactly the
        difference this class exists to expose: the measured beam has an
        envelope and this does not.
        """

        return math.hypot(horizontal_angle_rad, vertical_angle_rad) < 0.5 * math.pi


@dataclass(frozen=True, slots=True)
class TabulatedPattern:
    """A measured beam on a regular grid, bilinearly interpolated.

    Extrapolation is refused rather than clamped. A bearing outside the
    measured envelope is a question the artifact cannot answer, and returning
    the edge value would report the beam's rim intensity for a direction the
    lamp may not illuminate at all -- which is precisely the junction geometry
    where the optical link's behaviour decides the contribution.
    """

    pattern_id: str
    horizontal_angles_rad: tuple[float, ...]
    vertical_angles_rad: tuple[float, ...]
    #: ``intensity[vertical_index][horizontal_index]``, W/sr.
    intensity_w_per_sr: tuple[tuple[float, ...], ...]
    source: str

    def __post_init__(self) -> None:
        if len(self.horizontal_angles_rad) < 2 or len(self.vertical_angles_rad) < 2:
            raise HeadlampPatternError(
                "a tabulated pattern needs at least two samples on each axis",
                context={"pattern_id": self.pattern_id},
            )
        for name in ("horizontal_angles_rad", "vertical_angles_rad"):
            axis = getattr(self, name)
            if list(axis) != sorted(axis):
                raise HeadlampPatternError(f"{name} must be ascending",
                                           context={"pattern_id": self.pattern_id})
        if len(self.intensity_w_per_sr) != len(self.vertical_angles_rad):
            raise HeadlampPatternError("intensity rows must match the vertical axis",
                                       context={"pattern_id": self.pattern_id})
        for row in self.intensity_w_per_sr:
            if len(row) != len(self.horizontal_angles_rad):
                raise HeadlampPatternError(
                    "intensity columns must match the horizontal axis",
                    context={"pattern_id": self.pattern_id},
                )
            if any(value < 0.0 for value in row):
                raise HeadlampPatternError("radiant intensity cannot be negative",
                                           context={"pattern_id": self.pattern_id})

    def covers(self, horizontal_angle_rad: float, vertical_angle_rad: float) -> bool:
        """Whether both angles fall inside the measured grid.

        The horizontal envelope is the narrow one: the ECE R112 low-beam test
        points span +/-9 degrees, so a pair more than that off the transmitter's
        heading is a direction the artifact cannot answer for. Nothing here
        decides what a caller should do about it -- see the alignment
        short-circuit in :mod:`hybrid_v2x_rl.channels.vlc.model` for that.
        """

        return (
            math.isfinite(horizontal_angle_rad)
            and math.isfinite(vertical_angle_rad)
            and self.horizontal_angles_rad[0] <= horizontal_angle_rad <= self.horizontal_angles_rad[-1]
            and self.vertical_angles_rad[0] <= vertical_angle_rad <= self.vertical_angles_rad[-1]
        )

    def radiant_intensity_w_per_sr(
        self, horizontal_angle_rad: float, vertical_angle_rad: float
    ) -> float:
        horizontal = _interval(
            self.horizontal_angles_rad, horizontal_angle_rad, "horizontal", self.pattern_id
        )
        vertical = _interval(
            self.vertical_angles_rad, vertical_angle_rad, "vertical", self.pattern_id
        )
        low_h, high_h, weight_h = horizontal
        low_v, high_v, weight_v = vertical

        lower = (
            self.intensity_w_per_sr[low_v][low_h] * (1.0 - weight_h)
            + self.intensity_w_per_sr[low_v][high_h] * weight_h
        )
        upper = (
            self.intensity_w_per_sr[high_v][low_h] * (1.0 - weight_h)
            + self.intensity_w_per_sr[high_v][high_h] * weight_h
        )
        return lower * (1.0 - weight_v) + upper * weight_v


def _interval(
    axis: tuple[float, ...], value: float, name: str, pattern_id: str
) -> tuple[int, int, float]:
    """Bracketing indices and interpolation weight, refusing extrapolation."""

    if not math.isfinite(value) or value < axis[0] or value > axis[-1]:
        raise HeadlampPatternError(
            f"{name} angle outside the measured envelope",
            context={
                "pattern_id": pattern_id,
                "value_rad": value,
                "measured_from_rad": axis[0],
                "measured_to_rad": axis[-1],
            },
        )
    for index in range(len(axis) - 1):
        if value <= axis[index + 1]:
            span = axis[index + 1] - axis[index]
            weight = 0.0 if span == 0.0 else (value - axis[index]) / span
            return index, index + 1, weight
    return len(axis) - 2, len(axis) - 1, 1.0


def load_pattern(artifact_path: str | Path) -> TabulatedPattern:
    """Load a measured pattern, or say plainly that it is missing.

    The configuration declares ``headlamp_pattern: measured_non_lambertian``
    and names an artifact. If that artifact is absent this raises. It does not
    warn and continue, and it does not construct a Lambertian stand-in: a
    result computed from a cosine lobe while the manifest records a measured
    beam is indistinguishable from a real one afterwards.
    """

    path = Path(artifact_path)
    manifest = path / "manifest.json" if path.is_dir() else path
    if not manifest.exists():
        raise HeadlampPatternError(
            "the configured headlamp pattern artifact does not exist; "
            "measured_non_lambertian cannot be satisfied and no Lambertian "
            "fallback is permitted (implementation spec 12.1)",
            artifact_path=str(path),
        )

    payload = json.loads(manifest.read_text())
    schema = payload.get("schema")
    if schema != PATTERN_SCHEMA:
        raise HeadlampPatternError(
            "unrecognized headlamp pattern schema",
            artifact_path=str(manifest),
            context={"found": schema, "expected": PATTERN_SCHEMA},
        )
    if payload.get("units") != "W/sr":
        raise HeadlampPatternError(
            "headlamp pattern must be radiometric; photometric units weight by "
            "human visual response and a photodiode has none",
            artifact_path=str(manifest),
            context={"units": payload.get("units")},
        )
    return TabulatedPattern(
        pattern_id=payload["pattern_id"],
        horizontal_angles_rad=tuple(payload["horizontal_angles_rad"]),
        vertical_angles_rad=tuple(payload["vertical_angles_rad"]),
        intensity_w_per_sr=tuple(tuple(row) for row in payload["intensity_w_per_sr"]),
        source=payload["source"],
    )


__all__ = [
    "PATTERN_SCHEMA",
    "HeadlampPattern",
    "HeadlampPatternError",
    "LambertianPattern",
    "TabulatedPattern",
    "load_pattern",
]
