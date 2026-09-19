"""The vector a deployable policy actually sees.

Work plan §6.1 lists the features; this assembles them and pins their order.
Everything arrives from :mod:`hybrid_v2x_rl.observation.sensing`,
:mod:`~hybrid_v2x_rl.observation.tracks`, :mod:`~hybrid_v2x_rl.observation.forecast`,
:mod:`~hybrid_v2x_rl.observation.blockage` and :mod:`~hybrid_v2x_rl.observation.link_state`,
so every number here is already noisy, aged, or derived from something that is.

**Feature order is a contract.**  It follows the configured ``features`` list,
so the vector is reproducible from the frozen configuration rather than from
whatever order this module happens to build things in.  A silent reordering
would leave a trained checkpoint reading the wrong columns while every shape
check still passed.

**Absent is encoded, not defaulted.**  A leg that has never been measured has
no age, and §6.1 has no "validity" feature to carry that.  Rather than pick a
large number -- which would make "never measured" indistinguishable from "very
stale", two states a policy should treat differently -- an unmeasured age is
emitted as :data:`UNMEASURED_AGE_S`, a negative value no real age can take.
The policy sees a value outside the natural range and can learn what it means.
Quality for an unmeasured leg is :data:`UNMEASURED_QUALITY`, with the same
reasoning.

**Nothing exact reaches this vector.**  The forbidden-field list in
``hybrid_v2x_rl.config.validation`` guards the configuration; this module is guarded
by construction, because :mod:`hybrid_v2x_rl.geometry` is unimportable from here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.core.intersection_context import IntersectionContext
from hybrid_v2x_rl.core.pair_geometry import wrap_to_pi
from hybrid_v2x_rl.observation.blockage import BlockageForecast
from hybrid_v2x_rl.observation.forecast import PredictedState
from hybrid_v2x_rl.observation.link_state import LinkStateTracker
from hybrid_v2x_rl.observation.tracks import Track

#: Emitted where a leg has never been measured.  Negative, so it cannot be
#: confused with any real age, however stale.
UNMEASURED_AGE_S = -1.0

#: Emitted where a leg has never produced a quality reading.
UNMEASURED_QUALITY = 0.0

#: Features whose value is a fixed-width history rather than one number.
HISTORY_FEATURES: frozenset[str] = frozenset({"rf_quality_history", "vlc_quality_history"})


@runtime_checkable
class BuilderConfigSource(Protocol):
    """The subset of ``ObservationConfig`` this module needs."""

    @property
    def features(self) -> tuple[str, ...]: ...
    @property
    def history_packets(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    """Names, in order, of every column the policy receives.

    Built once from the configured feature list and then reused, so the width
    and the meaning of each column are fixed for the life of a run.
    """

    features: tuple[str, ...]
    history_packets: int

    @property
    def columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for feature in self.features:
            if feature in HISTORY_FEATURES:
                names.extend(f"{feature}[{index}]" for index in range(self.history_packets))
            else:
                names.append(feature)
        return tuple(names)

    @property
    def width(self) -> int:
        return len(self.columns)


@dataclass(frozen=True, slots=True)
class PairObservationInputs:
    """Everything the builder needs, all of it already degraded.

    ``transmitter``/``receiver`` are the *predicted* states of the tagged pair;
    ``transmitter_track`` and ``receiver_track`` supply the ages that predicted
    states no longer carry individually.
    """

    now_s: float
    transmitter: PredictedState
    receiver: PredictedState
    transmitter_track: Track
    receiver_track: Track
    blockage: BlockageForecast
    links: LinkStateTracker
    neighbour_count: int
    channel_busy_ratio: float
    fov_half_angle_rad: float
    #: Junction relationship of the *predicted* path.  Lawful state: own
    #: pose plus a road map, which a real vehicle carries.  Because it is
    #: computed from noisy poses it is honestly uncertain near a boundary,
    #: rather than the clean oracle bit the geometry engine would give.
    intersection: IntersectionContext


def _quality_and_age(links: LinkStateTracker, link: Link, now_s: float) -> tuple[float, float]:
    history = links.history(link)
    newest = history.latest
    if newest is None:
        return UNMEASURED_QUALITY, UNMEASURED_AGE_S
    return newest.value, newest.age_s(now_s)


def _pair_distance_m(inputs: PairObservationInputs) -> float:
    return math.hypot(
        inputs.receiver.x_m - inputs.transmitter.x_m,
        inputs.receiver.y_m - inputs.transmitter.y_m,
    )


def _pair_bearing_rad(inputs: PairObservationInputs) -> float:
    """Bearing to the receiver, relative to the transmitter's own heading.

    Relative rather than absolute: a policy should not have to learn the map's
    compass, and an absolute bearing would make an identical geometry look
    different on a northbound street and an eastbound one.
    """

    absolute = math.atan2(
        inputs.receiver.y_m - inputs.transmitter.y_m,
        inputs.receiver.x_m - inputs.transmitter.x_m,
    )
    return wrap_to_pi(absolute - inputs.transmitter.heading_rad)


def _optical_fov_margin_rad(inputs: PairObservationInputs) -> float:
    """How much acceptance angle is left before the photodiode loses the beam.

    Positive inside the cone, negative outside, so zero is the boundary.  The
    incidence angle is measured at the *receiver*, between its own heading and
    the direction back to the transmitter, which is why a leader turning at a
    junction drives this negative before anything blocks the path (§4.6.1).
    """

    back_bearing = math.atan2(
        inputs.transmitter.y_m - inputs.receiver.y_m,
        inputs.transmitter.x_m - inputs.receiver.x_m,
    )
    # The photodiode faces rearward, to see a follower behind it, so its
    # boresight is opposite the receiver's heading.
    boresight = inputs.receiver.heading_rad + math.pi
    incidence = abs(wrap_to_pi(back_bearing - boresight))
    return inputs.fov_half_angle_rad - incidence


def link_feature_values(
    links: LinkStateTracker, now_s: float
) -> Mapping[str, float | Sequence[float]]:
    """The features that depend on what the policy did, and on nothing else.

    Split out because it is the only part of the observation a *replay* cannot
    precompute. Everything else here is a function of the trace, so it is the
    same for every policy and every seed and can be measured once; these nine
    depend on the actions taken and must be rebuilt as a trajectory unfolds.

    Exposed rather than inlined so that a cached environment reconstructs them
    with this code and not with a copy of it. A second implementation that
    drifted from this one would train the policy on an observation the
    evaluator never produces, and nothing downstream would report a mismatch.
    """

    rf_quality, rf_age = _quality_and_age(links, Link.RF, now_s)
    vlc_quality, vlc_age = _quality_and_age(links, Link.VLC, now_s)
    return {
        "rf_quality": rf_quality,
        "rf_quality_age": rf_age,
        "rf_quality_history": links.rf.padded_values(fill=UNMEASURED_QUALITY),
        "vlc_quality": vlc_quality,
        "vlc_quality_age": vlc_age,
        "vlc_quality_history": links.vlc.padded_values(fill=UNMEASURED_QUALITY),
        "previous_action": (
            float(links.previous_action) if links.previous_action is not None else -1.0
        ),
        "last_delivery_outcome": (
            -1.0 if links.last_delivered is None else float(links.last_delivered)
        ),
        "consecutive_miss_count": float(links.consecutive_misses),
    }


#: Feature names :func:`link_feature_values` supplies. The complement is
#: trace-derived, and the two sets together must cover every configured feature.
LINK_FEATURES: frozenset[str] = frozenset(
    {
        "rf_quality",
        "rf_quality_age",
        "rf_quality_history",
        "vlc_quality",
        "vlc_quality_age",
        "vlc_quality_history",
        "previous_action",
        "last_delivery_outcome",
        "consecutive_miss_count",
    }
)


def _feature_values(inputs: PairObservationInputs) -> Mapping[str, float | Sequence[float]]:
    """One entry per §6.1 feature name."""

    tx_velocity = inputs.transmitter.speed_mps
    rx_velocity = inputs.receiver.speed_mps

    return {
        **link_feature_values(inputs.links, inputs.now_s),
        "rf_channel_busy_ratio": inputs.channel_busy_ratio,
        "neighbor_count": float(inputs.neighbour_count),
        "pair_distance": _pair_distance_m(inputs),
        "pair_bearing": _pair_bearing_rad(inputs),
        # Signed so that positive means the gap is opening, whichever way the
        # pair happens to be travelling.  Speed difference rather than a
        # projected closing rate, which is the same thing for a leader and
        # follower on a shared route and is what §6.1 asks for.
        "relative_speed": rx_velocity - tx_velocity,
        "heading_difference": wrap_to_pi(
            inputs.receiver.heading_rad - inputs.transmitter.heading_rad
        ),
        "optical_fov_margin": _optical_fov_margin_rad(inputs),
        # Work plan section 4.6.1: 65-121x lift, against a separation
        # signal that is weak and non-monotonic.  Both forms are offered --
        # the continuous distance lets the network choose its own
        # threshold and degrade gracefully, while the indicator is the
        # discriminator actually measured.
        "distance_to_junction": inputs.intersection.nearest_distance_to_junction_m,
        "path_spans_junction": float(inputs.intersection.spans_junction),
        "predicted_blockage_probability": inputs.blockage.probability,
        "predictor_confidence": inputs.blockage.confidence,
        # The older of the two, because a pair is only as well known as its
        # worse-tracked end.
        "track_age": max(
            inputs.transmitter_track.age_s(inputs.now_s),
            inputs.receiver_track.age_s(inputs.now_s),
        ),
    }


@dataclass(frozen=True, slots=True)
class ObservationBuilder:
    """Turns degraded state into the fixed-width vector §6.1 specifies."""

    schema: ObservationSchema

    @classmethod
    def from_config(cls, observation: BuilderConfigSource) -> ObservationBuilder:
        return cls(
            schema=ObservationSchema(
                features=tuple(observation.features),
                history_packets=int(observation.history_packets),
            )
        )

    def build(self, inputs: PairObservationInputs) -> tuple[float, ...]:
        """Assemble one observation, in the configured feature order."""

        values = _feature_values(inputs)
        missing = [name for name in self.schema.features if name not in values]
        if missing:
            raise KeyError(
                "configured observation features have no implementation: " + ", ".join(missing)
            )

        vector: list[float] = []
        for feature in self.schema.features:
            value = values[feature]
            if feature in HISTORY_FEATURES:
                if isinstance(value, int | float):
                    raise TypeError(f"{feature} must produce a sequence, not a scalar")
                history = tuple(float(item) for item in value)
                if len(history) != self.schema.history_packets:
                    raise ValueError(
                        f"{feature} produced {len(history)} values, "
                        f"expected {self.schema.history_packets}"
                    )
                vector.extend(history)
            else:
                if not isinstance(value, int | float):
                    raise TypeError(f"{feature} must produce a scalar, not a sequence")
                vector.append(float(value))

        if len(vector) != self.schema.width:
            raise ValueError(
                f"observation width {len(vector)} does not match schema {self.schema.width}"
            )
        if not all(math.isfinite(item) for item in vector):
            raise ValueError("observation contains a non-finite value")
        return tuple(vector)


__all__ = [
    "BuilderConfigSource",
    "HISTORY_FEATURES",
    "LINK_FEATURES",
    "UNMEASURED_AGE_S",
    "UNMEASURED_QUALITY",
    "ObservationBuilder",
    "ObservationSchema",
    "PairObservationInputs",
    "link_feature_values",
]
