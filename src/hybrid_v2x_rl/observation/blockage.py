"""How likely the optical path is to be cut within the forecast horizon.

This is §6.1's ``predicted_blockage_probability``, the feature RQ3 and H3 rest
on, and §6.2 is explicit that it must be a **probability** rather than a
future binary state: "the predictor propagates track uncertainty and produces a
blockage probability, not a perfect future blockage state".

Everything here runs on :class:`~hybrid_v2x_rl.observation.forecast.PredictedState`
values built from noisy, aged tracks.  Nothing reads simulator truth, and the
module cannot: :mod:`hybrid_v2x_rl.geometry` is unreachable from this package.  What
it does share with the occlusion engine is
:mod:`hybrid_v2x_rl.core.link_endpoints`, so the path whose blockage is predicted is
built by the same follower-front-to-leader-rear rule the engine measures
against.  Two copies of that convention could drift apart and mis-state the
forecast against the truth it is graded on.

**The method.**  For one candidate blocker, project the geometry into the
frame of the link:

* ``d`` -- signed perpendicular distance from the blocker's predicted centre to
  the link line;
* ``r_perp`` -- half-extent of the blocker's rotated footprint along that same
  perpendicular, exact for a rectangle via its support function;
* ``sigma_perp`` -- the blocker's positional uncertainty projected onto the
  perpendicular, plus the endpoints' own uncertainty interpolated to the
  blocker's position along the path.

The blocker cuts the line when ``|d| < r_perp``, so with ``d`` Gaussian:

    P = Phi((r_perp - d) / sigma) - Phi((-r_perp - d) / sigma)

A second, identical calculation along the path decides whether the blocker is
between the endpoints at all, and the two are multiplied.

**Three approximations, each stated because each can only flatter the result
if forgotten.**

1. *Perpendicular and along-path terms are treated as independent.*  Exact only
   when the uncertainty ellipse is aligned with the path; the ellipse is
   aligned with the blocker's own heading (§6.2 gives no heading noise), so for
   cross-traffic -- the dominant blocker class, §4.6.1 -- the two axes are
   close to aligned and the error is small.  It is largest for a blocker
   travelling at 45 degrees to the link.
2. *Blockers are treated as independent of one another.*  Queued vehicles are
   not, so a queue's true blocking probability is lower than the product rule
   suggests: correlated events overlap.  The bias is therefore towards
   over-predicting blockage, which costs the policy duplication rather than
   reliability.
3. *Vehicle dimensions are exact*, because §6.2 declares noise on position and
   speed only.  A real system estimates size from a classifier.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hybrid_v2x_rl.core.link_endpoints import DEFAULT_HEADLAMP_HEIGHT_M
from hybrid_v2x_rl.observation.forecast import PredictedState

#: Below this, a predicted separation is treated as degenerate and no forecast
#: is attempted: the link direction is not defined.
_MIN_PATH_LENGTH_M = 1e-6

#: Floor on any projected standard deviation, so a zero-noise configuration
#: yields a hard geometric verdict instead of dividing by zero.
_MIN_SIGMA_M = 1e-9


@dataclass(frozen=True, slots=True)
class BlockerShape:
    """A candidate blocker's dimensions, which §6.2 leaves un-noised."""

    length_m: float
    width_m: float
    height_m: float

    def half_extent_towards(self, heading_rad: float, bearing_rad: float) -> float:
        """Support function of the rectangle along ``bearing_rad``.

        Exact for a rectangle: the farthest the body reaches in that direction
        from its centre.  A vehicle crossing perpendicular to a link presents
        its *length*, which is why the point-with-tolerance approximation this
        replaces understated cross-traffic blockage by up to 2.5x (§4.6.1).
        """

        offset = bearing_rad - heading_rad
        return 0.5 * self.length_m * abs(math.cos(offset)) + 0.5 * self.width_m * abs(
            math.sin(offset)
        )


@dataclass(frozen=True, slots=True)
class BlockageForecast:
    """Predicted probability that the optical path is cut, and how sure of it."""

    probability: float
    confidence: float
    horizon_s: float
    considered: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _interval_probability(centre: float, half_width: float, sigma: float) -> float:
    """P(|X| < half_width) for X ~ N(centre, sigma), clamped to [0, 1]."""

    if half_width <= 0.0:
        return 0.0
    scale = max(sigma, _MIN_SIGMA_M)
    upper = _standard_normal_cdf((half_width - centre) / scale)
    lower = _standard_normal_cdf((-half_width - centre) / scale)
    return min(1.0, max(0.0, upper - lower))


def blockage_probability(
    transmitter: PredictedState,
    receiver: PredictedState,
    blockers: Iterable[tuple[PredictedState, BlockerShape]],
    *,
    horizon_s: float,
    link_height_m: float = DEFAULT_HEADLAMP_HEIGHT_M,
) -> BlockageForecast:
    """Probability that some body cuts the predicted optical path.

    ``transmitter`` and ``receiver`` are the predicted states of the pair, and
    the path is taken between them.  The endpoints' own uncertainty is carried
    into the perpendicular term, interpolated along the path, because a link
    whose ends are poorly known is itself poorly located.
    """

    dx = receiver.x_m - transmitter.x_m
    dy = receiver.y_m - transmitter.y_m
    length = math.hypot(dx, dy)
    if length < _MIN_PATH_LENGTH_M:
        return BlockageForecast(
            probability=0.0,
            confidence=min(transmitter.confidence, receiver.confidence),
            horizon_s=horizon_s,
            considered=0,
        )

    along_bearing = math.atan2(dy, dx)
    perp_bearing = along_bearing + 0.5 * math.pi
    ux, uy = dx / length, dy / length

    candidates: Sequence[tuple[PredictedState, BlockerShape]] = tuple(blockers)
    clear = 1.0
    considered = 0

    for state, shape in candidates:
        if state.vehicle_id in (transmitter.vehicle_id, receiver.vehicle_id):
            continue
        # A body shorter than the beam passes under it.  Every configured class
        # is at least 1.5 m against a 0.7 m path, so this almost never
        # discriminates (§4.6.2); it is here so a future short class behaves.
        if shape.height_m < link_height_m:
            continue

        rx = state.x_m - transmitter.x_m
        ry = state.y_m - transmitter.y_m
        along = rx * ux + ry * uy
        perp = -rx * uy + ry * ux

        # How far along the link this blocker sits, for interpolating the
        # endpoints' contribution to where the line actually is.
        position = min(1.0, max(0.0, along / length))
        endpoint_var = ((1.0 - position) * transmitter.std_towards(perp_bearing)) ** 2 + (
            position * receiver.std_towards(perp_bearing)
        ) ** 2
        sigma_perp = math.hypot(state.std_towards(perp_bearing), math.sqrt(endpoint_var))
        sigma_along = state.std_towards(along_bearing)

        crosses = _interval_probability(
            perp, shape.half_extent_towards(state.heading_rad, perp_bearing), sigma_perp
        )
        if crosses <= 0.0:
            considered += 1
            continue

        # Between the endpoints: centre the along-path interval on the link's
        # midpoint and give it half the link plus the body's own reach.
        between = _interval_probability(
            along - 0.5 * length,
            0.5 * length + shape.half_extent_towards(state.heading_rad, along_bearing),
            sigma_along,
        )

        clear *= 1.0 - crosses * between
        considered += 1

    return BlockageForecast(
        probability=min(1.0, max(0.0, 1.0 - clear)),
        confidence=min(transmitter.confidence, receiver.confidence),
        horizon_s=horizon_s,
        considered=considered,
    )


__all__ = [
    "BlockageForecast",
    "BlockerShape",
    "blockage_probability",
]
