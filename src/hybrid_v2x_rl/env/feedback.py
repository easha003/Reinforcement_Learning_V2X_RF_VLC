"""The measurement a receiver reports back, and the only way quality crosses.

Work plan section 6.3 makes one sequential claim, and it is the only reason
this problem is an MDP rather than a contextual bandit:

    A fresh post-transmission quality measurement is obtained on whichever legs
    were used; the unused leg's most recent measurement ages. The current
    action therefore affects future information freshness.

:mod:`hybrid_v2x_rl.observation.link_state` implements the ageing and calls quality
"an opaque scalar", deferring to M3 and M4 for its meaning. This module is that
meaning: the realized SINR of the radio's last attempt and the realized
electrical SNR of the optical leg, which are what the channel results already
compute and what a real receiver would feed back.

**The oracle value never crosses on its own.** ``RFChannelResult`` says of
itself that everything on it is oracle-side, and handing the policy an exact
SINR would leak a quantity no receiver measures exactly -- the same mistake the
position and blockage paths already guard against, arriving through a channel
nobody was watching. So a reading is corrupted by an estimation error and
quantized to a reporting step before it is allowed out, and this is the only
function that performs the conversion.

The noise is drawn from the packet's identity rather than a running generator,
so RF-only and DUP see *the same* measurement error on the same packet. That is
the matched-tape rule from :mod:`hybrid_v2x_rl.env.packet` applied to the feedback
channel: without it, the difference between two actions on one packet would
carry a sampling artefact in the observation as well as in the outcome.
"""

from __future__ import annotations

import math

from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.core.randomness import make_generator
from hybrid_v2x_rl.env.packet import PacketOutcome

#: Standard deviation of the reported quality's estimation error, in dB.
#:
#: A receiver's SINR estimate is not exact, and 1 dB is the order of magnitude
#: usually quoted for a wideband estimate over a slot. The value matters less
#: than its presence: at zero the policy would read the channel exactly.
MEASUREMENT_ERROR_DB: float = 1.0

#: Reporting granularity, in dB. Feedback is quantized in real systems, and a
#: continuous report would let the policy resolve differences no format carries.
REPORTING_STEP_DB: float = 1.0

#: Range each leg's reading is clipped to and normalized over, in dB.
#:
#: Clipping is not cosmetic. A fully blocked optical link has zero received
#: power and therefore an SNR of ``-inf``; unclipped, that single value would
#: make the feature column unusable and any normalization meaningless.
RF_QUALITY_SPAN_DB: tuple[float, float] = (-10.0, 40.0)
VLC_QUALITY_SPAN_DB: tuple[float, float] = (0.0, 60.0)

_SPANS: dict[Link, tuple[float, float]] = {
    Link.RF: RF_QUALITY_SPAN_DB,
    Link.VLC: VLC_QUALITY_SPAN_DB,
}


def _report(value_db: float, span: tuple[float, float], noise_db: float) -> float:
    """One oracle reading, degraded to what the format could carry, in [0, 1].

    Normalized rather than raw so the two legs' readings occupy the same range
    and neither dominates a network input purely through its units.
    """

    low, high = span
    if not math.isfinite(value_db):
        # -inf is a real, physically meaningful reading: the link delivered no
        # power at all. It reports as the bottom of the scale, not as a hole.
        noisy = low if value_db < 0.0 else high
    else:
        noisy = value_db + noise_db
    quantized = round(noisy / REPORTING_STEP_DB) * REPORTING_STEP_DB
    return float(min(1.0, max(0.0, (quantized - low) / (high - low))))


def reported_quality(
    value_db: float,
    *,
    link: Link,
    root_seed: int,
    trace_id: str,
    pair_id: str,
    packet_index: int,
) -> float:
    """Return one reproducible noisy report addressed by packet and medium."""

    if not isinstance(link, Link):
        raise TypeError("link must be a Link")
    generator = make_generator(
        root_seed,
        f"feedback|{link.name}",
        trace_id=trace_id,
        episode_id=pair_id,
        packet_index=packet_index,
    )
    noise = float(generator.normal(0.0, MEASUREMENT_ERROR_DB))
    return _report(value_db, _SPANS[link], noise)


def measurements(
    outcome: PacketOutcome,
    *,
    root_seed: int,
    trace_id: str,
    pair_id: str,
    packet_index: int,
) -> dict[Link, float]:
    """What the transmitter learns from the legs this packet actually spent.

    Returns only the legs the action used. An empty entry is not the same as a
    zero: the caller passes this straight to
    :meth:`hybrid_v2x_rl.env.perception.Perception.record`, which refreshes exactly
    the keys present and lets every other leg's reading age.
    """

    readings: dict[Link, float] = {}
    for link, value in (
        (Link.RF, outcome.rf_quality_db),
        (Link.VLC, outcome.vlc_quality_db),
    ):
        if value is None:
            continue
        # Seeded per packet *and* per leg, so the two legs' errors are
        # independent while staying identical across the actions that use them.
        readings[link] = reported_quality(
            value,
            link=link,
            root_seed=root_seed,
            trace_id=trace_id,
            pair_id=pair_id,
            packet_index=packet_index,
        )
    return readings


__all__ = [
    "MEASUREMENT_ERROR_DB",
    "REPORTING_STEP_DB",
    "RF_QUALITY_SPAN_DB",
    "VLC_QUALITY_SPAN_DB",
    "measurements",
    "reported_quality",
]
