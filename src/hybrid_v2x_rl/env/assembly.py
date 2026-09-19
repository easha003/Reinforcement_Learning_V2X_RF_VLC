"""Turn a resolved configuration into the objects the environment runs on.

There is exactly one place where the frozen service and channel profiles become
a :class:`~hybrid_v2x_rl.env.packet.PacketLifecycle`, and this is it. That matters more
than it looks: the same construction was already being written out by hand in
two test modules, and the oracle and the baselines needed it a third time. Four
copies of "how many coded bits does a transmission carry" is four chances for a
config change to be honoured in three of them.

**Nothing here decides anything.** Every number is read from the configuration
or derived from it by an arithmetic identity that is stated in the comment next
to it. If a value appears in this file that cannot be traced back to a
configuration field, it is a parameter that escaped the provenance table.
"""

from __future__ import annotations

import numpy as np

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand, headline_parameters
from hybrid_v2x_rl.channels.rf.model import NRV2XChannel
from hybrid_v2x_rl.channels.vlc.headlamp_pattern import load_pattern
from hybrid_v2x_rl.channels.vlc.model import VVLCChannel
from hybrid_v2x_rl.channels.vlc.noise import CLEAR_DAY, CLEAR_NIGHT, AmbientCondition
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.config.models import BITS_PER_RESOURCE_ELEMENT, ProjectConfig
from hybrid_v2x_rl.env.packet import PacketLifecycle, Timing

_AMBIENT: dict[str, AmbientCondition] = {
    "clear_night": CLEAR_NIGHT,
    "clear_day": CLEAR_DAY,
}


def build_rf_channel(
    config: ProjectConfig,
    *,
    band: SensitivityBand | None = None,
    rf_usage_fraction: float = 1.0,
) -> NRV2XChannel:
    """The radio façade for this profile.

    ``blocklength`` is in *channel uses*, not coded bits: the finite-blocklength
    normal approximation counts complex symbols, and the resource grid delivers
    ``bits_per_element`` of them per symbol. Dividing by the modulation order is
    therefore an identity, not a fudge -- but it has to follow the configured
    modulation, because a profile that switched to QPSK while this stayed at
    four would silently claim four times the blocklength it has.
    """

    bits_per_element = BITS_PER_RESOURCE_ELEMENT[config.rf.modulation]
    channel_uses = int(config.rf.available_coded_bits() / bits_per_element)
    return NRV2XChannel(
        carrier_hz=config.rf.carrier_hz,
        bandwidth_hz=config.rf.bandwidth_hz,
        tx_power_dbm=config.rf.tx_power_dbm,
        blocklength=channel_uses,
        information_bits=(config.service.payload_bytes + config.rf.timing.framing_overhead_bytes) * 8,
        collision=headline_parameters(
            band or SensitivityBand.NOMINAL,
            # What this profile actually commits per packet, not a default.
            committed_airtime_s=config.rf.timing.airtime_s
            * config.service.rf_attempts_per_packet,
            # One means every packet puts a copy on the radio, which is what a
            # single-medium profile does. Below one the radio is carrying only
            # part of the offered traffic because the rest went by light, and
            # the contention, the half-duplex term and the pool claim all fall
            # together. A caller scoring an allocation has to build the channel
            # that allocation produces, not the one it started from.
            rf_usage_fraction=rf_usage_fraction,
        ),
    )


def build_vlc_channel(config: ProjectConfig) -> VVLCChannel:
    """The optical façade for this profile.

    :func:`~hybrid_v2x_rl.channels.vlc.headlamp_pattern.load_pattern` raises if the
    calibration artifact is missing rather than falling back to a Lambertian
    lamp, so a profile pointing at an absent artifact fails here instead of
    quietly producing a different physics.
    """

    return VVLCChannel(
        pattern=load_pattern(config.vlc.pattern_artifact),
        # The acceptance semi-angle must be the *same* number here as in the
        # geometry test. receiver.py says so in its own docstring -- "there is
        # one number in the codebase and the two uses cannot drift apart" --
        # and this line is where that promise was being broken: the receiver
        # was constructed with its module default, so a profile that narrowed
        # the cone tightened the acceptance test and left the concentrator gain
        # at 60 degrees. That is the wide-cone-availability with narrow-cone-gain
        # error the configuration warns about, running in the opposite
        # direction: narrow acceptance paid for, wide-cone gain received.
        receiver=OpticalReceiver(
            fov_half_angle_rad=np.deg2rad(config.vlc.receiver_fov_deg)
        ),
        electrical_bandwidth_hz=config.vlc.electrical_bandwidth_hz,
        payload_bytes=config.service.payload_bytes,
        framing_bytes=config.vlc.timing.framing_overhead_bytes,
        code_rate=config.vlc.timing.code_rate,
        ambient=_AMBIENT[config.vlc.ambient_condition],
    )


def build_timing(config: ProjectConfig) -> Timing:
    """The deadline arithmetic, which validates itself on construction."""

    return Timing(
        deadline_s=config.service.deadline_s,
        predecision_lead_s=config.service.predecision_lead_s,
        rf_airtime_s=config.rf.timing.airtime_s,
        vlc_airtime_s=config.vlc.timing.airtime_s,
        rf_attempts=config.service.rf_attempts_per_packet,
    )


def build_lifecycle(
    config: ProjectConfig,
    *,
    band: SensitivityBand | None = None,
    rf_usage_fraction: float = 1.0,
) -> PacketLifecycle:
    """Both façades and the deadline, assembled and checked for feasibility."""

    return PacketLifecycle(
        rf=build_rf_channel(config, band=band, rf_usage_fraction=rf_usage_fraction),
        vlc=build_vlc_channel(config),
        timing=build_timing(config),
    )


def build_rollout(
    config: ProjectConfig,
    *,
    buildings,
    root_seed: int = 0,
    band: SensitivityBand | None = None,
    rf_usage_fraction: float = 1.0,
):
    """A rollout wired to this profile, including the receiver's acceptance cone.

    Imported lazily so :mod:`hybrid_v2x_rl.env.rollout` can import this module for its
    own construction helpers without a cycle.
    """

    from hybrid_v2x_rl.env.rollout import Rollout

    return Rollout(
        lifecycle=build_lifecycle(
            config, band=band, rf_usage_fraction=rf_usage_fraction
        ),
        buildings=tuple(buildings),
        root_seed=root_seed,
        fov_half_angle_rad=np.deg2rad(config.vlc.receiver_fov_deg),
    )


__all__ = [
    "build_lifecycle",
    "build_rf_channel",
    "build_rollout",
    "build_timing",
    "build_vlc_channel",
]
