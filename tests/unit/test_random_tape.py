"""Matched randomness for the nine-action population environment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hybrid_v2x_rl.core.policy_actions import (
    MAX_RESERVED_RF_ATTEMPTS,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.random_tape import (
    MATCHED_TAPE_SCHEMA,
    ActionRandomness,
    MatchedPacketTapeFactory,
    PacketRandomnessIdentity,
    RandomTapeError,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

TRACE_ID = "synthetic-d20-train-000"
IDENTITY = PacketRandomnessIdentity(
    trace_id=TRACE_ID,
    pair_episode_id="pair-episode-a",
    packet_index=7,
)


def _vehicle(vehicle_id: str) -> VehicleTraceRecord:
    index = int(vehicle_id.split("-")[-1])
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=0.7,
        vehicle_id=vehicle_id,
        x_m=10.0 * index,
        y_m=0.0,
        heading_rad=0.0,
        speed_mps=5.0,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge-0_0",
        edge_id="edge-0",
        route_id="route-0",
        vehicle_type="passenger",
    )


def _frame() -> PopulationFrame:
    vehicles = tuple(_vehicle(f"veh-{index}") for index in range(1, 5))
    by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    pairs = (
        PopulationPair(
            pair_id="pair-a",
            episode_step=7,
            transmitter=by_id["veh-1"],
            receiver=by_id["veh-2"],
            lifecycle=PairLifecycle(born=False),
        ),
        PopulationPair(
            pair_id="pair-b",
            episode_step=2,
            transmitter=by_id["veh-3"],
            receiver=by_id["veh-4"],
            lifecycle=PairLifecycle(born=False),
        ),
    )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path(TRACE_ID),
            trace_id=TRACE_ID,
            split="train",
            density=20.0,
            replicate=0,
        ),
        index=7,
        time_s=0.7,
        vehicles=vehicles,
        pairs=pairs,
    )


def test_every_packet_owns_four_rf_entries_and_one_vlc_entry() -> None:
    tape = MatchedPacketTapeFactory(41).build(IDENTITY)

    assert len(tape.rf_attempts) == MAX_RESERVED_RF_ATTEMPTS == 4
    assert tape.vlc is not None
    assert all(
        0.0 <= draw <= 1.0
        for attempt in tape.rf_attempts
        for draw in (
            attempt.collision_draw,
            attempt.decoding_draw,
            attempt.half_duplex_draw,
        )
    )
    assert 0.0 <= tape.vlc.decoding_draw <= 1.0


def test_same_root_and_identity_reproduce_the_complete_tape() -> None:
    factory = MatchedPacketTapeFactory(41)

    assert factory.build(IDENTITY) == factory.build(IDENTITY)
    assert MatchedPacketTapeFactory(41).build(IDENTITY) == factory.build(IDENTITY)


def test_schema_and_reference_vector_are_pinned_for_artifact_replay() -> None:
    tape = MatchedPacketTapeFactory(41).build(IDENTITY)

    assert MATCHED_TAPE_SCHEMA == "hybrid-rf-vlc-rl.matched-packet-tape.v1"
    assert tape.rf_attempts[0].collision_draw == pytest.approx(
        0.6738407755885545
    )
    assert tape.rf_attempts[3].half_duplex_draw == pytest.approx(
        0.7269294308568429
    )
    assert tape.vlc.decoding_draw == pytest.approx(0.938595618052453)


def test_each_stable_identity_component_changes_the_tape() -> None:
    baseline = MatchedPacketTapeFactory(41).build(IDENTITY)
    alternatives = (
        MatchedPacketTapeFactory(42).build(IDENTITY),
        MatchedPacketTapeFactory(41).build(
            replace(IDENTITY, trace_id="synthetic-d20-train-001")
        ),
        MatchedPacketTapeFactory(41).build(
            replace(IDENTITY, pair_episode_id="pair-episode-b")
        ),
        MatchedPacketTapeFactory(41).build(
            replace(IDENTITY, packet_index=8)
        ),
    )

    assert all(candidate != baseline for candidate in alternatives)
    assert len(set(alternatives)) == len(alternatives)


def test_population_order_and_size_cannot_shift_a_pair_tape() -> None:
    factory = MatchedPacketTapeFactory(91)
    identities = tuple(
        replace(IDENTITY, pair_episode_id=f"pair-{index}")
        for index in range(5)
    )

    forward = {identity: factory.build(identity) for identity in identities}
    reverse = {identity: factory.build(identity) for identity in reversed(identities)}
    factory.build(replace(IDENTITY, pair_episode_id="extra-pair"))
    after_extra = {identity: factory.build(identity) for identity in identities}

    assert forward == reverse == after_extra


@pytest.mark.parametrize("action", tuple(PolicyAction))
def test_every_action_selects_exactly_its_reserved_randomness(
    action: PolicyAction,
) -> None:
    tape = MatchedPacketTapeFactory(12).build(IDENTITY)
    spec = action_resources(action)

    view = tape.view_for_action(action)

    assert view.identity is tape.identity
    assert view.action is action
    assert view.rf_attempts == tape.rf_attempts[: spec.reserved_rf_attempts]
    assert len(view.rf_attempts) == spec.reserved_rf_attempts
    assert (view.vlc is tape.vlc) is spec.uses_vlc


@pytest.mark.parametrize("attempts", (1, 2, 3, 4))
def test_rf_and_dup_actions_share_the_same_rf_prefix(attempts: int) -> None:
    tape = MatchedPacketTapeFactory(19).build(IDENTITY)
    rf_action = PolicyAction(attempts)
    dup_action = PolicyAction(attempts + 4)

    rf = tape.view_for_action(rf_action)
    duplicate = tape.view_for_action(dup_action)

    assert rf.rf_attempts == duplicate.rf_attempts == tape.rf_attempts[:attempts]
    assert rf.vlc is None
    assert duplicate.vlc is tape.vlc


def test_counterfactual_view_order_does_not_draw_or_mutate() -> None:
    tape = MatchedPacketTapeFactory(23).build(IDENTITY)
    reverse = {
        action: tape.view_for_action(action)
        for action in reversed(tuple(PolicyAction))
    }

    assert reverse == tape.counterfactual_views()
    assert tape == MatchedPacketTapeFactory(23).build(IDENTITY)


def test_each_link_mechanism_and_attempt_has_an_independent_namespace() -> None:
    tape = MatchedPacketTapeFactory(29).build(IDENTITY)
    scalar_draws = [
        draw
        for attempt in tape.rf_attempts
        for draw in (
            attempt.collision_draw,
            attempt.decoding_draw,
            attempt.half_duplex_draw,
        )
    ]
    scalar_draws.append(tape.vlc.decoding_draw)

    assert len(set(scalar_draws)) == 13


def test_population_pair_address_uses_trace_episode_and_episode_step() -> None:
    frame = _frame()
    pair = frame.pairs[0]
    factory = MatchedPacketTapeFactory(31)

    from_population = factory.for_population_pair(frame, pair)
    from_identity = factory.build(
        PacketRandomnessIdentity(
            trace_id=frame.source.trace_id,
            pair_episode_id=pair.episode_id,
            packet_index=pair.episode_step,
        )
    )

    assert from_population == from_identity


def test_tape_and_action_views_fail_closed_on_shape_drift() -> None:
    tape = MatchedPacketTapeFactory(37).build(IDENTITY)
    rf_two = tape.view_for_action(PolicyAction.RF_2)

    with pytest.raises(RandomTapeError, match="exactly four"):
        replace(tape, rf_attempts=tape.rf_attempts[:3])
    with pytest.raises(RandomTapeError, match="prefix length"):
        replace(rf_two, rf_attempts=rf_two.rf_attempts[:1])
    with pytest.raises(RandomTapeError, match="exact PolicyAction"):
        tape.view_for_action(1)  # type: ignore[arg-type]
    with pytest.raises(RandomTapeError, match="valid unsigned seed"):
        MatchedPacketTapeFactory(-1)

    assert isinstance(rf_two, ActionRandomness)


def test_population_pair_must_belong_to_the_supplied_frame() -> None:
    frame = _frame()
    unrelated = replace(frame.pairs[0], pair_id="pair-unrelated")

    with pytest.raises(RandomTapeError, match="does not belong"):
        MatchedPacketTapeFactory(43).for_population_pair(frame, unrelated)


@pytest.mark.parametrize(
    ("trace_id", "pair_episode_id", "packet_index"),
    (
        ("", "pair-a", 0),
        (TRACE_ID, "", 0),
        (TRACE_ID, "pair-a", -1),
        (TRACE_ID, "pair-a", True),
    ),
)
def test_invalid_packet_identity_is_rejected(
    trace_id: object,
    pair_episode_id: object,
    packet_index: object,
) -> None:
    with pytest.raises(RandomTapeError):
        PacketRandomnessIdentity(  # type: ignore[arg-type]
            trace_id=trace_id,
            pair_episode_id=pair_episode_id,
            packet_index=packet_index,
        )
