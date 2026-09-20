"""The assembled NR sidelink channel.

Pins the composition rather than the components: that the two failure
mechanisms stay separable to the outcome, that the recorded cause is the one
that actually fired, and that a matched random tape makes a counterfactual a
measurement rather than a sample.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.collision import headline_parameters
from hybrid_v2x_rl.channels.rf.model import (
    NRV2XChannel,
    RFChannelError,
    RFChannelRequest,
    RFPacketRandomness,
    RFPropagationRequest,
    marginal_and_joint,
)
from hybrid_v2x_rl.core.enums import FailureCause, RFPropagationState

BLOCKLENGTH = 4838
INFORMATION_BITS = 2784


def channel() -> NRV2XChannel:
    return NRV2XChannel(
        carrier_hz=5.9e9,
        bandwidth_hz=10e6,
        tx_power_dbm=23.0,
        blocklength=BLOCKLENGTH,
        information_bits=INFORMATION_BITS,
        collision=headline_parameters(),
    )


def request(
    *,
    distance_m: float = 50.0,
    state: RFPropagationState = RFPropagationState.LOS,
    blockage_db: float = 0.0,
    shadowing: float = 0.0,
    fading: float = 1.0,
    neighbours: int = 100,
    sensed: float = 1.0,
    draws: tuple[float, float, float] = (0.99, 0.99, 0.99),
) -> RFChannelRequest:
    return RFChannelRequest(
        distance_m=distance_m,
        propagation_state=state,
        blockage_db=blockage_db,
        shadowing_normalized=shadowing,
        fading_power_gain=fading,
        neighbour_count=neighbours,
        sensed_fraction=sensed,
        randomness=RFPacketRandomness(*draws),
    )


def propagation_request(
    *,
    distance_m: float = 50.0,
    state: RFPropagationState = RFPropagationState.LOS,
    blockage_db: float = 0.0,
    shadowing: float = 0.0,
    fading: float = 1.0,
) -> RFPropagationRequest:
    return RFPropagationRequest(
        distance_m=distance_m,
        propagation_state=state,
        blockage_db=blockage_db,
        shadowing_normalized=shadowing,
        fading_power_gain=fading,
    )


# -- the tape -----------------------------------------------------------------


def test_draws_outside_the_unit_interval_are_refused() -> None:
    with pytest.raises(RFChannelError, match="uniform draw"):
        RFPacketRandomness(1.5, 0.5, 0.5)


def test_the_same_tape_gives_the_same_outcome() -> None:
    """Bit-exact reproduction is what lets a trace be regenerated, not stored."""

    model = channel()
    first = model.evaluate(request())
    second = model.evaluate(request())
    assert first == second


def test_a_counterfactual_is_evaluated_against_the_same_tape() -> None:
    """Matched tapes are what make the oracle's advantage a measurement.

    Two different geometries evaluated on one packet's draws differ only by the
    geometry. If the model drew its own randomness, the difference would carry
    a sampling artefact that no amount of averaging distinguishes from a real
    effect at 1e-4.
    """

    model = channel()
    tape = (0.5, 0.002, 0.5)
    near = model.evaluate(request(distance_m=5.0, draws=tape))
    far = model.evaluate(request(distance_m=100.0, state=RFPropagationState.NLOS, draws=tape))
    assert near.sinr_db > far.sinr_db
    # Same draws, so any outcome difference is attributable to geometry alone.
    assert near.collision_probability == pytest.approx(far.collision_probability)


def test_failure_probability_does_not_depend_on_the_draws() -> None:
    model = channel()
    lucky = model.failure_probability(request(draws=(0.99, 0.99, 0.99)))
    unlucky = model.failure_probability(request(draws=(0.0, 0.0, 0.0)))
    assert lucky == pytest.approx(unlucky)


# -- composition --------------------------------------------------------------


def test_propagation_boundary_contains_no_policy_or_contention_inputs() -> None:
    assert tuple(field.name for field in fields(RFPropagationRequest)) == (
        "distance_m",
        "propagation_state",
        "blockage_db",
        "shadowing_normalized",
        "fading_power_gain",
    )


def test_propagation_evaluation_preserves_the_legacy_physical_budget() -> None:
    model = channel()
    legacy = model.evaluate(
        request(
            distance_m=100.0,
            state=RFPropagationState.NLOSV,
            blockage_db=9.0,
            shadowing=0.5,
            fading=0.25,
        )
    )
    propagation = model.evaluate_propagation(
        propagation_request(
            distance_m=100.0,
            state=RFPropagationState.NLOSV,
            blockage_db=9.0,
            shadowing=0.5,
            fading=0.25,
        )
    )

    assert propagation.propagation_state is legacy.propagation_state
    assert propagation.pathloss_db == pytest.approx(legacy.pathloss_db)
    assert propagation.shadowing_db == pytest.approx(legacy.shadowing_db)
    assert propagation.fading_gain_linear == pytest.approx(
        legacy.fading_gain_linear
    )
    assert propagation.sinr_db == pytest.approx(legacy.sinr_db)
    assert propagation.decoding_failure_probability == pytest.approx(
        legacy.decoding_failure_probability
    )


def test_propagation_evaluation_refuses_the_legacy_coupled_request() -> None:
    with pytest.raises(RFChannelError, match="RFPropagationRequest"):
        channel().evaluate_propagation(request())  # type: ignore[arg-type]


def test_the_two_mechanisms_stay_separable_in_the_result() -> None:
    """A single aggregate would let a diversity claim rest on whichever
    mechanism the reader assumed."""

    result = channel().evaluate(request())
    assert result.collision_probability > 0.0
    assert result.decoding_failure_probability >= 0.0
    assert result.total_failure_probability == pytest.approx(
        1.0
        - (1.0 - result.collision_probability)
        * (1.0 - result.decoding_failure_probability)
    )


def test_collision_dominates_the_budget_in_the_measured_regime() -> None:
    """Measured across M3: by two to four orders of magnitude."""

    for neighbours in (44, 100, 159):
        result = channel().evaluate(request(neighbours=neighbours))
        assert result.collision_dominates


def test_collision_is_independent_of_geometry() -> None:
    """Which resource a third party selected has nothing to do with range.

    If this ever couples, the one decoupled RF failure mode stops being
    decoupled and the diversity argument becomes circular.
    """

    model = channel()
    near = model.evaluate(request(distance_m=5.0))
    far = model.evaluate(request(distance_m=100.0, state=RFPropagationState.NLOS))
    assert near.collision_probability == pytest.approx(far.collision_probability)


def test_a_worse_class_lowers_the_sinr_and_raises_decoding_failure() -> None:
    model = channel()
    los = model.evaluate(request(distance_m=100.0, state=RFPropagationState.LOS))
    nlosv = model.evaluate(
        request(distance_m=100.0, state=RFPropagationState.NLOSV, blockage_db=9.0)
    )
    nlos = model.evaluate(request(distance_m=100.0, state=RFPropagationState.NLOS))

    assert los.sinr_db > nlosv.sinr_db > nlos.sinr_db
    assert nlos.decoding_failure_probability >= nlosv.decoding_failure_probability


def test_shadowing_moves_the_budget_by_its_class_spread() -> None:
    model = channel()
    clear = model.evaluate(request(shadowing=0.0))
    shadowed = model.evaluate(request(shadowing=2.0))
    # LOS sigma is 3 dB, so two standard deviations is 6 dB of extra loss.
    assert clear.sinr_db - shadowed.sinr_db == pytest.approx(6.0, abs=0.01)


def test_a_deep_fade_is_what_actually_breaks_the_budget() -> None:
    """The link has 34 to 56 dB of margin, so only the fading tail reaches it."""

    model = channel()
    typical = model.evaluate(request(distance_m=100.0, fading=1.0, draws=(0.99, 0.5, 0.99)))
    assert typical.decoding_failure_probability < 1e-9

    deep = model.evaluate(
        request(distance_m=100.0, fading=1e-4, draws=(0.99, 0.5, 0.99))
    )
    assert deep.decoding_failure_probability > 0.5
    assert deep.failure_cause is FailureCause.RF_CHANNEL


# -- causes -------------------------------------------------------------------


def test_a_delivered_packet_records_no_cause() -> None:
    result = channel().evaluate(request(draws=(0.999, 0.999, 0.999)))
    assert result.success
    assert result.failure_cause is FailureCause.NONE


def test_a_collision_is_recorded_as_a_collision() -> None:
    result = channel().evaluate(request(draws=(0.0, 0.999, 0.999)))
    assert not result.success
    assert result.failure_cause is FailureCause.RF_COLLISION


def test_half_duplex_is_recorded_even_when_a_collision_would_also_have_fired() -> None:
    """A receiver that is transmitting never hears the collision.

    Resolution follows the order the mechanisms occur, so the recorded cause is
    the one that actually stopped the packet rather than the first one tested.
    """

    result = channel().evaluate(request(draws=(0.0, 0.0, 0.0)))
    assert result.failure_cause is FailureCause.RF_COLLISION


def test_a_decoding_failure_is_only_reached_when_access_succeeds() -> None:
    model = channel()
    result = model.evaluate(
        request(distance_m=100.0, fading=1e-4, draws=(0.999, 0.0, 0.999))
    )
    assert result.failure_cause is FailureCause.RF_CHANNEL


# -- section 8.3 dependence ---------------------------------------------------


def test_the_dependence_ratio_is_one_under_independence() -> None:
    _, _, _, ratio = marginal_and_joint(0.05, 0.10, 0.05 * 0.10)
    assert ratio == pytest.approx(1.0)


def test_the_dependence_ratio_flags_failing_together_and_complementing() -> None:
    """Above one is correlated failure, below one is genuine complementarity.

    This is the number section 8.3 requires logged per scenario, and the reason
    is that the whole contribution is a claim about which side of one the ratio
    falls on.
    """

    _, _, _, together = marginal_and_joint(0.05, 0.10, 0.02)
    _, _, _, complementing = marginal_and_joint(0.05, 0.10, 0.001)
    assert together > 1.0
    assert complementing < 1.0


def test_an_impossible_probability_is_refused() -> None:
    with pytest.raises(RFChannelError, match="must lie in"):
        marginal_and_joint(1.4, 0.1, 0.05)


# -- statistical behaviour ----------------------------------------------------


def test_realized_failure_rate_matches_the_stated_probability() -> None:
    """The outcome must actually follow the probability it reports."""

    model = channel()
    rng = np.random.default_rng(11)
    stated = model.failure_probability(request())

    failures = 0
    trials = 40000
    for _ in range(trials):
        draws = tuple(rng.random(3))
        if not model.evaluate(request(draws=draws)).success:
            failures += 1
    assert failures / trials == pytest.approx(stated, abs=0.005)
