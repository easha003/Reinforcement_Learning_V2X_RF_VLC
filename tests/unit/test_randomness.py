"""Tests for deterministic, context-addressed random streams."""

import numpy as np
import pytest

from hybrid_v2x_rl.core.randomness import (
    RANDOM_STREAM_NAMES,
    RandomStreams,
    derive_child_seed,
    derive_stream_seeds,
    make_generator,
)


def test_child_seed_is_stable_and_unsigned_64_bit() -> None:
    seed = derive_child_seed(
        20260728,
        "fading",
        trace_id="trace-0042",
        episode_id=7,
        packet_index=901,
    )
    assert seed == 15_274_119_018_919_600_874
    assert 0 <= seed < 2**64


def test_each_identity_component_changes_the_child_namespace() -> None:
    baseline = derive_child_seed(
        9,
        "collision",
        trace_id="trace-a",
        episode_id="episode-a",
        packet_index=10,
    )
    alternatives = {
        derive_child_seed(
            10,
            "collision",
            trace_id="trace-a",
            episode_id="episode-a",
            packet_index=10,
        ),
        derive_child_seed(
            9,
            "decoding",
            trace_id="trace-a",
            episode_id="episode-a",
            packet_index=10,
        ),
        derive_child_seed(
            9,
            "collision",
            trace_id="trace-b",
            episode_id="episode-a",
            packet_index=10,
        ),
        derive_child_seed(
            9,
            "collision",
            trace_id="trace-a",
            episode_id="episode-b",
            packet_index=10,
        ),
        derive_child_seed(
            9,
            "collision",
            trace_id="trace-a",
            episode_id="episode-a",
            packet_index=11,
        ),
    }
    assert baseline not in alternatives
    assert len(alternatives) == 5


def test_identifier_type_is_part_of_seed_identity() -> None:
    numeric = derive_child_seed(1, "sensor", trace_id=1)
    textual = derive_child_seed(1, "sensor", trace_id="1")
    assert numeric != textual


def test_explicit_generators_reproduce_without_global_rng() -> None:
    first = make_generator(
        123,
        "shadowing",
        trace_id="trace",
        episode_id=2,
        packet_index=5,
    ).normal(size=16)
    second = make_generator(
        123,
        "shadowing",
        trace_id="trace",
        episode_id=2,
        packet_index=5,
    ).normal(size=16)
    np.testing.assert_array_equal(first, second)


def test_packet_addressing_makes_evaluation_order_irrelevant() -> None:
    def draw(packet_index: int) -> float:
        return float(
            make_generator(
                456,
                "decoding",
                trace_id="trace",
                episode_id="episode",
                packet_index=packet_index,
            ).random()
        )

    ascending = {index: draw(index) for index in range(10)}
    descending = {index: draw(index) for index in reversed(range(10))}
    assert ascending == descending


def test_random_stream_bundle_contains_all_independent_streams() -> None:
    bundle = RandomStreams.from_root_seed(
        88,
        trace_id="trace",
        episode_id=3,
    )
    assert tuple(bundle.as_dict()) == RANDOM_STREAM_NAMES

    repeated = RandomStreams.from_root_seed(
        88,
        trace_id="trace",
        episode_id=3,
    )
    for name in RANDOM_STREAM_NAMES:
        first_draw = bundle.as_dict()[name].integers(0, 2**32, size=8)
        second_draw = repeated.as_dict()[name].integers(0, 2**32, size=8)
        np.testing.assert_array_equal(first_draw, second_draw)


def test_seed_map_matches_individual_derivation() -> None:
    seeds = derive_stream_seeds(
        111,
        trace_id="trace-z",
        episode_id=4,
        packet_index=99,
    )
    assert tuple(seeds) == RANDOM_STREAM_NAMES
    assert seeds["policy"] == derive_child_seed(
        111,
        "policy",
        trace_id="trace-z",
        episode_id=4,
        packet_index=99,
    )


@pytest.mark.parametrize(
    ("root_seed", "stream_name", "packet_index", "exception"),
    [
        (-1, "fading", 0, ValueError),
        (True, "fading", 0, TypeError),
        (1, "", 0, ValueError),
        (1, " fading", 0, ValueError),
        (1, "fading", -1, ValueError),
    ],
)
def test_invalid_seed_components_are_rejected(
    root_seed: object,
    stream_name: object,
    packet_index: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        derive_child_seed(  # type: ignore[arg-type]
            root_seed,
            stream_name,
            packet_index=packet_index,
        )
