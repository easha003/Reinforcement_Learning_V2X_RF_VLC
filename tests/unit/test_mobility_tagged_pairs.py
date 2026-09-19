"""Tests for deterministic natural tagged-pair extraction."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from hybrid_v2x_rl.mobility.tagged_pairs import (
    TaggedPairConfig,
    TaggedPairExtractor,
    VehicleRecord,
    build_vehicle_frames,
)


@dataclass(frozen=True)
class _TraceLikeRecord:
    trace_id: str
    time_s: float
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    route_id: str
    lane_id: str


def _vehicle(
    time_s: float,
    vehicle_id: str,
    x_m: float,
    *,
    route_id: str = "route-main",
    lane_id: str = "east-a_0",
    heading_rad: float = 0.0,
    planned_route: tuple[str, ...] | None = ("east-a", "east-b"),
) -> VehicleRecord:
    return VehicleRecord(
        trace_id="trace-001",
        time_s=time_s,
        vehicle_id=vehicle_id,
        x_m=x_m,
        y_m=0.0,
        heading_rad=heading_rad,
        route_id=route_id,
        lane_id=lane_id,
        planned_route=planned_route,
    )


def _two_vehicle_frames(
    observations: Iterable[tuple[float, float, float]],
) -> list[VehicleRecord]:
    records: list[VehicleRecord] = []
    for time_s, follower_x, leader_x in observations:
        records.extend(
            (
                _vehicle(time_s, "tx", follower_x),
                _vehicle(time_s, "rx", leader_x),
            )
        )
    return records


def test_extracts_follower_to_leader_with_required_schema() -> None:
    records = _two_vehicle_frames(((0.0, 0.0, 30.0), (0.5, 5.0, 35.0)))

    segments = TaggedPairExtractor().extract(records)

    assert len(segments) == 1
    segment = segments[0]
    assert (segment.tx_id, segment.rx_id) == ("tx", "rx")
    assert segment.initial_distance_m == 30.0
    assert segment.start_s == 0.0
    assert segment.end_s == 0.5
    assert segment.eligibility_reason == "trace_end"
    assert segment.has_intervening_vehicle is False
    assert tuple(segment.to_record()) == (
        "trace_id",
        "pair_id",
        "tx_id",
        "rx_id",
        "start_s",
        "end_s",
        "initial_distance_m",
        "route_id",
        "eligibility_reason",
        "has_intervening_vehicle",
    )


def test_rejects_opposing_and_cross_lane_candidates() -> None:
    records = [
        _vehicle(0.0, "tx", 0.0),
        _vehicle(0.0, "opposing", 30.0, heading_rad=math.pi),
        _vehicle(0.0, "other-lane", 30.0, lane_id="east-b_0"),
    ]

    assert TaggedPairExtractor().extract(records) == ()


def test_records_a_vehicle_observed_between_pair_endpoints() -> None:
    """The blocker flag still works when non-adjacent pairs are allowed."""

    records = [
        _vehicle(0.0, "tx", 0.0),
        _vehicle(0.0, "blocker", 20.0),
        _vehicle(0.0, "rx", 45.0),
        _vehicle(0.5, "tx", 5.0),
        _vehicle(0.5, "blocker", 25.0),
        _vehicle(0.5, "rx", 50.0),
    ]

    segments = TaggedPairExtractor(TaggedPairConfig(require_adjacent=False)).extract(records)

    long_pair = next(pair for pair in segments if (pair.tx_id, pair.rx_id) == ("tx", "rx"))
    adjacent_pair = next(pair for pair in segments if (pair.tx_id, pair.rx_id) == ("tx", "blocker"))
    assert long_pair.has_intervening_vehicle is True
    assert adjacent_pair.has_intervening_vehicle is False


def test_adjacency_is_required_by_default() -> None:
    """A pair straddling another vehicle is not an immediate leader-follower.

    Allowing it produced a same-lane vehicle in the optical path 97-100% of the
    time at 40 and 60 veh/lane-km, leaving V-VLC with no availability at all.
    """

    records = [
        _vehicle(0.0, "tx", 0.0),
        _vehicle(0.0, "blocker", 20.0),
        _vehicle(0.0, "rx", 45.0),
        _vehicle(0.5, "tx", 5.0),
        _vehicle(0.5, "blocker", 25.0),
        _vehicle(0.5, "rx", 50.0),
    ]

    pairs = {(p.tx_id, p.rx_id) for p in TaggedPairExtractor().extract(records)}

    assert ("tx", "rx") not in pairs, "straddling pair must be rejected"
    assert ("tx", "blocker") in pairs, "immediate leader-follower must survive"


def test_only_continuous_one_second_range_excursion_terminates_episode() -> None:
    records = _two_vehicle_frames(
        (
            (0.0, 0.0, 30.0),
            (0.2, 0.0, 70.0),
            (0.9, 0.0, 70.0),
            (1.0, 0.0, 30.0),
            (1.2, 0.0, 70.0),
            (2.1, 0.0, 70.0),
            (2.2, 0.0, 70.0),
        )
    )

    segments = TaggedPairExtractor().extract(records)

    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 2.2
    assert segments[0].eligibility_reason == "outside_range_1s"


def test_route_divergence_ends_at_observed_divergence_time() -> None:
    """Divergence is a temporal event: the leader leaves the follower's path.

    Implementation spec section 8.7 ends an episode when "the routes diverge".
    Two vehicles sharing an edge have not diverged however their itineraries
    differ later, so the leader must actually turn off before the episode ends.
    """

    records = _two_vehicle_frames(((0.0, 0.0, 30.0),))
    records.extend(
        (
            _vehicle(0.5, "tx", 5.0, planned_route=("east-a", "east-b")),
            _vehicle(
                0.5,
                "rx",
                35.0,
                lane_id="north-a_0",
                planned_route=("east-a", "north-a"),
            ),
        )
    )

    segments = TaggedPairExtractor().extract(records)

    assert len(segments) == 1
    assert segments[0].end_s == 0.5
    assert segments[0].eligibility_reason == "route_diverged"


def test_missing_vehicle_and_irregular_max_duration_never_pad_data() -> None:
    missing_records = _two_vehicle_frames(((0.0, 0.0, 30.0),))
    missing_records.append(_vehicle(0.5, "tx", 5.0))

    missing = TaggedPairExtractor().extract(missing_records)
    assert len(missing) == 1
    assert missing[0].end_s == 0.0
    assert missing[0].eligibility_reason == "vehicle_missing"

    irregular = _two_vehicle_frames(((0.0, 0.0, 30.0), (59.5, 10.0, 40.0), (60.2, 20.0, 50.0)))
    capped = TaggedPairExtractor().extract(irregular)
    first = capped[0]
    assert first.eligibility_reason == "max_duration"
    assert first.end_s == 59.5
    assert first.duration_s < 60.0
    assert all(pair.duration_s <= 60.0 for pair in capped)


def test_exact_duplicates_are_deduplicated_and_order_is_deterministic() -> None:
    records = _two_vehicle_frames(((0.0, 0.0, 30.0), (0.5, 5.0, 35.0)))
    forward = TaggedPairExtractor().extract([*records, records[0]])
    reverse = TaggedPairExtractor().extract(reversed([*records, records[0]]))

    assert forward == reverse
    assert len(build_vehicle_frames([*records, records[0]])) == 2


def test_conflicting_duplicate_observation_is_rejected() -> None:
    first = _vehicle(0.0, "tx", 0.0)
    conflict = _vehicle(0.0, "tx", 1.0)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        TaggedPairExtractor().extract((first, conflict))


def test_column_mappings_are_supported_without_dataframe_dependency() -> None:
    rows: list[dict[str, object]] = [
        {
            "trace_id": "trace-001",
            "time_s": time_s,
            "vehicle_id": vehicle_id,
            "x_m": x_m,
            "y_m": 0.0,
            "heading_rad": 0.0,
            "route_id": "route-main",
            "lane_id": "east_0",
            "planned_route": ["east-a", "east-b"],
        }
        for time_s, vehicle_id, x_m in (
            (0.0, "tx", 0.0),
            (0.0, "rx", 30.0),
            (0.5, "tx", 5.0),
            (0.5, "rx", 35.0),
        )
    ]

    segment = TaggedPairExtractor().extract(rows)[0]

    assert segment.tx_id == "tx"
    assert segment.rx_id == "rx"
    assert segment.duration_s == 0.5


def test_structural_trace_reader_records_are_supported_without_pyarrow_import() -> None:
    rows = [
        _TraceLikeRecord("trace-001", 0.0, "tx", 0.0, 0.0, 0.0, "route-main", "east_0"),
        _TraceLikeRecord("trace-001", 0.0, "rx", 30.0, 0.0, 0.0, "route-main", "east_0"),
        _TraceLikeRecord("trace-001", 0.5, "tx", 5.0, 0.0, 0.0, "route-main", "east_0"),
        _TraceLikeRecord("trace-001", 0.5, "rx", 35.0, 0.0, 0.0, "route-main", "east_0"),
    ]

    segment = TaggedPairExtractor().extract(rows)[0]

    assert (segment.tx_id, segment.rx_id, segment.duration_s) == ("tx", "rx", 0.5)
