"""Stream a stored trace into pair instants the rollout can evaluate.

The piece between a 412 MB trace on disk and one packet in the lifecycle. It
exists separately from :mod:`hybrid_v2x_rl.env.rollout` because the two answer
different questions: the rollout knows how to turn a pose into an outcome, this
knows how to get poses out of a trace without loading it.

**Streaming is not an optimization here, it is the only option.** A single
density-30 trace holds twenty million vehicle-state rows, and the campaign has
twenty-one of them. The vehicle parts are written in time order, so one pass
with a single live frame is enough -- and a frame is discarded the moment its
timestamp advances, which bounds memory at the number of vehicles alive at one
instant rather than the number in the trace.

**Packets are generated on the service period, not on the trace timestep.** The
trace is sampled finely enough to resolve car-following; the service generates a
packet every ``generation_period_s``. Evaluating one packet per trace frame
would silently inflate every count by the ratio between them and make a 1e-4
budget look like it was met on ten times the evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.geometry.spatial_index import SpatialIndex

#: The vehicle-state columns a pose needs. Reading only these is worth doing:
#: the unread columns are most of the file.
VEHICLE_COLUMNS = (
    "time_s",
    "vehicle_id",
    "x_m",
    "y_m",
    "heading_rad",
    "speed_mps",
    "length_m",
    "width_m",
    "height_m",
)


class EpisodeError(HybridV2XError):
    """A trace could not be streamed as pair instants."""


class VehiclePose:
    """One vehicle at one instant, carrying only what the geometry engine reads."""

    __slots__ = (
        "vehicle_id", "x_m", "y_m", "heading_rad", "speed_mps",
        "length_m", "width_m", "height_m",
    )

    def __init__(self, row: dict) -> None:
        self.vehicle_id = row["vehicle_id"]
        self.x_m = row["x_m"]
        self.y_m = row["y_m"]
        self.heading_rad = row["heading_rad"]
        self.speed_mps = row["speed_mps"]
        self.length_m = row["length_m"]
        self.width_m = row["width_m"]
        self.height_m = row["height_m"]


@dataclass(frozen=True, slots=True)
class PairInstant:
    """One tagged pair at one packet-generation instant."""

    trace_id: str
    pair_id: str
    index: int
    time_s: float
    transmitter: VehiclePose
    receiver: VehiclePose
    neighbours: Sequence[VehiclePose]
    index_of_frame: SpatialIndex
    #: True on the pair's last evaluated instant, so a caller knows when the
    #: rollout's per-pair correlated state can be released.
    final: bool


@dataclass(frozen=True, slots=True)
class TraceSource:
    """A trace directory, read lazily."""

    path: Path
    trace_id: str
    density: float

    @classmethod
    def discover(cls, path: str | Path) -> TraceSource:
        """Identify a trace from its directory, taking density from the name.

        The density group is a label the constraint is enforced per, so it has
        to come from somewhere durable. The trace id carries it (``-d30-``) and
        the manifest does not, so it is parsed here with a loud failure rather
        than defaulted -- an unlabelled trace silently joining the wrong density
        group would move a constraint without changing a number anywhere.
        """

        path = Path(path)
        trace_id = path.name
        for token in trace_id.split("-"):
            if token.startswith("d") and token[1:].isdigit():
                return cls(path=path, trace_id=trace_id, density=float(token[1:]))
        raise EpisodeError(
            "cannot read a density group from the trace id",
            context={"trace_id": trace_id},
        )


def _pair_windows(path: Path) -> dict[str, list[tuple[str, str, float, float]]]:
    """Tagged-pair windows, bucketed by transmitter so a frame lookup is cheap."""

    table = pq.read_table(
        path / "pairs.parquet",
        columns=["pair_id", "tx_id", "rx_id", "start_s", "end_s"],
    )
    windows: list[tuple[str, str, str, float, float]] = list(
        zip(
            table.column("pair_id").to_pylist(),
            table.column("tx_id").to_pylist(),
            table.column("rx_id").to_pylist(),
            table.column("start_s").to_pylist(),
            table.column("end_s").to_pylist(),
            strict=True,
        )
    )
    return windows


def iter_pair_instants(
    source: TraceSource,
    *,
    generation_period_s: float,
    max_packets: int | None = None,
    warmup_s: float = 0.0,
) -> Iterator[PairInstant]:
    """Yield every tagged pair at every packet-generation instant.

    ``max_packets`` truncates rather than subsamples, and it overshoots to the
    end of the frame it lands in. Truncation keeps whole pair episodes intact up
    to the cut, which matters because the shadowing and fading states are
    correlated along an episode: a subsample that skipped instants would
    decorrelate them and quietly report a link that recovers faster than it
    does. Finishing the frame matters for the same kind of reason -- a partial
    frame would report contender counts drawn from a partial population.

    **Truncation is biased and ``warmup_s`` is how you pay for it.** A trace
    starts with vehicles in their seeded formation -- evenly spaced, aligned,
    every pair pointing straight down its own lane. Measured on the first few
    thousand packets the optical geometric outage reads 4.6% at density 10; over
    the whole trace it is 17.4%, because the traffic has to disperse and turn
    before the geometry is representative of anything. Any truncated run must
    skip that transient, and no default can know how long it is, so the caller
    states it.
    """

    windows = _pair_windows(source.path)
    if not windows:
        return
    windows.sort(key=lambda w: w[3])
    starts = [w[3] for w in windows]
    last_index: dict[str, int] = {}
    emitted = 0

    frame: dict[str, VehiclePose] = {}
    current: float | None = None
    next_packet_s = -math.inf

    from bisect import bisect_right

    def emit(time_s: float) -> Iterator[PairInstant]:
        nonlocal emitted
        active = []
        for pair_id, tx_id, rx_id, _start_s, end_s in windows[: bisect_right(starts, time_s)]:
            if end_s < time_s:
                continue
            tx, rx = frame.get(tx_id), frame.get(rx_id)
            if tx is None or rx is None:
                continue
            active.append((pair_id, tx, rx, end_s))
        if not active:
            return
        neighbours = tuple(frame.values())
        index_of_frame = SpatialIndex.build(neighbours)
        for pair_id, tx, rx, end_s in active:
            index = last_index.get(pair_id, -1) + 1
            last_index[pair_id] = index
            emitted += 1
            yield PairInstant(
                trace_id=source.trace_id,
                pair_id=pair_id,
                index=index,
                time_s=time_s,
                transmitter=tx,
                receiver=rx,
                neighbours=neighbours,
                index_of_frame=index_of_frame,
                final=end_s < time_s + generation_period_s,
            )

    for part in sorted((source.path / "vehicles").glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(
            batch_size=65_536, columns=list(VEHICLE_COLUMNS)
        ):
            for row in batch.to_pylist():
                time_s = row["time_s"]
                if time_s != current:
                    if current is not None and current >= next_packet_s:
                        if current >= warmup_s:
                            yield from emit(current)
                        next_packet_s = current + generation_period_s
                        if max_packets is not None and emitted >= max_packets:
                            return
                    elif next_packet_s == -math.inf:
                        next_packet_s = time_s
                    frame = {}
                    current = time_s
                frame[row["vehicle_id"]] = VehiclePose(row)

    if current is not None and current >= next_packet_s:
        yield from emit(current)


__all__ = [
    "VEHICLE_COLUMNS",
    "EpisodeError",
    "PairInstant",
    "TraceSource",
    "VehiclePose",
    "iter_pair_instants",
]
