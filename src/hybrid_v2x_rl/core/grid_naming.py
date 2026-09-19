"""Canonical names for the junctions of a rectangular grid.

Pure string formatting over two indices.  It lives in ``core`` because both
:mod:`hybrid_v2x_rl.mobility.grid_network`, which builds the network, and
:mod:`hybrid_v2x_rl.core.intersection_context`, which the observation layer uses to
locate itself on the map, need the same names.  Putting it in mobility would
have made the junction context import the mobility package, which is the wrong
direction and would have kept a lawful, map-derived quantity out of the
observation layer for no reason beyond where a two-line helper happened to sit.
"""

from __future__ import annotations


def avenue_letter(index: int) -> str:
    """``0`` -> ``"A"``, ``1`` -> ``"B"``, and so on."""

    if index < 0:
        raise ValueError("avenue index must be non-negative")
    return chr(ord("A") + index)


def junction_id(avenue_index: int, cross_street_index: int) -> str:
    """Return the canonical junction identifier, e.g. ``B7``."""

    if cross_street_index < 0:
        raise ValueError("cross-street index must be non-negative")
    return f"{avenue_letter(avenue_index)}{cross_street_index}"


__all__ = [
    "avenue_letter",
    "junction_id",
]
