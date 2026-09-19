"""Work plan section 5: exact geometry must not reach a deployable policy.

Everything the geometry engine computes is *hidden simulator state*.  It exists
to generate outcomes and to answer oracle questions.  The learned and
deterministic policies see a noisy, aged reconstruction — the sensor tracks and
blockage forecast that complete M2 in week 7 — never these objects.

Three separate guards, because they fail in different ways:

1. the configuration cannot *name* a forbidden observation field;
2. the geometry layer cannot be reached *from* the observation layer;
3. the quantities themselves are recognisable, so a future field that smuggles
   one in under another name is still caught by review of this list.

Section 5.3's module-boundary tests live beside the modules they constrain;
this file covers the policy boundary rather than the channel boundary.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import hybrid_v2x_rl.geometry
from hybrid_v2x_rl.config import headline_config_layers, load_config, load_headline_config
from hybrid_v2x_rl.config.validation import FORBIDDEN_OBSERVATION_FIELDS
from hybrid_v2x_rl.core.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Every quantity the geometry engine produces that would trivialise the task.
#: Naming them here is the point: an addition to the geometry layer that is not
#: obviously safe should be added to this list and then justified.
EXACT_GEOMETRY_QUANTITIES = frozenset(
    {
        "blocker_ids",
        "nearest_blocker_id",
        "building_blocked",
        "is_blocked",
        "is_obstructed",
        "vehicle_blocker_ids",
        "within_field_of_view",
    }
)


def geometry_module_names() -> list[str]:
    return [
        f"hybrid_v2x_rl.geometry.{info.name}" for info in pkgutil.iter_modules(hybrid_v2x_rl.geometry.__path__)
    ]


def imported_names(module_name: str) -> set[str]:
    """Modules actually imported by ``module_name``, parsed rather than grepped.

    A text search matches prose in docstrings, which is exactly what this file
    discusses at length, so the guard would fire on its own explanations.
    """

    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


# -- 1. the configuration cannot name hidden state ---------------------------


def test_the_headline_configuration_hides_exact_geometry() -> None:
    config = load_headline_config(PROJECT_ROOT)
    assert config.geometry.exact_geometry_hidden_from_policy


@pytest.mark.parametrize("field", sorted(FORBIDDEN_OBSERVATION_FIELDS))
def test_every_forbidden_observation_field_is_rejected(field: str, tmp_path: Path) -> None:
    """The loader refuses each one by name, not just the first."""

    override = tmp_path / f"leak_{field}.yaml"
    override.write_text(
        f"observation:\n  features:\n    - rf_quality\n    - {field}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="forbidden hidden field"):
        load_config(
            (*headline_config_layers(PROJECT_ROOT), override),
            project_root=PROJECT_ROOT,
        )


def test_the_forbidden_list_covers_what_the_geometry_engine_produces() -> None:
    """Occlusion, blocker identity and alignment truth must all be named.

    The engine grew after this list was written, so this checks the list kept
    up rather than assuming it did.
    """

    joined = " ".join(FORBIDDEN_OBSERVATION_FIELDS)
    for topic in ("blocker", "blockage", "geometry", "channel_state", "counterfactual"):
        assert topic in joined, f"no forbidden field mentions {topic!r}"


# -- 2. the observation layer cannot reach the geometry layer ----------------


def test_no_observation_module_imports_the_geometry_engine() -> None:
    """Week 7 builds observations from noisy tracks, never from exact geometry.

    Installed while the observation package was still empty, so its first
    module could not take the shortcut.  It now guards six modules, and
    ``tests/unit/test_observation_pipeline.py`` re-states it against the
    package as it actually stands.
    """

    observation_root = PROJECT_ROOT / "src" / "hybrid_v2x_rl" / "observation"
    sources = sorted(observation_root.rglob("*.py"))

    offenders: list[str] = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.startswith("hybrid_v2x_rl.geometry") for name in names):
                offenders.append(str(source.relative_to(PROJECT_ROOT)))
                break

    assert not offenders, (
        f"observation code must not import the geometry engine; offending files: {offenders}"
    )


# -- 3. the engine keeps its results out of anything policy-facing -----------


def test_geometry_results_are_plain_values_and_carry_no_policy_hooks() -> None:
    """Nothing in the geometry layer offers to serialise itself for a policy.

    An ``as_observation`` or ``to_features`` helper is how exact state usually
    escapes: it looks convenient and nobody re-reads the leakage rule.
    """

    forbidden_helpers = ("as_observation", "to_features", "as_policy_input", "observe")

    for name in geometry_module_names():
        module = importlib.import_module(name)
        source = Path(module.__file__ or "").read_text(encoding="utf-8")
        for helper in forbidden_helpers:
            assert f"def {helper}" not in source, f"{name} exposes {helper}"


def test_core_geometry_depends_on_nothing_in_the_project() -> None:
    """The primitives are usable by the observation layer only while pure.

    They sit in ``hybrid_v2x_rl.core`` rather than ``hybrid_v2x_rl.geometry`` precisely so
    the leakage guard above can stay absolute: pure convex mathematics over
    whatever coordinates it is handed has no more access to hidden state than
    ``math`` does.  One convenience import from elsewhere in the project would
    dissolve that argument without any test noticing, so this is the test that
    notices.
    """

    imported = imported_names("hybrid_v2x_rl.core.geometry")

    offenders = sorted(name for name in imported if name.startswith("hybrid_v2x_rl"))
    assert not offenders, (
        f"core geometry must stay free of project dependencies; found {offenders}"
    )


def test_the_geometry_package_does_not_re_export_what_moved_to_core() -> None:
    """One obvious import path per symbol.

    This hides nothing on its own -- the guard above already forbids importing
    this package at all -- but a symbol reachable from two layers invites code
    that is unclear about which one it belongs to, and it would let a future
    relaxation of the guard quietly widen its own reach.
    """

    moved = (
        "Point",
        "Segment",
        "OrientedRectangle",
        "segment_intersects_rectangle",
        "LinkPath",
        "optical_link_path",
        "rf_link_path",
    )
    smuggled = [name for name in moved if hasattr(hybrid_v2x_rl.geometry, name)]
    assert not smuggled, (
        f"hybrid_v2x_rl.geometry must not re-export what now lives in core; found {smuggled}"
    )


def test_core_link_endpoints_depends_on_nothing_stateful() -> None:
    """The mounting convention travels with the primitives, for the same reason.

    Where a headlamp or photodiode sits on a body is hardware geometry.  The
    predictor needs the identical follower-front-to-leader-rear rule the
    occlusion engine uses, and two copies of it could drift apart without any
    test noticing.
    """

    imported = imported_names("hybrid_v2x_rl.core.link_endpoints")
    offenders = sorted(
        name for name in imported if name.startswith("hybrid_v2x_rl") and "core." not in name
    )
    assert not offenders, f"core link endpoints reached outside core: {offenders}"


def test_intersection_context_reasons_about_the_map_and_nothing_hidden() -> None:
    """It is the map-derived quantity the policy is allowed to have.

    Junction distance and span come from own pose plus a road map, which a real
    vehicle carries.  Every other geometry result depends on where *other*
    vehicles are, which it does not.  That is why this module moved to ``core``
    with the primitives rather than needing an exception carved into the guard
    above: it does not belong to the exact-state layer at all, and the
    two-line junction naming helper that once tied it to ``mobility`` moved to
    ``core.grid_naming`` for the same reason.
    """

    imported = imported_names("hybrid_v2x_rl.core.intersection_context")

    for forbidden in ("vehicle_occlusion", "rf_visibility", "spatial_index", "building_geometry"):
        assert not any(forbidden in name for name in imported), (
            f"intersection context reached {forbidden}: {sorted(imported)}"
        )
    outside_core = sorted(
        name for name in imported if name.startswith("hybrid_v2x_rl") and not name.startswith("hybrid_v2x_rl.core")
    )
    assert not outside_core, f"intersection context reached outside core: {outside_core}"
    # It may still reason about the road and about the link's own endpoints.
    assert any("core.geometry" in name or "grid_naming" in name for name in imported)


def test_no_vlc_module_imports_the_rf_block_error_model() -> None:
    """Work plan section 7.3 forbids the transfer, so a test enforces it.

    IM/DD optical detection is not a complex-AWGN channel and a generic
    finite-blocklength expression does not apply to it. The implementation spec
    states the rule as "VLC does not import this file"; stating it in prose is
    how the leakage barrier was enforced before it was a test, and that did not
    hold either.
    """

    import ast
    from pathlib import Path

    vlc_root = Path("src/hybrid_v2x_rl/channels/vlc")
    offenders = []
    for module in vlc_root.rglob("*.py"):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "channels.rf.bler" in node.module:
                    offenders.append(module)
            elif isinstance(node, ast.Import):
                if any("channels.rf.bler" in alias.name for alias in node.names):
                    offenders.append(module)
    assert not offenders, f"VLC modules must not import the RF BLER model: {offenders}"
