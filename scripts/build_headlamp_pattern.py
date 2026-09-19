"""Build a smooth, ECE R112-compliant headlamp radiant-intensity artifact.

Run as::

    python scripts/build_headlamp_pattern.py [--out artifacts/calibration/vlc]

**Why this replaces the test-point artifact.** UN Regulation No. 112 is a
*compliance envelope*, not a beam. It states nine point limits, four zone
limits and eight grouped-point minima, all inside 9L..9R horizontally and
4U..4D vertically. A grid built directly from those points is a sparse scatter
of spikes embedded in whatever the author chose to put between them, and
bilinear interpolation across it produces a beam shape that is an artefact of
the sampling grid rather than a property of any lamp. The previous artifact
did exactly that, and it also mis-stated Zone IV (1,875 cd against the
regulation's 2,500), placed 75R's horizontal coordinate on the vertical axis,
and applied the Zone IV floor far outside Zone IV's actual box.

The model here is instead a smooth two-lobe intensity distribution with a
sharp cut-off, *fitted so that every R112 Class B limit is satisfied by
construction*, then sampled on a dense grid. Compliance is verified
numerically before the artifact is written; the script fails rather than
emitting a non-compliant pattern.

**What it is and is not.** It is a plausible compliant design. It is not a
measured lamp, and it must not be described as one. Where a measured pattern
is available -- Alsalami et al. or Memedi et al. publish them -- it should
replace this.

**Two variants, because the regulation stops at 9 degrees.** Beyond 9L/9R the
regulation imposes no minimum, so what a lamp emits there is a design choice.
The ``wide`` variant lets the model's own falloff continue; the ``narrow``
variant applies a steeper decay past the last regulated angle. Reporting both
turns the unregulated region from a hidden assumption into a stated band.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

# --- ECE R112 Rev.4, para 6.2.4, RH traffic, Class B ------------------------
# (name, vertical_deg [+ = up], horizontal_deg [+ = right], min_cd, max_cd)
POINTS = [
    ("B 50 L", +0.57, -3.43, None,   350),
    ("BR",     +1.00, +2.50, None,  1750),
    ("75 R",   -0.57, +1.15, 10100, None),
    ("75 L",   -0.57, -3.43, None, 10600),
    ("50 L",   -0.86, -3.43, None, 13200),
    ("50 R",   -0.86, +1.72, 10100, None),
    ("50 V",   -0.86,  0.00,  5100, None),
    ("25 L",   -1.72, -9.00,  1700, None),
    ("25 R",   -1.72, +9.00,  1700, None),
]
#: Grouped minima from the second table (para 6.2.4).
GROUPS = [
    ("points 1+2+3", [(+4.0, -8.0), (+4.0, 0.0), (+4.0, +8.0)], 190),
    ("points 4+5+6", [(+2.0, -4.0), (+2.0, 0.0), (+2.0, +4.0)], 375),
]
SINGLES = [("point 7", 0.0, -8.0, 65), ("point 8", 0.0, -4.0, 125)]

ZONE_IV = dict(v=(-1.72, -0.86), h=(-5.15, +5.15), minimum=2500)
ZONE_I = dict(v=(-4.0, -1.72), h=(-9.0, +9.0))          # max = 2 x I(50R)
ZONE_III_VERTS = [(+1.0, -8.0), (+4.0, -8.0), (+4.0, +8.0), (+2.0, +8.0),
                  (+1.5, +6.0), (+1.5, +1.5), (0.0, 0.0), (0.0, -4.0)]
ZONE_III_MAX = 625

LUMINOUS_EFFICACY_LM_PER_W = 300.0
REGULATED_H_DEG = 9.0


# --- the beam model ---------------------------------------------------------

def cutoff_height_deg(h: float) -> float:
    """Vertical position of the cut-off at horizontal angle ``h``.

    Flat at 0.57D to the left, rising through the elbow into the shoulder on
    the right -- the asymmetry R112 para 6.2.1 requires of a right-hand-traffic
    passing beam, and the single feature that most distinguishes a real low
    beam from a cosine lobe.
    """

    if h <= 0.0:
        return -0.57
    return -0.57 + 1.30 * math.tanh(h / 2.2)


def _gauss(h, v, amp, h0, v0, sh, sv):
    return amp * math.exp(-0.5 * (((h - h0) / sh) ** 2 + ((v - v0) / sv) ** 2))


def intensity_cd(h: float, v: float, *, narrow: bool = False) -> float:
    """Luminous intensity in candela at horizontal ``h``, vertical ``v`` (deg)."""

    # Sharp suppression above the cut-off: glare control is the reason a
    # passing beam exists, and R112 caps B50L at 350 cd barely half a degree
    # above a region required to deliver 10,100.
    edge = (v - cutoff_height_deg(h)) / 0.115
    above = 1.0 / (1.0 + math.exp(min(60.0, edge)))

    hot = _gauss(h, v, 26000.0, +1.45, -1.00, 2.95, 0.98)
    spread = _gauss(h, v, 3600.0, +0.90, -1.90, 7.20, 3.20)
    foreground = _gauss(h, v, 1500.0, 0.00, -5.50, 11.0, 4.50)

    # Stray light above the cut-off, as an additive lobe rather than a fraction
    # of the main beam. It has to be additive: points 1-6 sit at 2U and 4U
    # where the main beam is suppressed by four orders of magnitude, yet they
    # carry grouped minima of 375 and 190 cd. Real lamps do emit here, and the
    # glare limits (B50L 350, zone III 625) exist precisely to bound it -- so
    # this term is simultaneously the thing the minima require and the thing
    # the maxima constrain.
    stray = _gauss(h, v, 245.0, 0.00, +0.60, 12.0, 4.20)

    value = above * (hot + spread + foreground) + stray

    if narrow and abs(h) > REGULATED_H_DEG:
        # Steeper decay past the last regulated angle -- the other end of the
        # band, not a different physics.
        value *= math.exp(-((abs(h) - REGULATED_H_DEG) / 3.0) ** 2)
    return value


# --- compliance verification ------------------------------------------------

def _zone_iii_contains(h: float, v: float) -> bool:
    if not (0.0 <= v <= 4.0 and -8.0 <= h <= 8.0):
        return False
    if v >= 2.0:
        return True
    if h <= 0.0:
        return v >= 0.0 and h >= -8.0 and v >= (0.0 if h >= -4.0 else 1.0)
    return v >= 1.5


def verify(narrow: bool) -> list[str]:
    """Return a list of R112 violations; empty means compliant."""

    bad: list[str] = []
    intensity_at = lambda h, v: intensity_cd(h, v, narrow=narrow)  # noqa: E731

    for name, v, h, lo, hi in POINTS:
        got = intensity_at(h, v)
        if lo is not None and got < lo:
            bad.append(f"{name}: {got:.0f} cd < min {lo}")
        if hi is not None and got > hi:
            bad.append(f"{name}: {got:.0f} cd > max {hi}")

    for name, coords, lo in GROUPS:
        total = sum(intensity_at(h, v) for v, h in coords)
        if total < lo:
            bad.append(f"{name}: sum {total:.0f} cd < min {lo}")

    for name, v, h, lo in SINGLES:
        got = intensity_at(h, v)
        if got < lo:
            bad.append(f"{name}: {got:.0f} cd < min {lo}")

    lo_v, hi_v = ZONE_IV["v"]
    lo_h, hi_h = ZONE_IV["h"]
    worst = min(
        intensity_at(lo_h + (hi_h - lo_h) * i / 40, lo_v + (hi_v - lo_v) * j / 12)
        for i in range(41) for j in range(13)
    )
    if worst < ZONE_IV["minimum"]:
        bad.append(f"zone IV: worst {worst:.0f} cd < min {ZONE_IV['minimum']}")

    cap = 2.0 * intensity_at(+1.72, -0.86)
    lo_v, hi_v = ZONE_I["v"]
    lo_h, hi_h = ZONE_I["h"]
    peak = max(
        intensity_at(lo_h + (hi_h - lo_h) * i / 60, lo_v + (hi_v - lo_v) * j / 20)
        for i in range(61) for j in range(21)
    )
    if peak > cap:
        bad.append(f"zone I: peak {peak:.0f} cd > cap {cap:.0f} (2x I(50R))")

    peak3 = max(
        (intensity_at(h / 4, v / 4) for v in range(0, 17) for h in range(-32, 33)
         if _zone_iii_contains(h / 4, v / 4)),
        default=0.0,
    )
    if peak3 > ZONE_III_MAX:
        bad.append(f"zone III: peak {peak3:.0f} cd > max {ZONE_III_MAX}")
    return bad


# --- artifact ---------------------------------------------------------------

def build(narrow: bool) -> dict:
    horizontal = [round(-25.0 + 0.5 * i, 3) for i in range(101)]     # -25..+25
    vertical = [round(-12.0 + 0.25 * i, 3) for i in range(65)]       # -12..+4

    grid = [
        [max(0.0, intensity_cd(h, v, narrow=narrow)) / LUMINOUS_EFFICACY_LM_PER_W
         for h in horizontal]
        for v in vertical
    ]
    provenance = [
        [
            "r112_regulated_domain"
            if abs(h) <= REGULATED_H_DEG and -4.0 <= v <= 4.0
            else "modelled_beyond_regulated_domain"
            for h in horizontal
        ]
        for v in vertical
    ]
    variant = "narrow" if narrow else "wide"
    return {
        "schema": "hybrid-rf-vlc-rl.vlc.headlamp.v1",
        "pattern_id": f"headlamp-r112-compliant-{variant}-v2",
        "units": "W/sr",
        "luminous_efficacy_lm_per_w": LUMINOUS_EFFICACY_LM_PER_W,
        "screen_distance_m": 25.0,
        "source": (
            "MODELLED, not measured. A smooth two-lobe intensity distribution with "
            "an asymmetric cut-off, fitted so that every UN Regulation No. 112 "
            "Rev.4 Class B passing-beam limit (para 6.2.4: nine test points, the "
            "grouped minima at points 1-8, and zones I, III and IV) is satisfied by "
            "construction. Compliance is verified numerically by "
            "scripts/build_headlamp_pattern.py before the artifact is written. "
            "R112 constrains only 9L..9R by 4U..4D; cells outside that box are the "
            "model's own extrapolation and are labelled "
            "'modelled_beyond_regulated_domain'. This is a plausible compliant "
            f"design, not a measured lamp. Variant '{variant}': "
            + ("a steeper decay is applied beyond 9 degrees horizontally."
               if narrow else
               "the model's own falloff continues beyond 9 degrees horizontally.")
        ),
        "regulated_domain_deg": {"horizontal": [-9.0, 9.0], "vertical": [-4.0, 4.0]},
        "horizontal_angles_rad": [math.radians(x) for x in horizontal],
        "vertical_angles_rad": [math.radians(x) for x in vertical],
        "intensity_w_per_sr": grid,
        "cell_provenance": provenance,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("artifacts/calibration/vlc"))
    args = ap.parse_args()

    ok = True
    for narrow in (False, True):
        variant = "narrow" if narrow else "wide"
        problems = verify(narrow)
        print(f"[{variant}] R112 Class B compliance: "
              f"{'PASS' if not problems else str(len(problems)) + ' VIOLATIONS'}")
        for p in problems:
            print(f"    {p}")
        if problems:
            ok = False
            continue
        payload = build(narrow)
        d = args.out / payload["pattern_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(payload, indent=1))
        print(f"    wrote {d/'manifest.json'}")
    if not ok:
        print("\nrefusing to emit a non-compliant pattern")
        return 1

    print("\nkey values of the wide variant (cd):")
    for name, v, h, lo, hi in POINTS:
        got = intensity_cd(h, v)
        limit = f"min {lo}" if lo else f"max {hi}"
        print(f"  {name:<7} ({v:+5.2f},{h:+6.2f})  {got:>8.0f}   [{limit}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
