"""The optical front end: what the photodiode collects, and what it costs to widen.

Implementation spec section 12.2.  Everything here is a property of the
*detector*.  Where the light comes from is geometry's business
(:mod:`hybrid_v2x_rl.core.pair_geometry`), what arrives is
:mod:`hybrid_v2x_rl.channels.vlc.optical_gain`'s, and whether what arrives is legible
is ``noise.py``'s.  This module holds the thermal-noise *parameters* the spec
lists and forms no noise power from them, so that the night and day ambient
configurations of section 12.4 remain separate configurations rather than a
runtime boolean reaching in here for constants.

**Two constants are sourced; the rest are not, and each says which.**  Active
area and junction capacitance come from the Vishay BPW34 datasheet (document
81521) and are marked SOURCED.  Everything else carries ``DECLARED`` or
``UNVERIFIED``, exactly as the NLOSv blockage coefficients in
:mod:`hybrid_v2x_rl.channels.rf.pathloss_37885` do.
They are order-of-magnitude plausible for a silicon PIN front end driven by a
phosphor-converted white headlamp and nothing stronger should be read into
them.  Naming them makes a correction one edit and stops an invented number
reading as a measurement.

**One semi-angle, imported rather than repeated.**  ``receiver_fov_deg`` is
read as the receiver semi-angle psi_c, and it is used twice: geometry tests
acceptance against it, and the concentrator gain goes as ``n^2 / sin^2 psi_c``.
The configuration warns that a wide cone for availability combined with
narrow-cone gain in the link budget counts the same physics twice, so the
default here is *imported* from the geometry layer rather than restated.  There
is one number in the codebase and the two uses cannot drift apart.

**No field-of-view cutoff lives here.**  :func:`OpticalReceiver.effective_area_m2`
is continuous across psi_c and keeps falling as ``cos psi`` beyond it.  That is
deliberate: the acceptance test is applied in exactly one place, and a second
copy of it here would zero the same rays twice while looking like a more careful
model.  The consequence is that this module's output is *conditional on the
caller having applied the acceptance test* -- beyond psi_c an ideal
non-imaging concentrator delivers nothing, and the number returned here is one
the physical device cannot produce.  A caller that asks for effective area at
80 degrees is asking a question geometry already answered.

**Detector area is not a free parameter.**  This is the structural finding of
the module and it constrains the whole optical design.  Junction capacitance
scales with active area, and the front end's RC pole must stay above the
configured electrical bandwidth, so the load resistance is forced down in
proportion as the area grows: ``R_L <= 1 / (2 pi B C)`` with ``C = c_A A``.
Thermal noise variance goes as ``1 / R_L``, hence as ``A``.  Signal
photocurrent goes as ``A``, so electrical signal power goes as ``A^2``.  The
ratio therefore goes as ``A``, not ``A^2``:

    doubling the detector buys 3 dB electrical, not 6.

Concentrator gain does not pay that tax -- it raises the collected power
without touching the junction capacitance -- so the same factor ``g`` buys
``g^2``.  Read against the configuration's own sensitivity, narrowing psi_c
from 60 to 30 degrees is 4.8 dB optical and 9.5 dB electrical, and *buying that
back with raw area instead would take a nine-fold larger detector*, whose
capacitance would then cap the load resistance near 51 ohm against the 455 ohm
the 7.5 mm^2 part allows at 5 MHz.  The wide cone is expensive in a way a
bigger photodiode cannot fix.
:func:`OpticalReceiver.max_load_resistance_ohm` makes the ceiling computable and
:func:`OpticalReceiver.check_front_end_feasible` refuses a receiver that
violates it, so this stays a constraint rather than a paragraph.

**Two idealizations, both optimistic, both stated rather than hidden.**  The
concentrator gain used here is the thermodynamic limit for a non-imaging
concentrator; a real compound parabolic concentrator reaches perhaps 90% of it,
so the budget is roughly 0.5 dB optical optimistic.  And the filter
transmission is held independent of incidence angle, which is right for a
broadband long-pass and wrong for a narrow dielectric bandpass, whose passband
blue-shifts with angle and can collapse near 60 degrees.  Note which way that
second one points: it penalises the wide cone specifically, so an
angle-dependent filter would erode the 60-degree option further rather than
rescue it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.link_endpoints import DEFAULT_PHOTODIODE_HEIGHT_M
from hybrid_v2x_rl.core.pair_geometry import DEFAULT_FOV_HALF_ANGLE_RAD, wrap_to_pi

#: Radiant sensitive area of a Vishay BPW34 silicon PIN photodiode, 7.5 mm^2.
#: SOURCED -- Vishay document 81521, a commodity part widely used in VLC
#: receivers.  The earlier value was 1.0e-4 m^2, a round 100 mm^2 chosen as an
#: order of magnitude; that is **13x this part** and therefore 11.25 dB
#: optimistic, because received power scales linearly with area.
#:
#: Larger single photodiodes exist, but area is not free: junction capacitance
#: scales with it and forces the load resistance down to keep the bandwidth,
#: so thermal-limited SNR grows as A rather than A^2.  Choosing a commodity
#: part with a published datasheet is worth more here than choosing a
#: favourable one without.
ACTIVE_AREA_M2 = 7.5e-6

#: Silicon responsivity at the phosphor-converted white LED wavelengths a
#: headlamp emits, roughly 550-600 nm.  DECLARED, and consistent with the
#: BPW34 datasheet: 50 uA at 1 mW/cm^2 over 7.5 mm^2 is 0.67 A/W at 900 nm,
#: and silicon responsivity falls to roughly this value in the visible.  It is
#: deliberately not the 900 nm peak, because the headlamp does not emit there.
RESPONSIVITY_A_PER_W = 0.4

#: In-band transmission of the optical filter that rejects ambient light.
#: Held independent of incidence angle; see the module docstring.  UNVERIFIED.
FILTER_TRANSMISSION = 0.9

#: Concentrator refractive index -- PMMA and optical glass both sit near 1.5.
#: This is the one constant here whose order of magnitude is not in doubt,
#: because the range of transparent dielectrics is narrow.  UNVERIFIED.
CONCENTRATOR_REFRACTIVE_INDEX = 1.5

#: BPW34 junction capacitance is 70 pF at 1 MHz with zero reverse bias, over
#: 7.5 mm^2, giving 9.33e-6 F/m^2.  SOURCED -- Vishay document 81521.  Zero
#: bias is the pessimistic end: reverse-biasing the diode lowers capacitance
#: substantially and a real receiver would do so, so the front-end bandwidth
#: check this feeds is conservative.
DETECTOR_CAPACITANCE_PER_AREA_F_PER_M2 = 9.33e-6

#: Transimpedance load resistance, ohm.  **Not independently chosen**: it is
#: bounded above by the RC pole of the configured 5 MHz front end with the
#: 7.5 mm^2 BPW34, which is 455 ohm.  250 ohm leaves a comfortable margin.  Raising it
#: lowers thermal noise and narrows the front end, and
#: :func:`OpticalReceiver.check_front_end_feasible` is where that trade is
#: refused rather than absorbed.  UNVERIFIED.
DEFAULT_LOAD_RESISTANCE_OHM = 250.0

#: Front-end absolute temperature, K.  A receiver behind a rear window in
#: summer runs hotter than this, but thermal noise is linear in temperature so
#: 298 K against 348 K is 0.7 dB -- the weakest sensitivity in the file.
#: UNVERIFIED.
DEFAULT_NOISE_TEMPERATURE_K = 298.0

#: Dimensionless excess-noise factor multiplying the ``4kT/R_L`` resistor term,
#: lumping the preamplifier's channel and gate noise.  The alternative is the
#: Kahn-Barry FET transimpedance model, which introduces six further constants
#: -- transconductance, channel noise factor, gate leakage, two noise-bandwidth
#: integrals and an open-loop gain -- none of which is sourced either.  Lumping
#: is the honest choice while the parameter table is empty; ``noise.py`` may
#: replace it with the structured model once there is something to cite.
#: UNVERIFIED.
DEFAULT_PREAMPLIFIER_NOISE_FACTOR = 1.5


class ReceiverError(HybridV2XError):
    """An optical front end was specified or interrogated inconsistently."""


@dataclass(frozen=True, slots=True)
class OpticalReceiver:
    """The rear-facing photodiode assembly of one vehicle.

    Defaults are the module constants, so a caller that wants the configured
    front end writes ``OpticalReceiver()`` and a caller that wants a variant
    names the field it is varying.  Every default is unsourced; see the module
    docstring.
    """

    active_area_m2: float = ACTIVE_AREA_M2
    responsivity_a_per_w: float = RESPONSIVITY_A_PER_W
    filter_transmission: float = FILTER_TRANSMISSION
    concentrator_refractive_index: float = CONCENTRATOR_REFRACTIVE_INDEX

    #: The receiver semi-angle psi_c, defaulting to the *same* value the
    #: acceptance test uses.  Imported, not restated; see the module docstring.
    fov_half_angle_rad: float = DEFAULT_FOV_HALF_ANGLE_RAD

    #: Mounting height, matching the headlamp so the optical path is horizontal
    #: and occlusion reduces to a footprint test plus a height comparison.
    height_m: float = DEFAULT_PHOTODIODE_HEIGHT_M

    load_resistance_ohm: float = DEFAULT_LOAD_RESISTANCE_OHM
    noise_temperature_k: float = DEFAULT_NOISE_TEMPERATURE_K
    preamplifier_noise_factor: float = DEFAULT_PREAMPLIFIER_NOISE_FACTOR
    capacitance_per_area_f_per_m2: float = DETECTOR_CAPACITANCE_PER_AREA_F_PER_M2

    def __post_init__(self) -> None:
        positive = {
            "active_area_m2": self.active_area_m2,
            "responsivity_a_per_w": self.responsivity_a_per_w,
            "load_resistance_ohm": self.load_resistance_ohm,
            "noise_temperature_k": self.noise_temperature_k,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ReceiverError(f"{name} must be finite and positive",
                                    context={name: value})
        if not 0.0 < self.filter_transmission <= 1.0:
            raise ReceiverError(
                "filter transmission is a fraction in (0, 1]; a filter cannot pass "
                "more light than reaches it",
                context={"filter_transmission": self.filter_transmission},
            )
        if not self.concentrator_refractive_index >= 1.0:
            raise ReceiverError(
                "refractive index below 1 would make the concentrator gain less than "
                "the bare aperture, which is not a physical dielectric",
                context={"n": self.concentrator_refractive_index},
            )
        if not 0.0 < self.fov_half_angle_rad <= 0.5 * math.pi:
            raise ReceiverError(
                "fov_half_angle_rad is the receiver *semi-angle* psi_c and must lie "
                "in (0, 90] degrees; a value above 90 usually means a full cone "
                "opening was passed where a semi-angle was expected, which would "
                "silently widen the acceptance test as well as this gain",
                context={"degrees": math.degrees(self.fov_half_angle_rad)},
            )
        if self.preamplifier_noise_factor < 1.0:
            raise ReceiverError(
                "the excess-noise factor multiplies the resistor's own thermal noise "
                "and cannot be below 1",
                context={"factor": self.preamplifier_noise_factor},
            )
        if not math.isfinite(self.capacitance_per_area_f_per_m2) or (
            self.capacitance_per_area_f_per_m2 < 0.0
        ):
            raise ReceiverError("capacitance per unit area must be finite and non-negative",
                                context={"value": self.capacitance_per_area_f_per_m2})
        if not math.isfinite(self.height_m) or self.height_m < 0.0:
            raise ReceiverError("mounting height must be finite and non-negative",
                                context={"height_m": self.height_m})

    # -- collection -----------------------------------------------------------

    @property
    def concentrator_gain(self) -> float:
        """``n^2 / sin^2 psi_c``, the ideal non-imaging concentrator.

        The dependence on psi_c is the whole reason the semi-angle convention is
        worth arguing about: this *falls* as the cone widens, so the pessimistic
        60-degree reading gives up gain rather than gaining coverage for free.
        """

        sine = math.sin(self.fov_half_angle_rad)
        return self.concentrator_refractive_index**2 / (sine * sine)

    def effective_area_m2(self, incidence_angle_rad: float) -> float:
        """Collecting area presented to a ray arriving at ``incidence_angle_rad``.

        ``A * T_s * g * cos psi``.  **No acceptance test is applied**: this
        keeps falling smoothly past psi_c rather than dropping to zero there,
        because the drop belongs to the geometry layer and applying it twice
        would remove the same rays twice.  The value returned beyond psi_c is
        therefore not one the physical concentrator can deliver, and callers
        must have established acceptance first.

        A ray at or beyond 90 degrees is refused rather than returned as zero or
        negative.  It arrives at the back of the detector, which is not a
        question about gain at all -- and since psi_c can never exceed 90
        degrees, geometry has already rejected any such link.  Refusing turns a
        caller that skipped the acceptance test into a failure rather than a
        plausible-looking zero.
        """

        if not math.isfinite(incidence_angle_rad) or incidence_angle_rad < 0.0:
            raise ReceiverError(
                "incidence angle must be finite and non-negative; it is measured off "
                "the photodiode boresight and carries no sign",
                context={"incidence_angle_rad": incidence_angle_rad},
            )
        if incidence_angle_rad >= 0.5 * math.pi:
            raise ReceiverError(
                "a ray at or beyond 90 degrees strikes the back of the detector; the "
                "acceptance test in the geometry layer rejects it before any gain is "
                "asked for",
                context={"degrees": math.degrees(incidence_angle_rad)},
            )
        return (
            self.active_area_m2
            * self.filter_transmission
            * self.concentrator_gain
            * math.cos(incidence_angle_rad)
        )

    def photocurrent_a(self, received_optical_power_w: float) -> float:
        """Signal photocurrent from received optical power.

        The only place responsivity is applied.  It is separated from the
        optical budget because the square in IM/DD -- electrical signal power
        goes as the square of *optical* power -- is where an optical decibel
        silently becomes two electrical decibels, and that conversion should
        happen once, visibly, in the modules that need it.
        """

        if not math.isfinite(received_optical_power_w) or received_optical_power_w < 0.0:
            raise ReceiverError("received optical power must be finite and non-negative",
                                context={"w": received_optical_power_w})
        return self.responsivity_a_per_w * received_optical_power_w

    # -- orientation ----------------------------------------------------------

    def boresight_rad(self, vehicle_heading_rad: float) -> float:
        """Direction the photodiode looks, given its vehicle's heading.

        Rearward: the receiver is the *leader* of a following pair and the
        transmitter is behind it.  Stated as a function rather than a comment
        because :mod:`hybrid_v2x_rl.core.pair_geometry` derives its incidence angle
        from the same convention, and two independent expressions of a mounting
        rule are how the two drift apart.
        """

        if not math.isfinite(vehicle_heading_rad):
            raise ReceiverError("heading must be finite",
                                context={"heading_rad": vehicle_heading_rad})
        return wrap_to_pi(vehicle_heading_rad + math.pi)

    # -- front-end feasibility ------------------------------------------------

    @property
    def capacitance_f(self) -> float:
        """Junction capacitance, which scales with active area."""

        return self.capacitance_per_area_f_per_m2 * self.active_area_m2

    def max_load_resistance_ohm(self, electrical_bandwidth_hz: float) -> float:
        """Largest load resistance whose RC pole still passes ``bandwidth``.

        ``1 / (2 pi B C)``.  Infinite for a hypothetical zero-capacitance
        detector, which is the honest answer rather than a division by zero.
        """

        if not math.isfinite(electrical_bandwidth_hz) or electrical_bandwidth_hz <= 0.0:
            raise ReceiverError("electrical bandwidth must be finite and positive",
                                context={"hz": electrical_bandwidth_hz})
        if self.capacitance_f == 0.0:
            return math.inf
        return 1.0 / (2.0 * math.pi * electrical_bandwidth_hz * self.capacitance_f)

    def check_front_end_feasible(self, electrical_bandwidth_hz: float) -> None:
        """Refuse a front end whose RC pole sits below its declared bandwidth.

        Without this the area/bandwidth/noise coupling is a docstring, and a
        larger detector reads as free signal gain.  It is not: the capacitance
        it brings forces the load resistance down and the thermal noise up.
        """

        ceiling = self.max_load_resistance_ohm(electrical_bandwidth_hz)
        if self.load_resistance_ohm > ceiling:
            raise ReceiverError(
                "load resistance exceeds the RC ceiling of the declared front end; "
                "either the detector is too large, the resistance too high, or the "
                "bandwidth is not achievable with this photodiode",
                context={
                    "load_resistance_ohm": self.load_resistance_ohm,
                    "ceiling_ohm": ceiling,
                    "bandwidth_hz": electrical_bandwidth_hz,
                    "capacitance_f": self.capacitance_f,
                },
            )


__all__ = [
    "ACTIVE_AREA_M2",
    "CONCENTRATOR_REFRACTIVE_INDEX",
    "DEFAULT_LOAD_RESISTANCE_OHM",
    "DEFAULT_NOISE_TEMPERATURE_K",
    "DEFAULT_PREAMPLIFIER_NOISE_FACTOR",
    "DETECTOR_CAPACITANCE_PER_AREA_F_PER_M2",
    "FILTER_TRANSMISSION",
    "RESPONSIVITY_A_PER_W",
    "OpticalReceiver",
    "ReceiverError",
]
