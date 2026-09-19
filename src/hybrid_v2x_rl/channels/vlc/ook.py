"""OOK bit error and packet failure for the IM/DD optical link.

Implementation spec section 12.5. The uncoded shortcut
``P_packet = 1 - (1 - P_b)^N`` is permitted "only when its bit-error
assumptions are declared", and if FEC is modelled "code rate and decoded-block
error must be explicit". Both are done here, and the declarations are the
substance of the module rather than an afterthought.

**The BER expression and its convention.** For on-off keying with equiprobable
symbols, a matched filter and a threshold midway between the levels,

    BER = Q(sqrt(gamma)),  gamma = (R P_avg)^2 / sigma^2

where ``P_avg`` is the *average* received optical power and ``sigma^2`` is the
total electrical noise variance. Conventions differ across the literature by
factors of two -- some define gamma against peak power, some fold the factor
into the argument of Q -- so it is stated explicitly: **average power, and no
factor inside the square root.**

Two consequences follow from squaring the photocurrent, and both matter:

* a decibel of optical loss costs **two** decibels of electrical SNR, so the
  link degrades as 1/d^4 rather than 1/d^2 and is far more range-sensitive than
  the radio;
* the Q-function is steep, so the link is close to binary in range -- it works,
  and then over a short distance it does not.

**The coding assumption, which is the weakest part.** The profile fixes a rate
of 1/4 but has never chosen a code. Rather than invent one, this models bounded-
distance decoding with a declared correctable fraction: a block fails when more
than ``correctable_fraction`` of its coded bits are wrong. That is optimistic
against a real short-blocklength code and pessimistic against a strong one, and
it is a placeholder rather than a result. It is *not* the finite-blocklength
normal approximation the radio uses -- section 7.3 forbids transferring that to
IM/DD, and a guard test enforces that this module never imports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.core.errors import HybridV2XError

#: Fraction of coded bits a rate-1/4 block is assumed to correct.  DECLARED and
#: unsourced: the profile fixes the rate but not the code.  Reported as a
#: sensitivity rather than a value, because a packet error computed through it
#: inherits its arbitrariness.
DEFAULT_CORRECTABLE_FRACTION = 0.05

#: Floor and ceiling so a caller never receives exactly zero or one and then
#: takes a logarithm of it.
_MIN_PROBABILITY = 1e-15
_MAX_PROBABILITY = 1.0 - 1e-15


class OOKError(HybridV2XError):
    """OOK error rates were requested with an invalid argument."""


def gaussian_tail(x: float) -> float:
    """``Q(x)``, the upper tail of the standard normal."""

    return 0.5 * math.erfc(x / math.sqrt(2.0))


def bit_error_rate(electrical_snr: float) -> float:
    """``Q(sqrt(gamma))`` for OOK; see the module docstring for the convention."""

    if electrical_snr < 0.0 or not math.isfinite(electrical_snr):
        raise OOKError("electrical SNR must be finite and non-negative",
                       context={"electrical_snr": electrical_snr})
    return min(_MAX_PROBABILITY, max(_MIN_PROBABILITY, gaussian_tail(math.sqrt(electrical_snr))))


def coded_bits(payload_bytes: int, framing_bytes: int, code_rate: float) -> int:
    """Coded block length for a payload, from the configured framing and rate."""

    if payload_bytes <= 0 or framing_bytes < 0:
        raise OOKError("payload must be positive and framing non-negative",
                       context={"payload_bytes": payload_bytes,
                                "framing_bytes": framing_bytes})
    if not 0.0 < code_rate <= 1.0:
        raise OOKError("code rate must lie in (0, 1]", context={"code_rate": code_rate})
    return int(round((payload_bytes + framing_bytes) * 8 / code_rate))


def uncoded_packet_error(bit_error: float, block_bits: int) -> float:
    """``1 - (1 - P_b)^N``.

    Spec 12.5 permits this only with its assumptions declared, so: bit errors
    are assumed **independent**. That holds for a memoryless AWGN-like detector
    and fails under burst noise or a fading optical path. This link has no
    modelled fast fading -- the optical channel varies with geometry, which is
    slow -- so independence is defensible here in a way it would not be for the
    radio.
    """

    if block_bits <= 0:
        raise OOKError("block length must be positive", context={"block_bits": block_bits})
    return min(_MAX_PROBABILITY, max(_MIN_PROBABILITY, 1.0 - (1.0 - bit_error) ** block_bits))


def coded_packet_error(
    bit_error: float,
    block_bits: int,
    correctable_fraction: float = DEFAULT_CORRECTABLE_FRACTION,
) -> float:
    """Block failure under bounded-distance decoding, by the normal approximation.

    A block fails when more than ``correctable_fraction * block_bits`` of its
    coded bits are in error. The binomial tail is approximated by a normal,
    which is accurate for the block lengths in force here -- 10,624 coded bits
    at rate 1/4 -- and the approximation is stated because at very low bit-error
    rates the true tail is lighter than the normal suggests.
    """

    if block_bits <= 0:
        raise OOKError("block length must be positive", context={"block_bits": block_bits})
    if not 0.0 <= correctable_fraction < 1.0:
        raise OOKError("correctable fraction must lie in [0, 1)",
                       context={"correctable_fraction": correctable_fraction})

    threshold = correctable_fraction * block_bits
    mean = bit_error * block_bits
    variance = block_bits * bit_error * (1.0 - bit_error)
    if variance <= 0.0:
        return _MIN_PROBABILITY if mean <= threshold else _MAX_PROBABILITY
    return min(
        _MAX_PROBABILITY,
        max(_MIN_PROBABILITY, gaussian_tail((threshold - mean) / math.sqrt(variance))),
    )


@dataclass(frozen=True, slots=True)
class OOKOutcome:
    """Bit and packet error for one realized optical SNR."""

    electrical_snr: float
    bit_error_rate: float
    packet_error_rate: float
    block_bits: int
    correctable_fraction: float

    @property
    def snr_db(self) -> float:
        return 10.0 * math.log10(self.electrical_snr) if self.electrical_snr > 0.0 else -math.inf


def evaluate(
    electrical_snr: float,
    *,
    payload_bytes: int,
    framing_bytes: int,
    code_rate: float,
    correctable_fraction: float = DEFAULT_CORRECTABLE_FRACTION,
) -> OOKOutcome:
    """Assemble bit and packet error for one SNR under the configured coding."""

    block = coded_bits(payload_bytes, framing_bytes, code_rate)
    ber = bit_error_rate(electrical_snr)
    return OOKOutcome(
        electrical_snr=electrical_snr,
        bit_error_rate=ber,
        packet_error_rate=coded_packet_error(ber, block, correctable_fraction),
        block_bits=block,
        correctable_fraction=correctable_fraction,
    )


__all__ = [
    "DEFAULT_CORRECTABLE_FRACTION",
    "OOKError",
    "OOKOutcome",
    "bit_error_rate",
    "coded_bits",
    "coded_packet_error",
    "evaluate",
    "gaussian_tail",
    "uncoded_packet_error",
]
