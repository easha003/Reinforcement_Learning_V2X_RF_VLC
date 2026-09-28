"""Physical receive-diversity profile and MRC arithmetic."""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.channels.rf.diversity import (
    ReceiveCombiningRule,
    ReceiveDiversityError,
    RFReceiveDiversity,
)


def test_siso_is_an_exact_identity_on_the_primary_branch() -> None:
    profile = RFReceiveDiversity.siso()

    assert profile.is_siso
    assert profile.combined_fading_power(0.375) == 0.375
    assert profile.as_dict() == {
        "antenna_count": 1,
        "combining_rule": "none",
        "branch_correlation": 0.0,
        "implementation_loss_db": 0.0,
        "secondary_branch_power_factor": 1.0,
    }


def test_zero_loss_equal_branch_mrc_has_three_db_combining_gain() -> None:
    profile = RFReceiveDiversity.two_branch_mrc(
        branch_correlation=0.0,
        implementation_loss_db=0.0,
    )

    combined = profile.combined_fading_power(1.0, 1.0)

    assert combined == pytest.approx(2.0)
    assert 10.0 * math.log10(combined) == pytest.approx(3.0102999566)


def test_secondary_branch_loss_applies_before_mrc() -> None:
    profile = RFReceiveDiversity.two_branch_mrc(
        branch_correlation=0.7,
        implementation_loss_db=7.0,
    )

    assert profile.secondary_branch_power_factor == pytest.approx(
        math.pow(10.0, -0.7)
    )
    assert profile.combined_fading_power(0.4, 0.8) == pytest.approx(
        0.4 + math.pow(10.0, -0.7) * 0.8
    )


def test_profile_rejects_unimplemented_or_incomplete_combinations() -> None:
    with pytest.raises(ReceiveDiversityError, match="SISO requires"):
        RFReceiveDiversity(
            antenna_count=2,
            combining_rule=ReceiveCombiningRule.NONE,
            branch_correlation=0.0,
            implementation_loss_db=0.0,
        )
    with pytest.raises(ReceiveDiversityError, match="requires exactly two"):
        RFReceiveDiversity(
            antenna_count=3,
            combining_rule=ReceiveCombiningRule.MAXIMUM_RATIO,
            branch_correlation=0.0,
            implementation_loss_db=0.0,
        )
    with pytest.raises(ReceiveDiversityError, match="secondary branch"):
        RFReceiveDiversity.two_branch_mrc(
            branch_correlation=0.0,
            implementation_loss_db=0.0,
        ).combined_fading_power(1.0)
