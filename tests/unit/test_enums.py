"""Regression tests for enum values persisted in Hybrid RF/VLC RL artifacts."""

from hybrid_v2x_rl.core.enums import (
    Action,
    DatasetSplit,
    FailureCause,
    Link,
    RFPropagationState,
)


def test_action_integer_values_are_stable() -> None:
    assert [(member.name, member.value) for member in Action] == [
        ("RF", 0),
        ("VLC", 1),
        ("DUP", 2),
    ]
    assert Action(0) is Action.RF
    assert int(Action.DUP) == 2


def test_link_and_rf_state_values_are_stable() -> None:
    assert [member.value for member in Link] == ["rf", "vlc"]
    assert [member.value for member in RFPropagationState] == [
        "los",
        "nlosv",
        "nlos",
    ]
    assert Link("vlc") is Link.VLC


def test_failure_cause_values_are_stable() -> None:
    assert [member.value for member in FailureCause] == [
        "none",
        "rf_collision",
        "rf_channel",
        "vlc_occlusion",
        "vlc_alignment",
        "vlc_channel",
        "phy_infeasible",
        "decoding_late",
        "joint_failure",
    ]


def test_dataset_split_values_are_stable() -> None:
    assert [member.value for member in DatasetSplit] == [
        "train",
        "validation",
        "test",
    ]
    assert str(DatasetSplit.TEST) == "test"
