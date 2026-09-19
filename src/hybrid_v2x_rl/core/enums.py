"""Versioned enum values used in configuration and persisted artifacts."""

from enum import IntEnum, StrEnum, unique


@unique
class Action(IntEnum):
    """Committed transmission action for one packet."""

    RF = 0
    VLC = 1
    DUP = 2


@unique
class Link(StrEnum):
    """Physical communication leg."""

    RF = "rf"
    VLC = "vlc"


@unique
class RFPropagationState(StrEnum):
    """TR 37.885-style RF visibility state."""

    LOS = "los"
    NLOSV = "nlosv"
    NLOS = "nlos"


@unique
class FailureCause(StrEnum):
    """Canonical terminal cause attached to a link or packet failure."""

    NONE = "none"
    RF_COLLISION = "rf_collision"
    RF_CHANNEL = "rf_channel"
    VLC_OCCLUSION = "vlc_occlusion"
    VLC_ALIGNMENT = "vlc_alignment"
    VLC_CHANNEL = "vlc_channel"
    PHY_INFEASIBLE = "phy_infeasible"
    DECODING_LATE = "decoding_late"
    JOINT_FAILURE = "joint_failure"


@unique
class DatasetSplit(StrEnum):
    """Trajectory-level dataset partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


__all__ = [
    "Action",
    "DatasetSplit",
    "FailureCause",
    "Link",
    "RFPropagationState",
]
