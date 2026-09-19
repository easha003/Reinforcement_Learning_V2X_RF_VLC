"""Stable, dependency-light contracts shared by Hybrid RF/VLC RL modules."""

from hybrid_v2x_rl.core.enums import (
    Action,
    DatasetSplit,
    FailureCause,
    Link,
    RFPropagationState,
)
from hybrid_v2x_rl.core.errors import (
    ArtifactVersionError,
    CalibrationError,
    ConfigurationError,
    HybridV2XError,
    ObservationLeakageError,
    PhysicalInfeasibilityError,
    StatisticalPowerError,
    TraceIntegrityError,
)
from hybrid_v2x_rl.core.randomness import (
    RANDOM_STREAM_NAMES,
    RandomStreams,
    derive_child_seed,
    derive_seed,
    derive_stream_seeds,
    make_generator,
)
from hybrid_v2x_rl.core.units import (
    angular_difference_rad,
    db_to_linear,
    dbm_to_w,
    linear_to_db,
    validate_probability,
    w_to_dbm,
    wrap_angle,
    wrap_angle_positive_rad,
    wrap_angle_rad,
)

__all__ = [
    "RANDOM_STREAM_NAMES",
    "Action",
    "ArtifactVersionError",
    "CalibrationError",
    "ConfigurationError",
    "DatasetSplit",
    "FailureCause",
    "HybridV2XError",
    "Link",
    "ObservationLeakageError",
    "PhysicalInfeasibilityError",
    "RFPropagationState",
    "RandomStreams",
    "StatisticalPowerError",
    "TraceIntegrityError",
    "angular_difference_rad",
    "db_to_linear",
    "dbm_to_w",
    "derive_child_seed",
    "derive_seed",
    "derive_stream_seeds",
    "linear_to_db",
    "make_generator",
    "validate_probability",
    "w_to_dbm",
    "wrap_angle",
    "wrap_angle_positive_rad",
    "wrap_angle_rad",
]
