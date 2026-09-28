"""Fail-closed physical-consistency audit for the headline RF profile.

This audit does not change the radio model.  It makes the resource-accounting
boundary explicit before a physical reliability intervention is frozen: the
link budget must not consume more resource blocks than one collision-model
subchannel is said to provide.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Final

from hybrid_v2x_rl.channels.rf.bler import required_snr_db
from hybrid_v2x_rl.channels.rf.collision import headline_parameters
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import BITS_PER_RESOURCE_ELEMENT, ProjectConfig

RF_PHYSICAL_PROFILE_AUDIT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.rf-physical-profile-audit.v1"
)

# ETSI TS 138 101-1 V16.24.0, Tables 6.2E.1.1-0, 7.3E.2-1 and A.7.2.1.2-1.
ETSI_NR_V2X_SOURCE: Final = (
    "https://www.etsi.org/deliver/etsi_ts/138100_138199/13810101/"
    "16.24.00_60/ts_13810101v162400p.pdf"
)
NR_V2X_N47_POWER_CLASS_3_DBM: Final = 23.0
NR_V2X_N47_10MHZ_REFERENCE_SENSITIVITY_DBM: Final = -92.1
NR_V2X_N47_10MHZ_REFERENCE_RB: Final = 24
NR_V2X_N47_10MHZ_REFERENCE_TBS_BITS: Final = 1608


def build_rf_physical_profile_audit(config: ProjectConfig) -> dict[str, object]:
    """Return a deterministic audit of the active RF and pool semantics."""

    rf = config.rf
    try:
        calibration_artifact = rf.calibration_artifact.relative_to(
            config.paths.project_root
        ).as_posix()
    except ValueError:
        calibration_artifact = rf.calibration_artifact.as_posix()
    collision = headline_parameters(
        committed_airtime_s=rf.timing.airtime_s
        * config.service.rf_attempts_per_packet
    )
    bits_per_element = BITS_PER_RESOURCE_ELEMENT[rf.modulation]
    information_bits = (
        config.service.payload_bytes + rf.timing.framing_overhead_bytes
    ) * 8
    coded_block_bits = rf.timing.coded_block_bits(config.service.payload_bytes)
    full_carrier_coded_bits = rf.available_coded_bits()
    full_carrier_channel_uses = int(full_carrier_coded_bits / bits_per_element)

    divisible_pool = rf.resource_blocks % collision.subchannels == 0
    rb_per_subchannel = (
        rf.resource_blocks // collision.subchannels if divisible_pool else None
    )
    subchannel_coded_bits = full_carrier_coded_bits / collision.subchannels
    subchannel_channel_uses = int(subchannel_coded_bits / bits_per_element)
    minimum_code_rate_for_one_subchannel = information_bits / subchannel_coded_bits
    allocation_spans_subchannels = (
        rf.resource_blocks / rb_per_subchannel
        if rb_per_subchannel is not None
        else None
    )

    checks = {
        "resource_blocks_match_10mhz_30khz_reference": (
            rf.resource_blocks == NR_V2X_N47_10MHZ_REFERENCE_RB
        ),
        "nominal_tx_power_within_power_class_3": (
            rf.tx_power_dbm <= NR_V2X_N47_POWER_CLASS_3_DBM
        ),
        "pool_divides_into_integer_subchannels": divisible_pool,
        "configured_block_fits_full_carrier": (
            coded_block_bits <= full_carrier_coded_bits
        ),
        "configured_block_fits_one_pool_subchannel": (
            coded_block_bits <= subchannel_coded_bits
        ),
        "link_allocation_equals_one_pool_subchannel": (
            allocation_spans_subchannels == 1.0
        ),
        "declared_calibration_artifact_exists": rf.calibration_artifact.exists(),
    }
    allocation_check_names = (
        "resource_blocks_match_10mhz_30khz_reference",
        "pool_divides_into_integer_subchannels",
        "configured_block_fits_full_carrier",
        "configured_block_fits_one_pool_subchannel",
        "link_allocation_equals_one_pool_subchannel",
    )
    allocation_contract_consistent = all(
        checks[name] for name in allocation_check_names
    )
    freeze_ready = all(checks.values())

    return {
        "schema": RF_PHYSICAL_PROFILE_AUDIT_SCHEMA,
        "scope": (
            "read-only RF link/resource consistency audit; no trace replay, "
            "policy evaluation, training, or test-split access"
        ),
        "config_hash": config_hash(config),
        "test_split_opened": False,
        "training_run_started": False,
        "active_profile": {
            "carrier_hz": rf.carrier_hz,
            "system_bandwidth_hz": rf.bandwidth_hz,
            "subcarrier_spacing_hz": rf.subcarrier_spacing_hz,
            "resource_blocks": rf.resource_blocks,
            "modulation": rf.modulation,
            "configured_code_rate": rf.timing.code_rate,
            "tx_power_dbm": rf.tx_power_dbm,
            "information_bits": information_bits,
            "coded_block_bits": coded_block_bits,
            "full_carrier_available_coded_bits": full_carrier_coded_bits,
            "active_finite_blocklength_channel_uses": full_carrier_channel_uses,
            "active_information_rate_bits_per_channel_use": (
                information_bits / full_carrier_channel_uses
            ),
            "required_snr_db": {
                "bler_1e-1": required_snr_db(
                    1e-1, full_carrier_channel_uses, information_bits
                ),
                "bler_1e-3": required_snr_db(
                    1e-3, full_carrier_channel_uses, information_bits
                ),
                "bler_1e-4": required_snr_db(
                    1e-4, full_carrier_channel_uses, information_bits
                ),
                "bler_1e-5": required_snr_db(
                    1e-5, full_carrier_channel_uses, information_bits
                ),
            },
            "declared_calibration_artifact": calibration_artifact,
        },
        "collision_pool_interpretation": {
            "resource_unit": "one full 10 MHz, 24-RB carrier allocation",
            "legacy_resource_count_field": "subchannels",
            "full_carrier_resources": collision.subchannels,
            "subchannels": collision.subchannels,
            "resource_blocks_per_subchannel": rb_per_subchannel,
            "one_subchannel_available_coded_bits": subchannel_coded_bits,
            "one_subchannel_channel_uses": subchannel_channel_uses,
            "minimum_code_rate_to_fit_current_block_in_one_subchannel": (
                minimum_code_rate_for_one_subchannel
            ),
            "active_link_allocation_spans_subchannels": allocation_spans_subchannels,
        },
        "standard_reference": {
            "source": ETSI_NR_V2X_SOURCE,
            "release": "ETSI TS 138 101-1 V16.24.0 (3GPP TS 38.101-1 Release 16)",
            "band": "n47",
            "power_class": 3,
            "nominal_max_output_power_dbm": NR_V2X_N47_POWER_CLASS_3_DBM,
            "reference_sensitivity_dbm_10mhz_30khz_scs": (
                NR_V2X_N47_10MHZ_REFERENCE_SENSITIVITY_DBM
            ),
            "reference_channel_10mhz_30khz_scs": {
                "resource_blocks": NR_V2X_N47_10MHZ_REFERENCE_RB,
                "modulation": "qpsk",
                "mcs_index": 4,
                "transport_block_bits": NR_V2X_N47_10MHZ_REFERENCE_TBS_BITS,
            },
            "scope_note": (
                "The reference channel establishes a standards anchor; it is "
                "not a BLER calibration for the repository's 16QAM profile."
            ),
        },
        "checks": checks,
        "decision": {
            "allocation_contract_consistent": allocation_contract_consistent,
            "physical_profile_freeze_ready": freeze_ready,
            "training_authorization": False,
            "blocking_findings": [
                name for name, passed in checks.items() if not passed
            ],
            "required_resolution": (
                "Retain the corrected full-10-MHz allocation contract and supply "
                "the missing calibration evidence before freezing a receive-diversity "
                "or other physical reliability intervention."
            ),
        },
    }


def write_rf_physical_profile_audit(
    payload: dict[str, object], output_path: Path
) -> Path:
    """Atomically write an audit payload as stable, human-readable JSON."""

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "ETSI_NR_V2X_SOURCE",
    "RF_PHYSICAL_PROFILE_AUDIT_SCHEMA",
    "build_rf_physical_profile_audit",
    "write_rf_physical_profile_audit",
]
