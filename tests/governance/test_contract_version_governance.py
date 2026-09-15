"""Governance RED/GREEN: contract bump implies a new template generation.

The gap: ALIGN_CONTRACT_VERSION had zero consumers in governance —
a bump changed nothing downstream. These tests pin the trigger wiring:

(a) bumped contract -> required new generation (via check_compatibility
    AND the required_generation_for trigger);
(b) reproducible geometry -> re-embedded to the new generation;
(c) unreproducible geometry -> re_enrollment_required;
(d) cross-generation comparison refused.

(b)-(d) ride the existing migration machinery; (a) is the new wiring.
Synthetic only, no camera, no thresholds touched.
"""

from facecore.contracts.migration import ModelMigrationManifest
from facecore.governance.contract_guard import (
    PREPROCESSING_GENERATION_FAMILY,
    current_preprocessing_generation,
    required_generation_for,
)
from facecore.pipeline.align import ALIGN_CONTRACT_VERSION


def _manifest(preprocessing_generation: str) -> ModelMigrationManifest:
    return ModelMigrationManifest(
        embedder_artifact_hash="0" * 64,
        detector_generation="yunet-2023mar",
        preprocessing_generation=preprocessing_generation,
        tensor_layout="NCHW",
        normalization_contract="scale=1/128;mean=127.5;std=128",
        embedding_dimension=128,
        numerical_precision="fp32",
        quantization_type="none",
        execution_runtime="onnxruntime-cpu-arm64",
    )


def test_current_generation_derives_from_live_contract() -> None:
    assert current_preprocessing_generation() == (
        f"{PREPROCESSING_GENERATION_FAMILY}+align{ALIGN_CONTRACT_VERSION}"
    )


def test_stale_generation_requires_migration_to_current() -> None:
    """(a) Any non-live generation triggers, never silently compat."""
    assert required_generation_for("sface-112-rgb") == (
        current_preprocessing_generation()
    )
    assert required_generation_for("sface-112-rgb+align1") == (
        current_preprocessing_generation()
    )
    assert required_generation_for(current_preprocessing_generation()) is None


def test_bump_surfaces_as_migration_required_not_compat() -> None:
    """(a) The 9-field predicate must say MIGRATION_REQUIRED on bump."""
    stored = _manifest("sface-112-rgb")
    runtime = _manifest(current_preprocessing_generation())
    assert stored.check_compatibility(runtime) == "MIGRATION_REQUIRED"
    assert runtime.check_compatibility(runtime) == "COMPATIBLE"


def test_runtime_manifest_tracks_live_contract() -> None:
    """(a) The CLI runtime manifest must carry the derived generation.

    Pre-fix it hard-codes the stale 'sface-112-rgb' string, so the live
    contract bump never reaches check_compatibility — this is RED until
    the wiring lands.
    """
    from facecore.cli import _runtime_manifest

    assert (
        _runtime_manifest().preprocessing_generation
        == current_preprocessing_generation()
    )
