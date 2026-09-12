"""Task 1 RED/GREEN: Phase-1B governance & lifecycle contracts.

Source of truth: 1B plan §11 Task 1 + S1B manifest §3 KeyProvider contract.
RED: ``ModuleNotFoundError: No module named 'facecore.contracts.candidate'``.
GREEN: all tests pass; ``mypy src tests`` adds zero new errors; ``ruff`` clean.
"""

import dataclasses
import json

import pytest

from facecore.contracts.candidate import (
    CandidateStatus,
    CandidateTemplate,
    EvidenceRecord,
)
from facecore.contracts.confirmation import (
    ActorType,
    ConfirmationRequest,
    ConfirmationVerdict,
)
from facecore.contracts.crypto import (
    EncryptedBlob,
    KeyNotFoundError,
    KeyProviderProtocol,
    KeyReference,
    StoreCorruptionError,
    WrappedKey,
)
from facecore.contracts.drift import DriftMetrics, DriftPolicy, DriftStatus
from facecore.contracts.export import (
    ExportContainer,
    ExportedIdentity,
    ExportMetadata,
    TombstoneRecord,
)
from facecore.contracts.migration import (
    GenerationStatus,
    MigrationResult,
    ModelIncompatibilityError,
    ModelMigrationManifest,
)
from facecore.contracts.policy import GovernancePolicy
from facecore.contracts.result import (
    DRIFT_BOUNDARY_EXCEEDED,
    IDENTITY_RE_ENROLLMENT_REQUIRED,
    Decision,
    IdentificationResult,
    Quality,
    ResultStatus,
)
from facecore.contracts.template import FaceTemplate, TemplateRevision


def _blob() -> EncryptedBlob:
    return EncryptedBlob(
        format_version=1,
        cipher_id=1,
        nonce=b"n" * 12,
        ciphertext=b"c" * 32,
    )


def _revision() -> TemplateRevision:
    return TemplateRevision(revision=1, template_id="t-1", supersedes=None)


def _manifest(
    field: str | None = None, mutant: str | int | None = None
) -> ModelMigrationManifest:
    base = ModelMigrationManifest(
        embedder_artifact_hash="aa" * 32,
        detector_generation="yunet-2023mar",
        preprocessing_generation="sface-align-v1",
        tensor_layout="NCHW",
        normalization_contract="scale=1.0/255,mean=[0,0,0],std=[1,1,1]",
        embedding_dimension=128,
        numerical_precision="fp32",
        quantization_type="none",
        execution_runtime="onnxruntime-cpu-arm64==1.30.0",
    )
    if field is None:
        return base
    if field == "embedding_dimension":
        assert isinstance(mutant, int)
        return dataclasses.replace(base, embedding_dimension=mutant)
    assert isinstance(mutant, str)
    str_fields: dict[str, str] = {
        "embedder_artifact_hash": base.embedder_artifact_hash,
        "detector_generation": base.detector_generation,
        "preprocessing_generation": base.preprocessing_generation,
        "tensor_layout": base.tensor_layout,
        "normalization_contract": base.normalization_contract,
        "numerical_precision": base.numerical_precision,
        "quantization_type": base.quantization_type,
        "execution_runtime": base.execution_runtime,
    }
    str_fields[field] = mutant
    return ModelMigrationManifest(
        embedder_artifact_hash=str_fields["embedder_artifact_hash"],
        detector_generation=str_fields["detector_generation"],
        preprocessing_generation=str_fields["preprocessing_generation"],
        tensor_layout=str_fields["tensor_layout"],
        normalization_contract=str_fields["normalization_contract"],
        embedding_dimension=base.embedding_dimension,
        numerical_precision=str_fields["numerical_precision"],
        quantization_type=str_fields["quantization_type"],
        execution_runtime=str_fields["execution_runtime"],
    )


def _candidate(
    status: CandidateStatus = CandidateStatus.PENDING,
) -> CandidateTemplate:
    return CandidateTemplate(
        template_id="c-1",
        identity_id="person-001",
        generation_id="G1",
        status=status,
        key_id="key-1",
        encrypted_embedding=_blob(),
        encrypted_exemplar=_blob(),
        exemplar_crop_box=(10.0, 20.0, 100.0, 100.0),
        exemplar_landmarks=((1.0, 2.0), (3.0, 4.0)),
        quality_score=0.9,
        evidence_log=(
            EvidenceRecord(
                event_type="seed",
                timestamp="2026-09-12T00:00:00+08:00",
                sequence_number=1,
                score=0.9,
            ),
        ),
        expires_at="2026-09-19T00:00:00+08:00",
        created_at="2026-09-12T00:00:00+08:00",
    )


def _governed_template() -> FaceTemplate:
    return FaceTemplate(
        template_id="t-1",
        identity_id="person-001",
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
        revision=_revision(),
        generation_id="G1",
        encrypted_embedding=_blob(),
        encrypted_exemplar=_blob(),
        exemplar_crop_box=(10.0, 20.0, 100.0, 100.0),
        exemplar_landmarks=((1.0, 2.0),),
        key_id="key-1",
    )


def _container() -> ExportContainer:
    return ExportContainer(
        metadata=ExportMetadata(
            kdf_salt_hex="ab" * 16,
            created_at="2026-09-12T00:00:00+08:00",
        ),
        policy=GovernancePolicy.provisional_v1(),
        model_manifest=_manifest(),
        identities=(
            ExportedIdentity(
                identity_id="person-001",
                display_name="Test Person",
                status="active",
                current_revision=1,
            ),
        ),
        active_templates=(_governed_template(),),
        retired_templates=(),
        candidates=tuple(_candidate(s) for s in CandidateStatus),
        revisions=(_revision(),),
        tombstones=(
            TombstoneRecord(
                target_type="identity",
                target_id="person-009",
                key_ids=("k1", "k2"),
                status="key_destroyed",
            ),
        ),
        envelope=_blob(),
    )


class _StubKeyProvider:
    """Minimal in-memory KeyProviderProtocol implementation for tests."""

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}

    def create_key(self, identity_id: str) -> str:
        key_id = f"key-{identity_id}-1"
        self._keys[key_id] = b"k" * 32
        return key_id

    def get_key(self, key_id: str) -> bytes:
        try:
            return self._keys[key_id]
        except KeyError:
            raise KeyNotFoundError(key_id) from None

    def destroy_key(self, key_id: str) -> None:
        self._keys.pop(key_id, None)

    def destroy_identity_keys(self, identity_id: str) -> None:
        prefix = f"key-{identity_id}-"
        for key_id in [k for k in self._keys if k.startswith(prefix)]:
            del self._keys[key_id]

    def wrap_key(self, key_id: str, wrapping_key: bytes) -> WrappedKey:
        return WrappedKey(
            wrapped_dek=self.get_key(key_id),
            nonce=b"w" * 12,
            tag=b"t" * 16,
            key_id=key_id,
        )

    def unwrap_and_store_key(
        self, wrapped_key: WrappedKey, unwrapping_key: bytes
    ) -> str:
        key_id = f"rehomed-{wrapped_key.key_id}"
        self._keys[key_id] = wrapped_key.wrapped_dek
        return key_id


def test_candidate_defaults_to_zero_corroboration_and_pending() -> None:
    candidate = CandidateTemplate(
        template_id="c-1",
        identity_id="person-001",
        generation_id="G1",
        key_id="key-1",
        encrypted_embedding=_blob(),
    )
    assert candidate.additional_corroboration_count == 0
    assert candidate.status is CandidateStatus.PENDING


def test_candidate_rejects_negative_corroboration() -> None:
    with pytest.raises(ValueError):
        CandidateTemplate(
            template_id="c-1",
            identity_id="person-001",
            generation_id="G1",
            key_id="key-1",
            encrypted_embedding=_blob(),
            additional_corroboration_count=-1,
        )


def test_candidate_status_covers_all_five_lifecycle_states() -> None:
    assert {s.value for s in CandidateStatus} == {
        "pending",
        "promoted",
        "rejected",
        "expired",
        "generation_retired",
    }


def test_face_template_1a_construction_still_valid() -> None:
    """Phase-1A frozen compatibility: exemplar fields default to absent."""
    tpl = FaceTemplate(
        template_id="t-1",
        identity_id="person-001",
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
        revision=_revision(),
    )
    assert tpl.encrypted_exemplar is None
    assert tpl.key_id is None


def test_face_template_governance_ready_requires_exemplar_fields() -> None:
    tpl = FaceTemplate(
        template_id="t-1",
        identity_id="person-001",
        model_version="sface-2021dec-fp32",
        embedding_dim=128,
        revision=_revision(),
    )
    with pytest.raises(ValueError, match="governance"):
        tpl.assert_governance_ready()


def test_face_template_governance_ready_passes_with_full_fields() -> None:
    _governed_template().assert_governance_ready()


def test_cross_generation_comparison_refused() -> None:
    g1 = dataclasses.replace(_governed_template(), generation_id="G1")
    g2 = dataclasses.replace(_governed_template(), generation_id="G2")
    with pytest.raises(ValueError, match="generation"):
        g1.assert_comparable(g2)


def test_governance_policy_provisional_values_match_plan_section_6() -> None:
    policy = GovernancePolicy.provisional_v1()
    assert policy.governance_policy_version == 1
    assert policy.provisional is True
    assert policy.candidate_update_threshold == 0.88
    assert policy.additional_corroboration_min_events == 1
    assert policy.burst_suppression_min_interval_secs == 60.0
    assert policy.promotion_margin == 0.12
    assert policy.template_bank_capacity == 5
    assert policy.utility_weight_quality == 0.30
    assert policy.utility_weight_support == 0.30
    assert policy.utility_weight_recency == 0.20
    assert policy.utility_weight_coverage == 0.20
    assert policy.utility_penalty_redundancy == 0.15
    assert policy.utility_penalty_outlier == 0.15
    assert policy.drift_max_centroid_shift == 0.20
    assert policy.drift_max_initial_distance == 0.25
    assert policy.rollback_max_depth == 5
    assert policy.retired_retention_days == 90
    assert policy.revision_history_max_count == 20
    assert policy.match_events_max_count == 10000
    assert policy.backup_max_count == 5
    assert policy.exemplar_margin == 0.0


def test_export_container_covers_all_candidate_statuses() -> None:
    container = _container()
    assert container.candidate_statuses() == {
        "pending",
        "promoted",
        "rejected",
        "expired",
        "generation_retired",
    }
    payload = json.loads(container.to_json())
    assert set(payload.keys()) == {
        "metadata",
        "policy",
        "model_manifest",
        "identities",
        "active_templates",
        "retired_templates",
        "candidates",
        "revisions",
        "tombstones",
        "envelope",
    }
    assert payload["policy"]["governance_policy_version"] == 1
    assert (
        payload["model_manifest"]["execution_runtime"]
        == "onnxruntime-cpu-arm64==1.30.0"
    )
    assert len(payload["candidates"]) == 5
    assert payload["policy"]["backup_max_count"] == 5
    active = payload["active_templates"][0]
    assert active["generation_id"] == "G1"
    assert active["encrypted_embedding"]["ciphertext_hex"]
    assert active["encrypted_exemplar"]["ciphertext_hex"]
    assert active["exemplar_crop_box"] == [10.0, 20.0, 100.0, 100.0]
    candidate = payload["candidates"][0]
    assert candidate["key_id"] == "key-1"
    assert candidate["encrypted_embedding"]["ciphertext_hex"]
    assert candidate["encrypted_exemplar"]["ciphertext_hex"]
    assert candidate["exemplar_crop_box"] == [10.0, 20.0, 100.0, 100.0]
    assert candidate["exemplar_landmarks"] == [[1.0, 2.0], [3.0, 4.0]]
    assert candidate["quality_score"] == 0.9
    assert candidate["expires_at"] == "2026-09-19T00:00:00+08:00"
    assert candidate["created_at"] == "2026-09-12T00:00:00+08:00"
    restored = ExportContainer.from_json(container.to_json())
    assert json.loads(restored.to_json()) == payload
    assert payload["envelope"]["cipher_id"] == 1


def test_key_provider_protocol_structural() -> None:
    stub = _StubKeyProvider()
    assert isinstance(stub, KeyProviderProtocol)
    key_id = stub.create_key("person-001")
    assert stub.get_key(key_id) == b"k" * 32
    stub.destroy_key(key_id)
    stub.destroy_key(key_id)
    with pytest.raises(KeyNotFoundError):
        stub.get_key(key_id)


def test_encrypted_blob_rejects_bad_nonce_version_cipher() -> None:
    with pytest.raises(ValueError):
        EncryptedBlob(
            format_version=1, cipher_id=1, nonce=b"short", ciphertext=b"c" * 32
        )
    with pytest.raises(ValueError):
        EncryptedBlob(
            format_version=2, cipher_id=1, nonce=b"n" * 12, ciphertext=b"c" * 32
        )
    with pytest.raises(ValueError):
        EncryptedBlob(
            format_version=1, cipher_id=9, nonce=b"n" * 12, ciphertext=b"c" * 32
        )


def test_key_reference_rejects_empty() -> None:
    with pytest.raises(ValueError):
        KeyReference(key_id="")
    assert KeyReference(key_id="key-1").key_id == "key-1"


def test_storage_errors_are_structured_exit_4_errors() -> None:
    assert issubclass(KeyNotFoundError, Exception)
    assert issubclass(StoreCorruptionError, Exception)
    assert KeyNotFoundError.exit_code == 4
    assert StoreCorruptionError.exit_code == 4


def test_manifest_identical_returns_compatible() -> None:
    assert _manifest().check_compatibility(_manifest()) == "COMPATIBLE"


def test_manifest_generation_difference_returns_migration_required() -> None:
    assert (
        _manifest().check_compatibility(
            _manifest("detector_generation", "yunet-2026may")
        )
        == "MIGRATION_REQUIRED"
    )
    assert (
        _manifest().check_compatibility(
            _manifest("preprocessing_generation", "sface-align-v2")
        )
        == "MIGRATION_REQUIRED"
    )


@pytest.mark.parametrize(
    ("field", "mutant"),
    [
        ("embedder_artifact_hash", "00" * 32),
        ("tensor_layout", "NHWC"),
        ("normalization_contract", "scale=other"),
        ("embedding_dimension", 512),
        ("numerical_precision", "int8"),
        ("quantization_type", "int8bq"),
        ("execution_runtime", "other-runtime==9.9.9"),
    ],
)
def test_manifest_hard_mismatch_raises(
    field: str, mutant: str | int
) -> None:
    with pytest.raises(ModelIncompatibilityError):
        _manifest().check_compatibility(_manifest(field, mutant))


def test_migration_result_defaults_and_generation_status() -> None:
    assert {s.value for s in GenerationStatus} == {
        "current",
        "superseded",
        "re_enrollment_required",
    }
    result = MigrationResult(from_generation="G1", to_generation="G2")
    assert result.active_migrated == 0
    assert result.status is GenerationStatus.CURRENT
    with pytest.raises(ValueError):
        MigrationResult(
            from_generation="G1", to_generation="G2", active_migrated=-1
        )


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (ConfirmationVerdict.CORRECT, True),
        (ConfirmationVerdict.NOT_ME, False),
        (ConfirmationVerdict.CANCELLED, False),
        (ConfirmationVerdict.TIMEOUT, False),
        (ConfirmationVerdict.EOF, False),
    ],
)
def test_confirmation_affirmative_only_on_correct(
    verdict: ConfirmationVerdict, expected: bool
) -> None:
    request = ConfirmationRequest(
        request_id="r-1",
        verdict=verdict,
        actor=ActorType.USER,
        created_at="2026-09-12T00:00:00+08:00",
    )
    assert request.is_affirmative is expected


def test_actor_taxonomy_rejects_unknown() -> None:
    assert {a.value for a in ActorType} == {"user", "operator"}
    with pytest.raises(ValueError):
        ConfirmationRequest(
            request_id="r-9",
            verdict=ConfirmationVerdict.CORRECT,
            actor=ActorType("intruder"),
            created_at="2026-09-12T00:00:00+08:00",
        )


def test_drift_policy_assess_boundary() -> None:
    policy = DriftPolicy()
    assert policy.provisional is True
    over = DriftMetrics(
        identity_id="person-001",
        centroid_shift=0.25,
        reference_kind="anchor",
        observed_at="2026-09-12T00:00:00+08:00",
    )
    assert policy.assess(over) is DriftStatus.BOUNDARY_EXCEEDED
    within = dataclasses.replace(over, centroid_shift=0.10)
    assert policy.assess(within) is DriftStatus.WITHIN_BOUNDS
    with pytest.raises(ValueError):
        DriftMetrics(
            identity_id="person-001",
            centroid_shift=-0.01,
            reference_kind="anchor",
            observed_at="2026-09-12T00:00:00+08:00",
        )


def test_result_serialization_frozen_unchanged() -> None:
    """Regression guard: Phase-1A spec §11 key set/order untouched."""
    result = IdentificationResult(
        status=ResultStatus.MATCHED,
        identity={"id": "person-001", "display_name": "Test", "metadata": {}},
        decision=Decision(
            score=0.82,
            runner_up_score=None,
            threshold=0.76,
            margin=None,
            reason_codes=[],
        ),
        quality=Quality(status="accepted", reason_codes=[]),
        model_version="sface-2021dec-fp32",
        template_revision=1,
        candidate_created=False,
    )
    payload = json.loads(result.to_json())
    assert list(payload.keys()) == [
        "schema_version",
        "status",
        "identity",
        "decision",
        "quality",
        "model_version",
        "template_revision",
        "candidate_created",
    ]


def test_governance_reason_codes_present() -> None:
    assert DRIFT_BOUNDARY_EXCEEDED == "drift_boundary_exceeded"
    assert IDENTITY_RE_ENROLLMENT_REQUIRED == "identity_re_enrollment_required"
