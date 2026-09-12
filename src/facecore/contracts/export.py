"""Phase-1B export contracts (plan §8: full-state container + outer envelope).

`ExportContainer` serializes complete governance state: policy profile, full
canonical 9-field model manifest, identities, revisions, active templates,
retired templates, candidates in all five statuses with evidence logs and
exemplars, audit tombstones, and the outer AEAD envelope keyed by
``K_envelope``. Actual encryption and Argon2id derivation belong to Task 6
(`storage/export.py`); this module defines the shape only.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from facecore.contracts.candidate import (
    CandidateStatus,
    CandidateTemplate,
    EvidenceRecord,
)
from facecore.contracts.crypto import EncryptedBlob
from facecore.contracts.migration import ModelMigrationManifest
from facecore.contracts.policy import GovernancePolicy
from facecore.contracts.template import FaceTemplate, TemplateRevision


@dataclass(frozen=True)
class ExportMetadata:
    kdf_salt_hex: str
    created_at: str
    export_format_version: int = 1
    kdf_algorithm: str = "argon2id"
    kdf_version: int = 1
    kdf_memory_cost_kib: int = 65536
    kdf_time_cost: int = 3
    kdf_parallelism: int = 1


@dataclass(frozen=True)
class ExportedIdentity:
    identity_id: str
    display_name: str
    status: str
    current_revision: int


@dataclass(frozen=True)
class TombstoneRecord:
    target_type: str
    target_id: str
    key_ids: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class ExportContainer:
    metadata: ExportMetadata
    policy: GovernancePolicy
    model_manifest: ModelMigrationManifest
    identities: tuple[ExportedIdentity, ...] = field(default_factory=tuple)
    active_templates: tuple[FaceTemplate, ...] = field(default_factory=tuple)
    retired_templates: tuple[FaceTemplate, ...] = field(default_factory=tuple)
    candidates: tuple[CandidateTemplate, ...] = field(default_factory=tuple)
    revisions: tuple[TemplateRevision, ...] = field(default_factory=tuple)
    tombstones: tuple[TombstoneRecord, ...] = field(default_factory=tuple)
    envelope: EncryptedBlob | None = None

    def candidate_statuses(self) -> set[str]:
        return {c.status.value for c in self.candidates}

    @staticmethod
    def _blob_payload(blob: EncryptedBlob | None) -> dict[str, object] | None:
        return blob.to_dict() if blob is not None else None

    @staticmethod
    def _blob_from_payload(payload: object) -> EncryptedBlob | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("encrypted blob must be a JSON object or null")
        return EncryptedBlob(
            format_version=int(payload["format_version"]),
            cipher_id=int(payload["cipher_id"]),
            nonce=bytes.fromhex(str(payload["nonce_hex"])),
            ciphertext=bytes.fromhex(str(payload["ciphertext_hex"])),
        )

    @staticmethod
    def _template_payload(template: FaceTemplate) -> dict[str, object]:
        return {
            "template_id": template.template_id,
            "identity_id": template.identity_id,
            "model_version": template.model_version,
            "embedding_dim": template.embedding_dim,
            "generation_id": template.generation_id,
            "revision": {
                "revision": template.revision.revision,
                "template_id": template.revision.template_id,
                "supersedes": template.revision.supersedes,
            },
            "key_id": template.key_id,
            "encrypted_embedding": ExportContainer._blob_payload(
                template.encrypted_embedding
            ),
            "encrypted_exemplar": ExportContainer._blob_payload(
                template.encrypted_exemplar
            ),
            "exemplar_crop_box": (
                list(template.exemplar_crop_box)
                if template.exemplar_crop_box is not None
                else None
            ),
            "exemplar_landmarks": (
                [list(point) for point in template.exemplar_landmarks]
                if template.exemplar_landmarks is not None
                else None
            ),
            "quality_score": template.quality_score,
            "utility_score": template.utility_score,
            "exemplar_margin": template.exemplar_margin,
        }

    @staticmethod
    def _candidate_payload(candidate: CandidateTemplate) -> dict[str, object]:
        return {
            "template_id": candidate.template_id,
            "identity_id": candidate.identity_id,
            "generation_id": candidate.generation_id,
            "status": candidate.status.value,
            "key_id": candidate.key_id,
            "encrypted_embedding": ExportContainer._blob_payload(
                candidate.encrypted_embedding
            ),
            "encrypted_exemplar": ExportContainer._blob_payload(
                candidate.encrypted_exemplar
            ),
            "exemplar_crop_box": (
                list(candidate.exemplar_crop_box)
                if candidate.exemplar_crop_box is not None
                else None
            ),
            "exemplar_landmarks": (
                [list(point) for point in candidate.exemplar_landmarks]
                if candidate.exemplar_landmarks is not None
                else None
            ),
            "quality_score": candidate.quality_score,
            "additional_corroboration_count": (
                candidate.additional_corroboration_count
            ),
            "evidence_log": [asdict(event) for event in candidate.evidence_log],
            "expires_at": candidate.expires_at,
            "created_at": candidate.created_at,
        }

    def to_dict(self) -> dict[str, object]:
        """Return every field needed for complete governance-state export."""
        return {
            "metadata": asdict(self.metadata),
            "policy": asdict(self.policy),
            "model_manifest": self.model_manifest.to_dict(),
            "identities": [asdict(identity) for identity in self.identities],
            "active_templates": [
                self._template_payload(template) for template in self.active_templates
            ],
            "retired_templates": [
                self._template_payload(template)
                for template in self.retired_templates
            ],
            "candidates": [
                self._candidate_payload(candidate) for candidate in self.candidates
            ],
            "revisions": [asdict(revision) for revision in self.revisions],
            "tombstones": [
                {
                    "target_type": tombstone.target_type,
                    "target_id": tombstone.target_id,
                    "key_ids": list(tombstone.key_ids),
                    "status": tombstone.status,
                }
                for tombstone in self.tombstones
            ],
            "envelope": self._blob_payload(self.envelope),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def _crop_from_payload(
        value: object,
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("exemplar_crop_box must contain four values")
        return (
            float(str(value[0])),
            float(str(value[1])),
            float(str(value[2])),
            float(str(value[3])),
        )

    @staticmethod
    def _landmarks_from_payload(
        value: object,
    ) -> tuple[tuple[float, float], ...] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("exemplar_landmarks must be a list")
        result: list[tuple[float, float]] = []
        for point in value:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("each exemplar landmark must contain two values")
            result.append((float(str(point[0])), float(str(point[1]))))
        return tuple(result)

    @staticmethod
    def _template_from_payload(data: dict[str, Any]) -> FaceTemplate:
        revision_data = cast(dict[str, Any], data["revision"])
        crop = ExportContainer._crop_from_payload(data.get("exemplar_crop_box"))
        landmarks = ExportContainer._landmarks_from_payload(
            data.get("exemplar_landmarks")
        )
        return FaceTemplate(
            template_id=str(data["template_id"]),
            identity_id=str(data["identity_id"]),
            model_version=str(data["model_version"]),
            embedding_dim=int(data["embedding_dim"]),
            generation_id=str(data.get("generation_id", "G1")),
            revision=TemplateRevision(
                revision=int(revision_data["revision"]),
                template_id=str(revision_data["template_id"]),
                supersedes=(
                    str(revision_data["supersedes"])
                    if revision_data.get("supersedes") is not None
                    else None
                ),
            ),
            key_id=str(data["key_id"]) if data.get("key_id") is not None else None,
            encrypted_embedding=ExportContainer._blob_from_payload(
                data.get("encrypted_embedding")
            ),
            encrypted_exemplar=ExportContainer._blob_from_payload(
                data.get("encrypted_exemplar")
            ),
            exemplar_crop_box=crop,
            exemplar_landmarks=landmarks,
            quality_score=float(data.get("quality_score", 0.0)),
            utility_score=float(data.get("utility_score", 0.0)),
            exemplar_margin=float(data.get("exemplar_margin", 0.0)),
        )

    @staticmethod
    def _candidate_from_payload(data: dict[str, Any]) -> CandidateTemplate:
        crop = ExportContainer._crop_from_payload(data.get("exemplar_crop_box"))
        landmarks = ExportContainer._landmarks_from_payload(
            data.get("exemplar_landmarks")
        )
        evidence = tuple(
            EvidenceRecord(
                event_type=str(event["event_type"]),
                timestamp=str(event["timestamp"]),
                sequence_number=int(event["sequence_number"]),
                score=(
                    float(event["score"])
                    if event.get("score") is not None
                    else None
                ),
            )
            for event in cast(list[dict[str, Any]], data.get("evidence_log", []))
        )
        encrypted_embedding = ExportContainer._blob_from_payload(
            data.get("encrypted_embedding")
        )
        if encrypted_embedding is None:
            raise ValueError("candidate encrypted_embedding is required")
        return CandidateTemplate(
            template_id=str(data["template_id"]),
            identity_id=str(data["identity_id"]),
            generation_id=str(data["generation_id"]),
            status=CandidateStatus(str(data["status"])),
            key_id=str(data["key_id"]),
            encrypted_embedding=encrypted_embedding,
            encrypted_exemplar=ExportContainer._blob_from_payload(
                data.get("encrypted_exemplar")
            ),
            exemplar_crop_box=crop,
            exemplar_landmarks=landmarks,
            quality_score=float(data.get("quality_score", 0.0)),
            additional_corroboration_count=int(
                data.get("additional_corroboration_count", 0)
            ),
            evidence_log=evidence,
            expires_at=str(data.get("expires_at", "")),
            created_at=str(data.get("created_at", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "ExportContainer":
        """Decode a complete container without dropping any governance fields."""
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("export container must be a JSON object")
        metadata = ExportMetadata(**cast(dict[str, Any], raw["metadata"]))
        policy = GovernancePolicy(**cast(dict[str, Any], raw["policy"]))
        model_manifest = ModelMigrationManifest(
            **cast(dict[str, Any], raw["model_manifest"])
        )
        identities = tuple(
            ExportedIdentity(**cast(dict[str, Any], item))
            for item in cast(list[object], raw["identities"])
        )
        active = tuple(
            cls._template_from_payload(cast(dict[str, Any], item))
            for item in cast(list[object], raw["active_templates"])
        )
        retired = tuple(
            cls._template_from_payload(cast(dict[str, Any], item))
            for item in cast(list[object], raw["retired_templates"])
        )
        candidates = tuple(
            cls._candidate_from_payload(cast(dict[str, Any], item))
            for item in cast(list[object], raw["candidates"])
        )
        revisions = tuple(
            TemplateRevision(**cast(dict[str, Any], item))
            for item in cast(list[object], raw["revisions"])
        )
        tombstones = tuple(
            TombstoneRecord(
                target_type=str(data["target_type"]),
                target_id=str(data["target_id"]),
                key_ids=tuple(str(key_id) for key_id in data["key_ids"]),
                status=str(data["status"]),
            )
            for data in (
                cast(dict[str, Any], item)
                for item in cast(list[object], raw["tombstones"])
            )
        )
        return cls(
            metadata=metadata,
            policy=policy,
            model_manifest=model_manifest,
            identities=identities,
            active_templates=active,
            retired_templates=retired,
            candidates=candidates,
            revisions=revisions,
            tombstones=tombstones,
            envelope=cls._blob_from_payload(raw.get("envelope")),
        )
