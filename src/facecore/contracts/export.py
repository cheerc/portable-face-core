"""Phase-1B export contracts (plan §8: full-state container + outer envelope).

`ExportContainer` serializes complete governance state: policy profile, full
canonical 9-field model manifest, identities, revisions, active templates,
retired templates, candidates in all five statuses with evidence logs and
exemplars, audit tombstones, and the outer AEAD envelope keyed by
``K_envelope``. Actual encryption and Argon2id derivation belong to Task 6
(`storage/export.py`); this module defines the shape only.
"""

import json
from dataclasses import dataclass, field

from facecore.contracts.candidate import CandidateTemplate
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

    def _blob_payload(self, blob: EncryptedBlob | None) -> object:
        return blob.to_dict() if blob is not None else None

    def to_json(self) -> str:
        payload = {
            "metadata": {
                "export_format_version": self.metadata.export_format_version,
                "kdf_algorithm": self.metadata.kdf_algorithm,
                "kdf_version": self.metadata.kdf_version,
                "kdf_salt_hex": self.metadata.kdf_salt_hex,
                "kdf_memory_cost_kib": self.metadata.kdf_memory_cost_kib,
                "kdf_time_cost": self.metadata.kdf_time_cost,
                "kdf_parallelism": self.metadata.kdf_parallelism,
                "created_at": self.metadata.created_at,
            },
            "policy": {
                "governance_policy_version": (
                    self.policy.governance_policy_version
                ),
                "provisional": self.policy.provisional,
            },
            "model_manifest": self.model_manifest.to_dict(),
            "identities": [
                {
                    "identity_id": i.identity_id,
                    "display_name": i.display_name,
                    "status": i.status,
                    "current_revision": i.current_revision,
                }
                for i in self.identities
            ],
            "active_templates": [
                {
                    "template_id": t.template_id,
                    "identity_id": t.identity_id,
                    "model_version": t.model_version,
                    "embedding_dim": t.embedding_dim,
                    "revision": t.revision.revision,
                    "key_id": t.key_id,
                    "encrypted_exemplar": self._blob_payload(
                        t.encrypted_exemplar
                    ),
                }
                for t in self.active_templates
            ],
            "retired_templates": [
                {
                    "template_id": t.template_id,
                    "identity_id": t.identity_id,
                    "model_version": t.model_version,
                    "embedding_dim": t.embedding_dim,
                    "revision": t.revision.revision,
                    "key_id": t.key_id,
                    "encrypted_exemplar": self._blob_payload(
                        t.encrypted_exemplar
                    ),
                }
                for t in self.retired_templates
            ],
            "candidates": [
                {
                    "template_id": c.template_id,
                    "identity_id": c.identity_id,
                    "generation_id": c.generation_id,
                    "status": c.status.value,
                    "additional_corroboration_count": (
                        c.additional_corroboration_count
                    ),
                    "evidence_log": [
                        {
                            "event_type": e.event_type,
                            "timestamp": e.timestamp,
                            "sequence_number": e.sequence_number,
                            "score": e.score,
                        }
                        for e in c.evidence_log
                    ],
                }
                for c in self.candidates
            ],
            "revisions": [
                {
                    "revision": r.revision,
                    "template_id": r.template_id,
                    "supersedes": r.supersedes,
                }
                for r in self.revisions
            ],
            "tombstones": [
                {
                    "target_type": t.target_type,
                    "target_id": t.target_id,
                    "key_ids": list(t.key_ids),
                    "status": t.status,
                }
                for t in self.tombstones
            ],
            "envelope": self._blob_payload(self.envelope),
        }
        return json.dumps(payload)
