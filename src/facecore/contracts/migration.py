"""Phase-1B model migration contracts (S1B §9 canonical 9-field predicate).

Nine canonical fields; hard mismatch (fields 1, 4, 5, 6, 7, 8, 9) raises
`ModelIncompatibilityError` (exit code 3) with zero database writes, while
detector/preprocessing generation drift (fields 2, 3) returns
``MIGRATION_REQUIRED`` and routes to the Task 7 re-embedder. Candidate
all-status rule (plan Task 7): ``pending`` re-embeds; ``promoted`` becomes
``generation_retired``; ``rejected``/``expired`` stay archived with G1.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class ModelIncompatibilityError(Exception):
    """Raised on hard model-manifest mismatch (exit code 3)."""


class GenerationStatus(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    RE_ENROLLMENT_REQUIRED = "re_enrollment_required"


HARD_INCOMPATIBLE_FIELDS: tuple[str, ...] = (
    "embedder_artifact_hash",
    "tensor_layout",
    "normalization_contract",
    "embedding_dimension",
    "numerical_precision",
    "quantization_type",
    "execution_runtime",
    "score_metric",
)

MIGRATION_GENERATION_FIELDS: tuple[str, ...] = (
    "detector_generation",
    "preprocessing_generation",
)


@dataclass(frozen=True)
class ModelMigrationManifest:
    embedder_artifact_hash: str
    detector_generation: str
    preprocessing_generation: str
    tensor_layout: str
    normalization_contract: str
    embedding_dimension: int
    numerical_precision: str
    quantization_type: str
    execution_runtime: str
    score_metric: str = "cosine"

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedder_artifact_hash": self.embedder_artifact_hash,
            "detector_generation": self.detector_generation,
            "preprocessing_generation": self.preprocessing_generation,
            "tensor_layout": self.tensor_layout,
            "normalization_contract": self.normalization_contract,
            "embedding_dimension": self.embedding_dimension,
            "numerical_precision": self.numerical_precision,
            "quantization_type": self.quantization_type,
            "execution_runtime": self.execution_runtime,
            "score_metric": self.score_metric,
        }

    def check_compatibility(
        self, runtime: "ModelMigrationManifest"
    ) -> Literal["COMPATIBLE", "MIGRATION_REQUIRED"]:
        stored = self.to_dict()
        live = runtime.to_dict()
        for field_name in HARD_INCOMPATIBLE_FIELDS:
            if stored.get(field_name) != live.get(field_name):
                raise ModelIncompatibilityError(
                    f"Model incompatibility in field {field_name!r}: "
                    f"archive={stored.get(field_name)!r} != "
                    f"runtime={live.get(field_name)!r}"
                )
        for field_name in MIGRATION_GENERATION_FIELDS:
            if stored.get(field_name) != live.get(field_name):
                return "MIGRATION_REQUIRED"
        return "COMPATIBLE"


@dataclass(frozen=True)
class MigrationResult:
    from_generation: str
    to_generation: str
    active_migrated: int = 0
    candidates_migrated: int = 0
    candidates_generation_retired: int = 0
    identities_needing_re_enrollment: tuple[str, ...] = ()
    status: GenerationStatus = GenerationStatus.CURRENT

    def __post_init__(self) -> None:
        for name in (
            "active_migrated",
            "candidates_migrated",
            "candidates_generation_retired",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
