"""FaceTemplate / TemplateRevision: revision-shaped, never mutated in place.

Phase-1A frozen compatibility: ``encrypted_exemplar``, ``exemplar_crop_box``,
``exemplar_landmarks``, and ``key_id`` default to absent so every pre-1B
construction site (eval session, frozen 1A tests) keeps compiling unchanged.
Phase-1B governance paths must call `assert_governance_ready` before use.
"""

from dataclasses import dataclass

from facecore.contracts.crypto import EncryptedBlob


@dataclass(frozen=True)
class TemplateRevision:
    revision: int
    template_id: str
    supersedes: str | None


@dataclass(frozen=True)
class FaceTemplate:
    template_id: str
    identity_id: str
    model_version: str
    embedding_dim: int
    revision: TemplateRevision
    encrypted_exemplar: EncryptedBlob | None = None
    exemplar_crop_box: tuple[float, float, float, float] | None = None
    exemplar_landmarks: tuple[tuple[float, float], ...] | None = None
    key_id: str | None = None

    def assert_governance_ready(self) -> None:
        """Phase-1B gate: exemplar + crop + landmarks + key_id all present."""
        missing = [
            name
            for name, value in (
                ("encrypted_exemplar", self.encrypted_exemplar),
                ("exemplar_crop_box", self.exemplar_crop_box),
                ("exemplar_landmarks", self.exemplar_landmarks),
                ("key_id", self.key_id),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "FaceTemplate missing governance exemplar fields: "
                + ", ".join(missing)
            )

    def assert_comparable(self, other: "FaceTemplate") -> None:
        """Old and new generations never compare as if compatible (spec sec 12)."""
        if self.model_version != other.model_version:
            raise ValueError(
                "cross-model comparison refused: "
                f"{self.model_version!r} vs {other.model_version!r}"
            )
