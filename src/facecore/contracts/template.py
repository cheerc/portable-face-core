"""FaceTemplate / TemplateRevision: revision-shaped, never mutated in place."""

from dataclasses import dataclass


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

    def assert_comparable(self, other: "FaceTemplate") -> None:
        """Old and new generations never compare as if compatible (spec sec 12)."""
        if self.model_version != other.model_version:
            raise ValueError(
                "cross-model comparison refused: "
                f"{self.model_version!r} vs {other.model_version!r}"
            )
