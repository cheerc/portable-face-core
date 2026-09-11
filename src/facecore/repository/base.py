"""Repository protocol: revision-shaped read/write surface Phase 1B will persist.

Phase-1B methods (persistence, encryption, candidates, promotion, retirement,
deletion) are deliberately absent — Phase 1B adds them behind this protocol.
"""

from dataclasses import dataclass
from typing import Protocol

from facecore.contracts.template import FaceTemplate


@dataclass(frozen=True)
class IdentityView:
    identity_id: str
    display_name: str
    active_template: FaceTemplate
    current_revision: int


class Repository(Protocol):
    def create_identity(
        self, identity_id: str, display_name: str, template: FaceTemplate
    ) -> IdentityView: ...
    def get_identity(self, identity_id: str) -> IdentityView | None: ...
    def list_active_templates(self) -> list[FaceTemplate]: ...
    def current_revision(self, identity_id: str) -> int: ...
    def append_revision(
        self, identity_id: str, template: FaceTemplate
    ) -> IdentityView: ...
