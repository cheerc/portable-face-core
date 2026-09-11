"""InMemoryRepository: process-local dict store, discarded at exit."""

import copy
from dataclasses import dataclass, field

from facecore.contracts.template import FaceTemplate
from facecore.repository.base import IdentityView


@dataclass
class _Record:
    display_name: str
    revisions: list[FaceTemplate] = field(default_factory=list)


class InMemoryRepository:
    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}

    def create_identity(
        self, identity_id: str, display_name: str, template: FaceTemplate
    ) -> IdentityView:
        if identity_id in self._records:
            raise KeyError(f"identity already exists: {identity_id}")
        self._records[identity_id] = _Record(
            display_name=display_name, revisions=[template]
        )
        return self._view(identity_id)

    def get_identity(self, identity_id: str) -> IdentityView | None:
        if identity_id not in self._records:
            return None
        return self._view(identity_id)

    def list_active_templates(self) -> list[FaceTemplate]:
        return [
            copy.deepcopy(record.revisions[-1]) for record in self._records.values()
        ]

    def current_revision(self, identity_id: str) -> int:
        record = self._records.get(identity_id)
        if record is None:
            return 0
        return record.revisions[-1].revision.revision

    def append_revision(self, identity_id: str, template: FaceTemplate) -> IdentityView:
        record = self._records.get(identity_id)
        if record is None:
            raise KeyError(f"unknown identity: {identity_id}")
        # Copy rather than mutate: existing revisions are never touched in place.
        record.revisions.append(copy.deepcopy(template))
        return self._view(identity_id)

    def _view(self, identity_id: str) -> IdentityView:
        record = self._records[identity_id]
        active = copy.deepcopy(record.revisions[-1])
        return IdentityView(
            identity_id=identity_id,
            display_name=record.display_name,
            active_template=active,
            current_revision=active.revision.revision,
        )
