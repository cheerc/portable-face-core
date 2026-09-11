"""In-memory evaluation session: enroll N identities, identify probes (Task 8).

Identity is built only after the quality and single-face gates both pass;
invalid enrollment leaves no session state.
"""

from facecore import SCHEMA_VERSION
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.pipeline.decode import decode_image
from facecore.pipeline.detect import DetectedFace, enforce_single_face
from facecore.repository.memory import InMemoryRepository


class EvaluationSession:
    def __init__(self) -> None:
        self._repository = InMemoryRepository()

    def identity_count(self) -> int:
        return len(self._repository.list_active_templates())

    def enroll(self, data: bytes, identity_id: str) -> str:
        """Enroll one identity from image bytes; invalid input changes nothing.

        Undecodable bytes raise InputDecodeError (exit-2 type, never
        invalid_input — Task 3 contract); the CLI maps it to exit 2.
        """
        decoded = decode_image(data)
        # Zero-face in Phase 1A: without a detector artifact, any decodable
        # image is treated as no usable face (exactly-one-face gate, Task 5).
        _ = decoded
        faces: list[DetectedFace] = []
        status, _code, _face = enforce_single_face(faces)
        if status != "ok":
            return "invalid_input"
        template = FaceTemplate(
            template_id=f"t-{identity_id}-1",
            identity_id=identity_id,
            model_version="unevaluated",
            embedding_dim=0,
            revision=TemplateRevision(
                revision=1, template_id=f"t-{identity_id}-1", supersedes=None
            ),
        )
        self._repository.create_identity(identity_id, identity_id, template)
        return "enrolled"

    def schema_version(self) -> int:
        return SCHEMA_VERSION
