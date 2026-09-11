"""In-memory evaluation session: enroll N identities, identify probes (Task 8).

Identity is built only after the quality and single-face gates both pass;
invalid enrollment leaves no session state.
"""

from facecore import SCHEMA_VERSION
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.pipeline.decode import decode_image
from facecore.pipeline.detect import DetectedFace, enforce_single_face
from facecore.pipeline.yunet import YuNetDetector
from facecore.repository.memory import InMemoryRepository


class EvaluationSession:
    def __init__(self, detector: YuNetDetector | None = None) -> None:
        self._repository = InMemoryRepository()
        self._detector = detector

    def identity_count(self) -> int:
        return len(self._repository.list_active_templates())

    def enroll(self, data: bytes, identity_id: str) -> str:
        """Enroll one identity from image bytes; invalid input changes nothing.

        Undecodable bytes raise InputDecodeError (exit-2 type, never
        invalid_input — Task 3 contract); the CLI maps it to exit 2.
        With a detector (Task 5.5), faces come from the real adapter;
        without one, any decodable image is no usable face (Task 5 gate).
        """
        decoded = decode_image(data)
        faces: list[DetectedFace]
        if self._detector is None:
            faces = []
        else:
            faces = self._detector.detect(decoded)
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
