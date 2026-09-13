"""In-memory evaluation session: enroll N identities, identify probes.

Identity is built only after decode, detection, quality (frozen v1,
measured — closes 5.5 N1), single-face, align, and embed all pass;
invalid enrollment leaves no session state.
"""

import numpy as np

from facecore import SCHEMA_VERSION
from facecore.contracts.policy import PolicyProfile
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.pipeline.align import align_crop
from facecore.pipeline.decode import decode_image
from facecore.pipeline.detect import DetectedFace, enforce_single_face
from facecore.pipeline.embed import Embedder
from facecore.pipeline.measure import exposure_of, landmarks_of, pose_of, sharpness_of
from facecore.pipeline.quality import evaluate_quality
from facecore.pipeline.yunet import YuNetDetector
from facecore.repository.memory import InMemoryRepository


class EvaluationSession:
    def __init__(
        self,
        detector: YuNetDetector | None = None,
        embedder: Embedder | None = None,
        policy: PolicyProfile | None = None,
        detector_gate: float | None = None,
    ) -> None:
        self._repository = InMemoryRepository()
        self._detector = detector
        self._embedder = embedder
        self._policy = policy or PolicyProfile.frozen_v1()
        # Explicit decode-time gate; None preserves the detector default
        # (0.9). Set alongside a with_detector_gate policy so both gates
        # agree; mismatched pairing is a caller error, not silently fixed.
        self._detector_gate = detector_gate
        self._vectors: dict[str, np.ndarray] = {}
        self.last_quality_codes: list[str] = []

    def identity_count(self) -> int:
        return len(self._repository.list_active_templates())

    def gallery(self) -> dict[str, np.ndarray]:
        return dict(self._vectors)

    def snapshot_gallery(self) -> dict[str, np.ndarray]:
        """Deep copy of the live gallery for replay template-state digests.

        Task 9 wiring: lets the replay harness compare session template
        state across A/B runs in one shared shape. Read-only; session
        state is never mutated through the snapshot.
        """
        return {key: value.copy() for key, value in self._vectors.items()}

    def model_version(self) -> str:
        if self._embedder is not None:
            return self._embedder.model_version
        return "unevaluated"

    def enroll(self, data: bytes, identity_id: str) -> str:
        """Enroll one identity from image bytes; invalid input changes nothing.

        Undecodable bytes raise InputDecodeError (exit-2 type, never
        invalid_input — Task 3 contract); the CLI maps it to exit 2.
        Without detector+embedder, any decodable image is no usable face
        (Task 5 gate); with them, the full pipeline runs with measured
        quality gates (frozen v1).
        """
        decoded = decode_image(data)
        faces: list[DetectedFace]
        if self._detector is None:
            faces = []
        elif self._detector_gate is None:
            faces = self._detector.detect(decoded)
        else:
            faces = self._detector.detect(
                decoded, score_threshold=self._detector_gate
            )
        status, _code, face = enforce_single_face(faces)
        if status != "ok" or face is None:
            return "invalid_input"
        if self._detector is not None and self._embedder is not None:
            box = face.box
            shorter = int(min(box[2], box[3]))
            crop = align_crop(decoded.pixels, decoded.width, decoded.height, face)
            yaw, pitch = pose_of(face)
            mean_luma, clipped = exposure_of(crop)
            verdict = evaluate_quality(
                self._policy,
                detector_confidence=face.confidence,
                shorter_side_px=shorter,
                sharpness=sharpness_of(crop),
                mean_luma=mean_luma,
                clipped_fraction=clipped,
                yaw_deg=yaw,
                pitch_deg=pitch,
                landmarks=landmarks_of(face),
            )
            self.last_quality_codes = list(verdict.reason_codes)
            if verdict.status != "accepted":
                return "invalid_input"
            vector, model_version = self._embedder.embed(crop)
        else:
            self.last_quality_codes = []
            vector = np.zeros(4)
            model_version = "unevaluated"
        template = FaceTemplate(
            template_id=f"t-{identity_id}-1",
            identity_id=identity_id,
            model_version=model_version,
            embedding_dim=int(vector.shape[0]),
            revision=TemplateRevision(
                revision=1, template_id=f"t-{identity_id}-1", supersedes=None
            ),
        )
        self._repository.create_identity(identity_id, identity_id, template)
        self._vectors[identity_id] = vector
        return "enrolled"

    def schema_version(self) -> int:
        return SCHEMA_VERSION
