"""ModelManifest skeleton. Fields populated verbatim from Spike S1.

S1-UNVERIFIED fields (weight SHA-256, mobile-usability output) stay ``None``
until the pre-Task-6 completion list runs. Git blob SHA-1 values are locators
only and are never backfilled here as weight checksums.
"""

from dataclasses import dataclass
from enum import Enum


class ProvenanceStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    source_repo: str
    artifact_url: str
    retrieval_date: str
    code_license: str
    weight_license: str
    license_locator: str
    weight_sha256: str | None
    provenance: ProvenanceStatus
    provenance_note: str
    embedding_dim: int
    input_width: int
    input_height: int
    mobile_usability: str | None

    @classmethod
    def sface_2021dec_fp32(cls) -> "ModelManifest":
        return cls(
            model_id="face_recognition_sface_2021dec",
            source_repo="opencv/opencv_zoo",
            artifact_url=(
                "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            ),
            retrieval_date="2026-09-11",
            code_license="Apache-2.0",
            weight_license="Apache-2.0",
            license_locator=(
                "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
                "models/face_recognition_sface/LICENSE"
            ),
            # Measured 2026-09-11 (shasum -a 256) against the downloaded
            # artifact; == hash quoted in upstream issue #313.
            weight_sha256=(
                "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
            ),
            provenance=ProvenanceStatus.UNRESOLVED,
            provenance_note=(
                "Upstream issue #313 OPEN as of 2026-09-11; no maintainer reply."
            ),
            embedding_dim=128,
            input_width=112,
            input_height=112,
            # Measured 2026-09-11 via ORT 1.30.0 mobile-usability checker.
            mobile_usability=(
                "2026-09-11 checker: NNAPI 87/87 YES; CoreML-NN 87/87 YES; "
                "CoreML-MLProgram NO "
                "(unsupported: BatchNormalization, Flatten)"
            ),
        )

    def verify_sha256(self, actual: str) -> bool:
        """Fail closed: no recorded hash means no match, before any inference."""
        if self.weight_sha256 is None:
            return False
        return actual == self.weight_sha256
