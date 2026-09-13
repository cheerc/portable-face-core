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


SFACE_FP32_SHA = (
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
)
YUNET_2023MAR_SHA = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)
YUNET_2026MAY_SHA = (
    "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
)


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

    @classmethod
    def sface_2021dec_int8bq(cls) -> "ModelManifest":
        return cls(
            model_id="face_recognition_sface_2021dec_int8bq",
            source_repo="opencv/opencv_zoo",
            artifact_url=(
                "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                "models/face_recognition_sface/"
                "face_recognition_sface_2021dec_int8bq.onnx"
            ),
            retrieval_date="2026-09-13",
            code_license="Apache-2.0",
            weight_license="Apache-2.0",
            license_locator=(
                "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
                "models/face_recognition_sface/LICENSE"
            ),
            # Derived de-input artifact SHA-256 (shasum -a 256,
            # 2026-09-13): deterministic output of
            # scripts/derive_int8bq_deinput.py from the upstream bytes.
            # The integrity gate verifies this derived file; the upstream
            # SHA stays in provenance_note for audit.
            weight_sha256=(
                "853a6d3bd14dc247123437eacf479fd5c53b1c19ebf3e08c3413088f9bfc4162"
            ),
            provenance=ProvenanceStatus.UNRESOLVED,
            provenance_note=(
                "Derived de-input artifact: removed 174 redundant graph "
                "inputs shadowing DequantizeLinear outputs "
                "(block_quantize.py artifact; data-only input) from upstream "
                "SHA fb143eea07838aa532d1c95df5f69899974ea0140e1fba05e94204"
                "be13ed74ee; derived SHA 853a6d3bd14dc247123437eacf479fd5"
                "c53b1c19ebf3e08c3413088f9bfc4162. "
                "Upstream issue #313 OPEN as of 2026-09-11 (S1); applies to "
                "the int8bq lineage identically (same directory, same file "
                "lineage). No maintainer reply on file."
            ),
            embedding_dim=128,
            input_width=112,
            input_height=112,
            # Measured 2026-09-13 via ORT 1.30.0 mobile-usability checker
            # (onnx module installed for the checker run). Block-quantized
            # int8 (block_quantize.py, block_size=64); Zoo eval accuracy
            # 0.9932 vs 0.9940 fp32 (-0.08pp); ~3.6x smaller weights
            # (38.7 MB -> 10.7 MB).
            mobile_usability=(
                "quantized int8bq (block_quantize.py, block_size=64). "
                "2026-09-13 checker (ORT 1.30.0): NNAPI 143/143 YES; "
                "CoreML-NN 114/143 YES "
                "(unsupported: ai.onnx:DequantizeLinear); "
                "CoreML-MLProgram NO "
                "(unsupported: BatchNormalization, DequantizeLinear, Flatten)"
            ),
        )

    def verify_sha256(self, actual: str) -> bool:
        """Fail closed: no recorded hash means no match, before any inference."""
        if self.weight_sha256 is None:
            return False
        return actual == self.weight_sha256
