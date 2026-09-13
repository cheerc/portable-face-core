"""M2 RED/GREEN: Pair 2 manifests (int8bq embedder + 2026may detector).

RED: sface_2021dec_int8bq / yunet_2026may constructors do not exist yet.
GREEN: measured fields carry commander-verified values; Pair 1 untouched;
provenance stays UNRESOLVED (S1 item 4 + open issue #313).
"""

from facecore.contracts.manifest import (
    ModelManifest,
    ProvenanceStatus,
    YUNET_2026MAY_SHA,
)

UPSTREAM_INT8BQ_SHA = (
    "fb143eea07838aa532d1c95df5f69899974ea0140e1fba05e94204be13ed74ee"
)
DERIVED_INT8BQ_SHA = (
    "853a6d3bd14dc247123437eacf479fd5c53b1c19ebf3e08c3413088f9bfc4162"
)


def test_int8bq_manifest_carries_measured_sha() -> None:
    manifest = ModelManifest.sface_2021dec_int8bq()
    assert manifest.weight_sha256 == DERIVED_INT8BQ_SHA
    assert UPSTREAM_INT8BQ_SHA in manifest.provenance_note
    assert DERIVED_INT8BQ_SHA in manifest.provenance_note
    assert manifest.model_id == "face_recognition_sface_2021dec_int8bq"
    assert manifest.embedding_dim == 128
    assert manifest.input_height == 112
    assert manifest.input_width == 112
    assert manifest.provenance == ProvenanceStatus.UNRESOLVED
    assert "313" in manifest.provenance_note


def test_int8bq_quantization_annotated() -> None:
    manifest = ModelManifest.sface_2021dec_int8bq()
    assert "int8" in manifest.mobile_usability.lower()


def test_int8bq_hash_mismatch_fails_closed() -> None:
    manifest = ModelManifest.sface_2021dec_int8bq()
    assert manifest.verify_sha256("deadbeef") is False
    assert manifest.verify_sha256(DERIVED_INT8BQ_SHA) is True
    # Upstream bytes are NOT the deployment artifact: gate refuses them.
    assert manifest.verify_sha256(UPSTREAM_INT8BQ_SHA) is False


def test_2026may_detector_sha_recorded() -> None:
    assert (
        YUNET_2026MAY_SHA
        == "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
    )


def test_pair1_manifest_untouched() -> None:
    manifest = ModelManifest.sface_2021dec_fp32()
    assert (
        manifest.weight_sha256
        == "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
    )
    assert manifest.model_id == "face_recognition_sface_2021dec"
