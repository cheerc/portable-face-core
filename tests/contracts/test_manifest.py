"""Task 1 RED/GREEN: ModelManifest skeleton — S1 fields verbatim, UNVERIFIED empty."""

from facecore.contracts.manifest import ModelManifest, ProvenanceStatus


def test_s1_unverified_fields_stay_empty_not_backfilled() -> None:
    manifest = ModelManifest.sface_2021dec_fp32()
    # S1 recorded retrieval 2026-09-11; weight hash/checker are UNVERIFIED there.
    assert manifest.weight_sha256 is None
    assert manifest.mobile_usability is None
    assert manifest.provenance == ProvenanceStatus.UNRESOLVED


def test_manifest_carries_s1_verbatim_identity_and_io() -> None:
    manifest = ModelManifest.sface_2021dec_fp32()
    assert manifest.model_id == "face_recognition_sface_2021dec"
    assert manifest.embedding_dim == 128
    assert manifest.input_height == 112
    assert manifest.input_width == 112


def test_hash_mismatch_fails_closed() -> None:
    manifest = ModelManifest.sface_2021dec_fp32()
    assert manifest.verify_sha256("deadbeef") is False
