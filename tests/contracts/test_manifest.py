"""Task 1 RED/GREEN + Task 6 measured fill: manifest fields from S1/Phase-1.

Task 6 update: the pre-Task-6 completion list ran 2026-09-11, so the
measured fields now carry values; the no-backfill rule still holds —
values come from shasum/checker runs, never from git blob SHA-1.
"""

from facecore.contracts.manifest import ModelManifest, ProvenanceStatus


def test_measured_fields_carry_phase1_values_not_backfilled() -> None:
    manifest = ModelManifest.sface_2021dec_fp32()
    # Measured 2026-09-11; == hash quoted in upstream issue #313.
    assert (
        manifest.weight_sha256
        == "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
    )
    assert manifest.mobile_usability is not None
    assert "BatchNormalization" in manifest.mobile_usability
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
