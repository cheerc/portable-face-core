"""Task §6-4 RED/GREEN: identity add runs the true one-shot pipeline.

Source of truth: spec §7 (decode → exactly-one-face → quality → align →
embed) + §10/§11 (persistent revision 1) + acceptance §1 (photo-bytes gap).

All images/vectors are SYNTHETIC (Pillow checkerboard + stub detector /
embedder) — no biometric data enters the repo. The commander verifies
with 23 real photos (commander domain).

RED: ``ImportError: cannot import name '_enroll_photo'``.
"""

import io
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facecore.cli import _enroll_photo, cmd_identity_add
from facecore.pipeline.align import AlignedCrop
from facecore.pipeline.decode import DecodedImage
from facecore.pipeline.detect import DetectedFace


def _checkerboard(size: int = 200, cells: int = 8) -> bytes:
    """Mid-grey checkerboard PNG: sharp, mid-exposure, zero clipped."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    assert px is not None
    step = size // cells
    for y in range(size):
        for x in range(size):
            v = 100 if ((x // step) + (y // step)) % 2 == 0 else 160
            px[x, y] = (v, v, v)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_STUB_VECTOR = np.arange(1.0, 9.0)
_STUB_VECTOR = _STUB_VECTOR / np.linalg.norm(_STUB_VECTOR)


class _StubDetector:
    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = faces

    def detect(
        self, decoded: DecodedImage, score_threshold: float | None = None
    ) -> list[DetectedFace]:
        _ = (decoded, score_threshold)
        return list(self._faces)


class _StubEmbedder:
    def __init__(self, vector: np.ndarray | None = None) -> None:
        self._vector = (
            vector if vector is not None else _STUB_VECTOR.copy()
        )
        self.calls = 0

    @property
    def model_version(self) -> str:
        return "stub-embed-v1"

    def embed(self, crop: AlignedCrop) -> tuple[np.ndarray, str]:
        self.calls += 1
        return self._vector.copy(), "stub-embed-v1"


def _good_face() -> DetectedFace:
    return DetectedFace(
        box=(10.0, 10.0, 180.0, 180.0),
        landmarks=((60.0, 80.0), (140.0, 80.0), (100.0, 110.0),
                   (70.0, 150.0), (130.0, 150.0)),
        confidence=0.95,
    )


def _db_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "facecore.db"
    monkeypatch.setenv("FACECORE_DB", str(db))
    return db


def test_add_stores_true_embedding_not_photo_bytes(tmp_path: Path) -> None:
    from facecore.eval.session import EvaluationSession

    photo = _checkerboard()
    session = EvaluationSession(
        detector=_StubDetector([_good_face()]),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
    )
    outcome, vector, exemplar, version = session.enroll_details(photo, "p-01")
    assert outcome == "enrolled"
    assert vector is not None and exemplar is not None
    assert vector.tobytes() != photo
    assert exemplar != photo
    assert version == "stub-embed-v1"
    assert np.allclose(vector, _STUB_VECTOR)


def test_enroll_photo_wires_session_pipeline(tmp_path: Path) -> None:
    photo = _checkerboard()
    embedder = _StubEmbedder()
    outcome, vector, exemplar, version = _enroll_photo(
        photo,
        None,
        detector=_StubDetector([_good_face()]),  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
    )
    assert outcome == "enrolled"
    assert embedder.calls == 1
    assert vector is not None and exemplar is not None
    assert vector.tobytes() != photo
    assert version == "stub-embed-v1"


def test_zero_face_enroll_photo_invalid(tmp_path: Path) -> None:
    outcome, vector, exemplar, _ = _enroll_photo(
        _checkerboard(),
        None,
        detector=_StubDetector([]),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
    )
    assert outcome == "invalid_input"
    assert vector is None and exemplar is None


def test_multi_face_enroll_photo_invalid(tmp_path: Path) -> None:
    outcome, vector, exemplar, _ = _enroll_photo(
        _checkerboard(),
        None,
        detector=_StubDetector([_good_face(), _good_face()]),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
    )
    assert outcome == "invalid_input"
    assert vector is None and exemplar is None


def test_cmd_add_without_models_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _db_env(tmp_path, monkeypatch)
    photo = tmp_path / "p.png"
    photo.write_bytes(_checkerboard())
    rc = cmd_identity_add("person-01", "Test", photo, None)
    assert rc == 2
    assert not db.exists()


def test_cmd_add_undecodable_exit2_no_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _db_env(tmp_path, monkeypatch)
    photo = tmp_path / "bad.bin"
    photo.write_bytes(b"not an image at all")
    rc = cmd_identity_add("person-01", "Test", photo, tmp_path / "models")
    assert rc == 2
    assert not db.exists()


def test_cmd_add_duplicate_stays_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from facecore.governance.lifecycle import LifecycleManager
    from facecore.storage.key_provider import InMemoryKeyProvider
    from facecore.storage.sqlite_repo import SQLiteRepository

    db = _db_env(tmp_path, monkeypatch)
    repo = SQLiteRepository(str(db), InMemoryKeyProvider())
    repo.initialize()
    manager = LifecycleManager(repo)
    manager.add_identity("person-01", "Test", b"e" * 64, b"x" * 64)
    photo = tmp_path / "p.png"
    photo.write_bytes(_checkerboard())
    rc = cmd_identity_add(
        "person-01",
        "Someone Else",
        photo,
        None,
        detector=_StubDetector([_good_face()]),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
    )
    assert rc == 4
    con = sqlite3.connect(str(db))
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM face_templates WHERE identity_id = ?",
            ("person-01",),
        ).fetchone()[0]
    finally:
        con.close()
    assert count == 1


def test_full_add_roundtrip_db_matches_stub_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    """End-to-end: cmd add → DB decrypt equals the stub embedder vector."""
    import json

    from facecore.storage.key_provider import FileKeyProvider
    from facecore.storage.sqlite_repo import SQLiteRepository

    db = _db_env(tmp_path, monkeypatch)
    photo = tmp_path / "p.png"
    photo.write_bytes(_checkerboard())
    rc = cmd_identity_add(
        "person-01",
        "Test",
        photo,
        None,
        detector=_StubDetector([_good_face()]),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])  # type: ignore[attr-defined]
    assert out["status"] == "ok"
    assert out["revision"] == 1
    key_dir = db.parent / "keys"
    repo = SQLiteRepository(
        str(db), FileKeyProvider(key_dir=key_dir, db_path=db)
    )
    repo.initialize()
    assert repo.read_embedding(out["template_id"]) == _STUB_VECTOR.tobytes()
    con = sqlite3.connect(str(db))
    try:
        row = con.execute(
            "SELECT exemplar_blob, LENGTH(exemplar_blob)"
            " FROM face_templates WHERE id = ?",
            (out["template_id"],),
        ).fetchone()
    finally:
        con.close()
    assert row is not None and row[1] > 0


def test_zero_face_cmd_add_no_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = _db_env(tmp_path, monkeypatch)
    photo = tmp_path / "p.png"
    photo.write_bytes(_checkerboard())
    rc = cmd_identity_add(
        "person-01",
        "Test",
        photo,
        None,
        detector=_StubDetector([]),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),  # type: ignore[arg-type]
    )
    assert rc == 2
    assert not db.exists()
