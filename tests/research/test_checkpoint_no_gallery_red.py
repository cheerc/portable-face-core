"""RED: checkpoint builds detector only, no gallery (#89 follow-up fix).

Defect: cli.py true path unconditionally builds the full gallery (true
YuNet + SFace vs corpus) BEFORE the checkpoint branch — but checkpoint
never uses the gallery (synthetic gen/digest, scorer uses detector
only). Production true YuNet returns 0 faces on synthetic corpus, so
faceless-camera runs die at gallery construction (input_no_face, exit 2)
before presence logic runs. PR#94's canonical RED masked this: its
MagicMock(spec=YuNetDetector) returns a face for EVERY image, so the
gallery always built in tests.

RED: detector double returns 0 faces on CORPUS images (true-YuNet-like)
but the camera frames are faceless too — assert the cmd_live layer
completes checkpoint (exit 0 + presence_stop=False) instead of dying at
gallery build. The runner total is NOT asserted (synthetic zero-origin
frame clocks read as time-backwards in the fixed-window predicate — a
pre-existing hermetic-test clock artifact shared with the canonical RED
suite, out of scope here).
Pre-fix: cmd_live exit 2 (input_no_face). If GREEN pre-fix, STOP (report).

Synthetic UIDs only (F1 governance).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from facecore.live.capture import FakeCapture, FramePacket  # noqa: E402
from facecore.pipeline.yunet import YuNetDetector  # noqa: E402

BUILTIN_UID = "FFFF0000-0000-4000-8000-000000000002"
IPHONE_UID = "AAAA0000-0000-4000-8000-000000000001"


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "nogallery-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet-test",
        "quality_policy_version": "q-test-v1",
        "continuity_max_center_delta_ratio": 0.50,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


def _frames(h: int = 16, w: int = 16, n: int = 27) -> list[FramePacket]:
    return [
        FramePacket(
            sequence=seq,
            captured_ns=(seq - 1) * 200_000_000,
            rgb=np.ascontiguousarray(
                np.full((h, w, 3), 120 + (seq % 40), dtype=np.uint8)
            ),
        )
        for seq in range(1, n + 1)
    ]


def _synthetic_png(path: Path, seed: int) -> None:
    from PIL import Image

    arr = np.full((200, 200, 3), 120, dtype=np.uint8)
    arr[::2, ::2] = 160
    arr[1::2, 1::2] = 80
    arr[20:80, 20:80, 0] = (150 + seed) % 256
    Image.fromarray(arr, mode="RGB").save(path)


def _manifest(tmp_path: Path) -> Path:
    p1 = tmp_path / "enroll_01.png"
    p2 = tmp_path / "enroll_02.png"
    _synthetic_png(p1, 1)
    _synthetic_png(p2, 2)
    manifest = {
        "files": [
            {
                "path": str(p1),
                "role": "enrollment",
                "identity": "person-01",
                "conditions": [],
            },
            {
                "path": str(p2),
                "role": "enrollment",
                "identity": "person-02",
                "conditions": [],
            },
        ]
    }
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(manifest))
    return corpus


class _ScriptedEmbedder:
    model_version = "sface_2021dec"

    def embed(self, crop):  # noqa: ANN001, ANN202
        arr = np.array((1.0, 0.0, 0.0), dtype=np.float32)
        return arr / float(np.linalg.norm(arr)), "sface_2021dec"


def _run_checkpoint_faceless_corpus(tmp_path: Path, tag: str, **kw):
    """Checkpoint run: corpus images yield 0 faces (true-YuNet-like)."""
    from live_checkpoint import run_checkpoint

    store = tmp_path / f"store-{tag}"
    keys = tmp_path / f"keys-{tag}"
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    # Detector double: 0 faces on EVERYTHING (corpus AND camera frames),
    # like production true YuNet on synthetic/faceless pixels. Still a
    # YuNet-spec double so the isinstance guard passes.
    empty_detector = MagicMock(spec=YuNetDetector)
    empty_detector.detect.return_value = []
    base = dict(
        device="1",
        profile_path=_write_profile(tmp_path),
        record_consent=True,
        image_consent=True,
        store=store,
        key_dir=keys,
        models=models,
        corpus=_manifest(tmp_path),
        capture_factory=lambda _d: FakeCapture(frames=_frames()),
        detector_factory=lambda _m: empty_detector,
        embedder_factory=lambda _m: _ScriptedEmbedder(),
        expected_builtin_unique_id=BUILTIN_UID,
        expected_builtin_shape="16x16",
        camera_identity_probe=lambda: ([IPHONE_UID, BUILTIN_UID], 2),
        experiment_id="exp-nogallery",
        session_id=f"chk-{tag}",
    )
    base.update(kw)
    return run_checkpoint(**base)


def test_nogallery_faceless_checkpoint_completes(tmp_path: Path) -> None:
    """Faceless checkpoint must NOT die at gallery construction.

    Asserts the cmd_live layer (exit 0 + presence_stop False), not the
    runner total: synthetic zero-origin frame clocks read as
    time-backwards in the runner's fixed-window predicate (pre-existing
    hermetic-test clock artifact, same as the canonical RED suite) —
    that runner-level incompleteness is out of scope for this RED.
    The defect under test is gallery construction (exit 2,
    input_no_face); reaching cmd_live exit 0 proves it is gone.
    """
    code, summary = _run_checkpoint_faceless_corpus(tmp_path, "faceless")
    live = summary["phases"].get("live", {})
    assert live.get("exit_code") == 0, (
        f"NOGALLERY-RED: cmd_live layer exited {live.get('exit_code')} "
        f"(expected 0 — gallery construction must not refuse); "
        f"live={json.dumps(live, sort_keys=True)[:400]}"
    )
    assert live.get("presence_stop") is False, (
        f"NOGALLERY-RED: faceless run must carry presence_stop=False: {live}"
    )
