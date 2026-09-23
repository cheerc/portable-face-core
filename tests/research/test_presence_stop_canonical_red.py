"""Cross-layer RED: presence stop on the true-device canonical path.

Commander gate FAIL (PR #93 @ dd6af365): controller.py swallows
PresenceDetectedError when no frame_transform is present (ui != "qt",
the runner/runbook canonical config), so the canonical path yields
exit 0 + presence_stop=False + a committed faced bundle — a silent pass.

Unlike test_presence_wiring_red.RED2 (which stubbed cmd_live and
hand-printed the stderr line), this file executes the REAL cmd_live +
REAL controller. Only leaf dependencies are injected through the
products' own factory parameters (capture_factory, detector_factory,
embedder_factory) plus a synthetic enrollment manifest — no stubbing
of cmd_live or the controller.

RED-canonical (expected RED pre-fix, GREEN post-fix):
  runner device="0" + face-detecting YuNet-spec detector ->
  exit_code == 4 AND live.presence_stop is True.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SRC = str(REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from facecore.live.capture import FakeCapture, FramePacket  # noqa: E402
from facecore.pipeline.detect import DetectedFace  # noqa: E402
from facecore.pipeline.yunet import YuNetDetector  # noqa: E402


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "canon-red-v1",
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


def _frames() -> list[FramePacket]:
    return [
        FramePacket(
            sequence=seq,
            captured_ns=(seq - 1) * 200_000_000,
            rgb=np.ascontiguousarray(
                np.full((16, 16, 3), 120 + (seq % 40), dtype=np.uint8)
            ),
        )
        for seq in range(1, 28)
    ]


def _face() -> DetectedFace:
    return DetectedFace(
        box=(10.0, 10.0, 40.0, 40.0),
        landmarks=((20.0, 20.0),) * 5,
        confidence=0.99,
    )


def _faced_detector() -> MagicMock:
    """YuNet-spec double whose detect() always finds one face.

    spec= keeps the isinstance(context.detector, YuNetDetector) gate
    green while detect() is scripted — the real hybrid scorer runs.
    """
    det = MagicMock(spec=YuNetDetector)
    det.detect.return_value = [_face()]
    return det


class _ScriptedEmbedder:
    model_version = "sface_2021dec"

    def embed(self, crop):  # noqa: ANN001, ANN202
        arr = np.array((1.0, 0.0, 0.0), dtype=np.float32)
        return arr / float(np.linalg.norm(arr)), "sface_2021dec"


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


def _run_canonical(tmp_path: Path, tag: str):
    from live_checkpoint import run_checkpoint

    store = tmp_path / "store"
    keys = tmp_path / "keys"
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    return run_checkpoint(
        device="0",
        profile_path=_write_profile(tmp_path),
        record_consent=True,
        image_consent=True,
        store=store,
        key_dir=keys,
        models=models,
        corpus=_manifest(tmp_path),
        capture_factory=lambda _d: FakeCapture(frames=_frames()),
        detector_factory=lambda _m: _faced_detector(),
        embedder_factory=lambda _m: _ScriptedEmbedder(),
        experiment_id="exp-canon-red",
        session_id=f"canon-{tag}-{uuid4().hex[:8]}",
        # Pre-existing test path: bypass the #89 identity gate via the
        # same injection seam the gate itself uses for hermetic tests.
        camera_identity_probe=lambda: (["pinned-builtin-uid"], 1),
        expected_builtin_unique_id="pinned-builtin-uid",
        expected_builtin_shape="16x16",
    )


def test_canonical_path_faced_run_stops_loud(tmp_path: Path) -> None:
    """Cross-layer: faced canonical run must exit 4 + presence_stop True."""
    code, summary = _run_canonical(tmp_path, "faced")
    live = summary["phases"]["live"]
    assert code == 4, (
        f"CANON-RED: faced canonical run exited {code} (expected 4); "
        f"live={json.dumps(live, sort_keys=True)[:500]}"
    )
    assert live.get("presence_stop") is True, (
        f"CANON-RED: presence_stop={live.get('presence_stop')!r} "
        f"(expected True — silent pass on the canonical path)"
    )
    # The faced bundle must NOT be committed: list the store and prove
    # no session bundle for this run survived (R4 fail-closed spirit).
    store_dir = tmp_path / "store"
    leftovers = (
        [p.name for p in store_dir.rglob("*") if p.is_file()]
        if store_dir.exists()
        else []
    )
    assert not any(
        name.endswith(".json") and "clock" not in name for name in leftovers
    ), f"faced bundle must not be committed; store leftovers: {leftovers}"


def _run_canonical_faceless(tmp_path: Path, tag: str):
    from live_checkpoint import run_checkpoint

    store = tmp_path / "store"
    keys = tmp_path / "keys"
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    empty_detector = MagicMock(spec=YuNetDetector)
    empty_detector.detect.return_value = []
    return run_checkpoint(
        device="0",
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
        experiment_id="exp-canon-red",
        session_id=f"canon-{tag}-{uuid4().hex[:8]}",
        camera_identity_probe=lambda: (["pinned-builtin-uid"], 1),
        expected_builtin_unique_id="pinned-builtin-uid",
        expected_builtin_shape="16x16",
    )


def test_canonical_path_faceless_run_completes(tmp_path: Path) -> None:
    """Negative control: faceless canonical run completes, stop flag False."""
    code, summary = _run_canonical_faceless(tmp_path, "faceless")
    live = summary["phases"]["live"]
    assert live.get("presence_stop") is False, (
        f"faceless canonical run must carry presence_stop=False "
        f"(got {live.get('presence_stop')!r})"
    )
    assert code != 4 or "presence stop" not in str(
        live.get("stderr_msg", "")
    ), "faceless run must not report a presence stop"
