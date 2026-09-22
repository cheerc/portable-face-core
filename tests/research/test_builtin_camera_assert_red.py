"""Behavior-level RED: built-in camera assertion before capture (#89).

Commander ruling path (c): NO pyobjc, NO auto-select, NO resolve_local_camera
change, --device local stays unauthorized. Assertion-only surface:
operator passes --device N + the pinned built-in uniqueID; the system
refuses LOUD unless enumeration evidence says N is the built-in camera.

Binding conditions (all refuse, never warn):
1. pinned uniqueID absent from current enumeration -> refuse;
   predicted count != OpenCV-openable count -> refuse;
   enumeration itself fails -> refuse.
2. enumeration-INDEPENDENT cross-check after the assert (probe-frame
   shape vs expected built-in shape) — mismatch -> refuse.
3. NO hardcoded uniqueID in product code — pin arrives via CLI param;
   runbook may record this machine's value as example only.

RED items (pre-fix outcomes; post-fix assertions):
A1. true-device run WITHOUT --expected-builtin-unique-id -> exit 2,
    explicit message (never a silent pass-through to capture).
A2. pinned ID absent from enumeration -> refuse (nonzero exit).
A3. predicted-device count != openable count -> refuse.
A4. probe shape != expected built-in shape -> refuse (nonzero exit).
A5. all evidence agrees -> run proceeds (no refusal from the assert).
A6. int(device) coercions ('local' string) -> explicit refusal message,
    never a ValueError traceback (teardown:33, checkpoint:145,185).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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


BUILTIN_UID = "EAB7A68F-EC2B-4487-AADF-D8A91C1CB782"
IPHONE_UID = "D9B9EBF1-CD2F-4316-A481-7AD800000001"


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "builtin-assert-v1",
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


def _sp_json(uids: list[str]) -> str:
    return json.dumps(
        {
            "SPCameraDataType": [
                {
                    "_name": f"cam-{u[:4]}",
                    "spcamera_model-id": f"model-{u[:4]}",
                    "spcamera_unique-id": u,
                }
                for u in uids
            ]
        }
    )


class _FakeRunner:
    """CompletedProcess double for subprocess.run(system_profiler ...)."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.returncode = 0


def _run_true(tmp_path: Path, tag: str, **kw):
    from live_checkpoint import run_checkpoint

    store = tmp_path / f"store-{tag}"
    keys = tmp_path / f"keys-{tag}"
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    p1 = tmp_path / "enroll_01.png"
    p2 = tmp_path / "enroll_02.png"
    from PIL import Image

    for p, seed in ((p1, 1), (p2, 2)):
        arr = np.full((200, 200, 3), 120, dtype=np.uint8)
        arr[::2, ::2] = 160
        arr[20:80, 20:80, 0] = (150 + seed) % 256
        Image.fromarray(arr, mode="RGB").save(p)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
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
        )
    )
    empty_detector = MagicMock(spec=YuNetDetector)
    empty_detector.detect.return_value = []

    class _Emb:
        model_version = "sface_2021dec"

        def embed(self, crop):  # noqa: ANN001, ANN202
            arr = np.array((1.0, 0.0, 0.0), dtype=np.float32)
            return arr / float(np.linalg.norm(arr)), "sface_2021dec"

    base = dict(
        device="1",
        profile_path=_write_profile(tmp_path),
        record_consent=True,
        image_consent=True,
        store=store,
        key_dir=keys,
        models=models,
        corpus=corpus,
        capture_factory=lambda _d: FakeCapture(frames=_frames()),
        detector_factory=lambda _m: empty_detector,
        embedder_factory=lambda _m: _Emb(),
        experiment_id="exp-builtin-assert",
        session_id=f"chk-{tag}",
    )
    base.update(kw)
    with patch(
        "live_checkpoint.cmd_live", wraps=__import__(
            "facecore.research.cli", fromlist=["cmd_live"]
        ).cmd_live,
    ):
        return run_checkpoint(**base)


def _sp_run(uids: list[str]):
    """Patch target for subprocess.run returning system_profiler JSON."""
    return patch(
        "subprocess.run", return_value=_FakeRunner(_sp_json(uids))
    )


def test_a1_missing_pin_refuses(tmp_path: Path) -> None:
    """True-device run without the pinned ID must exit 2 explicitly."""
    code, summary = _run_true(tmp_path, "a1")
    assert code == 2, f"A1: expected exit 2 without pin, got {code}"
    text = json.dumps(summary)
    assert "expected-builtin" in text.lower() or "builtin-unique" in text.lower(), (
        f"A1: refusal must come from the assert layer naming the missing "
        f"built-in identity parameter: {text[:300]}"
    )


def test_a2_pin_absent_from_enumeration_refuses(tmp_path: Path) -> None:
    """Pinned ID not in current enumeration -> refuse."""
    with _sp_run([IPHONE_UID]):
        code, _ = _run_true(
            tmp_path, "a2", expected_builtin_unique_id=BUILTIN_UID
        )
    assert code != 0, "A2: pinned ID absent must refuse"


def test_a3_count_mismatch_refuses(tmp_path: Path) -> None:
    """Predicted count != openable count -> refuse."""
    # Enumeration sees 2 devices; only index 1 openable via factory.
    with _sp_run([IPHONE_UID, BUILTIN_UID]):
        code, _ = _run_true(
            tmp_path,
            "a3",
            expected_builtin_unique_id=BUILTIN_UID,
            capture_factory=lambda _d: FakeCapture(frames=_frames()),
        )
    # The factory hides openability from the assert layer; the assert
    # must still verify count via its own OpenCV probe. Pre-fix there is
    # no such check, so this documents the missing gate.
    assert code != 0, "A3: count mismatch must refuse"


def test_a4_shape_mismatch_refuses(tmp_path: Path) -> None:
    """Probe shape != expected built-in shape -> refuse."""
    with _sp_run([IPHONE_UID, BUILTIN_UID]):
        code, _ = _run_true(
            tmp_path,
            "a4",
            expected_builtin_unique_id=BUILTIN_UID,
            expected_builtin_shape="720x1280",
            capture_factory=lambda _d: FakeCapture(frames=_frames(16, 16)),
        )
    assert code != 0, "A4: shape mismatch must refuse"


def test_a5_all_agree_proceeds(tmp_path: Path) -> None:
    """Full agreement -> no refusal from the assert layer."""
    with _sp_run([IPHONE_UID, BUILTIN_UID]):
        code, summary = _run_true(
            tmp_path,
            "a5",
            expected_builtin_unique_id=BUILTIN_UID,
            expected_builtin_shape="16x16",
            capture_factory=lambda _d: FakeCapture(frames=_frames(16, 16)),
        )
    live = summary["phases"]["live"]
    assert "builtin" not in json.dumps(live.get("stderr_msg", "")).lower(), (
        f"A5: assert must not refuse when evidence agrees: {live}"
    )


def test_a6_local_string_refused_explicitly(tmp_path: Path) -> None:
    """'local' device string -> explicit refusal, never ValueError traceback."""
    from live_checkpoint import run_checkpoint

    code, summary = run_checkpoint(
        device="local",
        profile_path=_write_profile(tmp_path),
        record_consent=True,
        image_consent=True,
    )
    text = json.dumps(summary)
    assert code != 0, "A6: 'local' must not silently proceed"
    assert "traceback" not in text.lower(), "A6: never a traceback"
    assert "unauthorized" in text.lower() or "not authorized" in text.lower(), (
        f"A6: 'local' must be refused EXPLICITLY as unauthorized, "
        f"not a raw int() ValueError: {text[:300]}"
    )
