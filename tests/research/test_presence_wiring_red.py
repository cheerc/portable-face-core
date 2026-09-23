"""Behavior-level RED for presence-mode entrypoint wiring (issue #82).

Defect: the runner's true-device path never passes ``presence_mode`` to
``cmd_live``, so every runner/CLI checkpoint runs with
``presence_mode="collection"`` — the presence guard protects zero actual
executions. Runner JSON summary also has no presence-stop discriminator.

RED1 (wiring): true-device runner call must reach cmd_live with
  presence_mode="checkpoint". Pre-fix: kwarg absent (-> collection).
RED2 (observable stop): faced run -> exit 4 AND summary live phase carries
  an explicit presence_stop=True field. Pre-fix: no such field.
RED3 (negative control): faceless run completes, presence_stop is False.
  Pre-fix: no such field (KeyError).
RED4 (collection unpolluted): fake-device runner call must reach cmd_live
  with presence_mode="collection". Pre-existing guard (passes pre-fix).
RED5 (fake path intact): existing fake-device tests all pass (full suite).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from facecore.live.capture import CaptureSource, FakeCapture, FramePacket  # noqa: E402
import numpy as np  # noqa: E402


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "wire-test-v1",
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


class _ScriptedCapture(CaptureSource):
    """Opens, delivers one probe frame, then replays scripted frames."""

    def __init__(self, n_frames: int = 7) -> None:
        self._frames = [
            FramePacket(
                sequence=seq,
                captured_ns=seq * 200_000_000,
                rgb=np.ascontiguousarray(
                    np.full((16, 16, 3), 120 + (seq % 40), dtype=np.uint8)
                ),
            )
            for seq in range(1, n_frames + 1)
        ]
        self._open = False

    def open(self, device_id: str) -> None:
        self._open = True

    def read(self) -> FramePacket | None:
        if not self._open or not self._frames:
            return None
        return self._frames.pop(0)

    def close(self) -> None:
        self._open = False

    @property
    def is_closed(self) -> bool:
        return not self._open


def _true_device_kwargs(tmp_path: Path) -> dict:
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    seen: dict = {}
    import live_checkpoint as runner

    real_cmd_live = runner.cmd_live

    def _spy(**kwargs):  # noqa: ANN001, ANN202
        seen.update(kwargs)
        return real_cmd_live(**kwargs)

    with patch.object(runner, "cmd_live", side_effect=_spy):
        run_checkpoint(
            device="0",
            profile_path=profile_path,
            record_consent=True,
            image_consent=True,
            capture_factory=lambda _d: _ScriptedCapture(),
            # Pre-existing test path: bypass the #89 identity gate via the
            # same injection seam the gate itself uses for hermetic tests.
            camera_identity_probe=lambda: (["pinned-builtin-uid"], 1),
            expected_builtin_unique_id="pinned-builtin-uid",
            expected_builtin_shape="16x16",
        )
    return seen


def test_red1_true_device_reaches_checkpoint_mode(tmp_path: Path) -> None:
    """True-device runner must forward presence_mode='checkpoint'."""
    seen = _true_device_kwargs(tmp_path)
    assert seen.get("presence_mode") == "checkpoint", (
        f"RED1: true-device call reached cmd_live with "
        f"presence_mode={seen.get('presence_mode')!r} "
        f"(expected 'checkpoint' — guard protects zero executions)"
    )


def test_red2_faced_run_is_observable_presence_stop(tmp_path: Path) -> None:
    """Faced run: exit 4 AND summary live.presence_stop is True."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    import live_checkpoint as runner

    def _faced_cmd_live(**kwargs):  # noqa: ANN001, ANN202
        print(
            "research live: presence stop: presence_face_detected: "
            "1 face(s) in no-participant checkpoint frame 3",
            file=sys.stderr,
        )
        return 4

    with patch.object(runner, "cmd_live", side_effect=_faced_cmd_live):
        code, summary = run_checkpoint(
            device="0",
            profile_path=profile_path,
            record_consent=True,
            image_consent=True,
            capture_factory=lambda _d: _ScriptedCapture(),
            camera_identity_probe=lambda: (["pinned-builtin-uid"], 1),
            expected_builtin_unique_id="pinned-builtin-uid",
            expected_builtin_shape="16x16",
        )
    assert code == 4, f"RED2: expected exit 4, got {code}"
    live = summary["phases"]["live"]
    assert live.get("presence_stop") is True, (
        f"RED2: summary live phase has no presence_stop=True "
        f"(keys: {sorted(live)} — indistinguishable from other exit-4 causes)"
    )


def test_red3_faceless_run_completes_without_presence_stop(
    tmp_path: Path,
) -> None:
    """Faceless run: exit 0, presence_stop explicitly False."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
    )
    assert code == 0, f"RED3: expected exit 0, got {code}: {summary}"
    live = summary["phases"]["live"]
    assert live.get("presence_stop") is False, (
        f"RED3: summary live phase must carry presence_stop=False "
        f"(keys: {sorted(live)})"
    )


def test_red4_fake_path_stays_collection(tmp_path: Path) -> None:
    """Fake-device runner must reach cmd_live with presence_mode='collection'."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    seen: dict = {}
    import live_checkpoint as runner

    real_cmd_live = runner.cmd_live

    def _spy(**kwargs):  # noqa: ANN001, ANN202
        seen.update(kwargs)
        return real_cmd_live(**kwargs)

    with patch.object(runner, "cmd_live", side_effect=_spy):
        code, _ = run_checkpoint(
            device="fake",
            profile_path=profile_path,
            record_consent=True,
            image_consent=True,
        )
    assert code == 0
    assert seen.get("presence_mode") == "collection", (
        f"RED4: fake-device call reached cmd_live with "
        f"presence_mode={seen.get('presence_mode')!r} "
        f"(checkpoint semantics must never leak into collection paths)"
    )


def test_red5_fake_capture_still_importable() -> None:
    """Fake path primitives intact (full fake-device suite runs in GREEN)."""
    assert FakeCapture is not None
