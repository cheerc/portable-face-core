"""Behavior-level tests for the one-command live checkpoint runner (Issue #82).

RED evidence: these tests fail until scripts/live_checkpoint.py implements
``run_checkpoint``. Each negative path must map to a stable nonzero exit
code and a JSON-serializable summary — never a false PASS.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = str(REPO / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from facecore.live.capture import CaptureSource, FakeCapture, FramePacket  # noqa: E402


def _write_profile(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "chk-test-v1",
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


class _FailOpenCapture(CaptureSource):
    """Capture source that always fails to open."""

    def open(self, device_id: str) -> None:
        raise RuntimeError(f"cannot open device {device_id!r}")

    def read(self) -> FramePacket | None:
        return None

    def close(self) -> None:
        pass

    @property
    def is_closed(self) -> bool:
        return True


class _FailReadCapture(CaptureSource):
    """Capture source that opens but delivers no frames."""

    def open(self, device_id: str) -> None:
        pass

    def read(self) -> FramePacket | None:
        return None

    def close(self) -> None:
        pass

    @property
    def is_closed(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Import gate — runner must exist
# ---------------------------------------------------------------------------


def test_runner_module_importable() -> None:
    import live_checkpoint  # noqa: F401


# ---------------------------------------------------------------------------
# Happy path — fake device, full pipeline
# ---------------------------------------------------------------------------


def test_happy_path_fake_device(tmp_path: Path) -> None:
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
    )
    assert code == 0, f"expected exit 0, got {code}: {summary}"
    assert summary["verdict"] == "PASS"
    assert summary["exit_code"] == 0
    assert "phases" in summary
    assert "live" in summary["phases"]
    assert "store_lifecycle" in summary["phases"]
    lifecycle = summary["phases"]["store_lifecycle"]
    assert lifecycle["delete"]["pass"] is True
    assert lifecycle["restart_unreadable"]["pass"] is True
    assert lifecycle["canary"]["pass"] is True
    json.dumps(summary)


# ---------------------------------------------------------------------------
# Negative paths — each maps stable nonzero + JSON
# ---------------------------------------------------------------------------


def test_missing_consent_exits_2(tmp_path: Path) -> None:
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=False,
        image_consent=False,
    )
    assert code == 2
    assert summary["verdict"] == "FAIL"
    json.dumps(summary)


def test_device_mismatch_exits_nonzero_with_json(tmp_path: Path) -> None:
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="99",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        capture_factory=lambda dev: _FailOpenCapture(),
    )
    assert code != 0
    assert summary["verdict"] == "FAIL"
    json.dumps(summary)


def test_preflight_read_false_exits_nonzero(tmp_path: Path) -> None:
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)

    def _fail_preflight() -> dict[str, object]:
        return {"pass": False, "reason": "device probe failed"}

    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        preflight=True,
        preflight_fn=_fail_preflight,
    )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    assert summary["phases"]["preflight"]["pass"] is False
    json.dumps(summary)


def test_deadline_incomplete_exits_nonzero(tmp_path: Path) -> None:
    """Source exhausts before deadline → runner reports incomplete window."""
    from live_checkpoint import run_checkpoint
    import numpy as np

    single_frame = FakeCapture(
        frames=[
            FramePacket(
                sequence=1,
                captured_ns=0,
                rgb=np.full((16, 16, 3), 128, dtype=np.uint8),
            )
        ]
    )
    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        capture_factory=lambda dev: single_frame,
    )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    live_phase = summary["phases"]["live"]
    assert live_phase.get("window") in (
        "fixed-window-incomplete",
        "early-stop",
    )
    json.dumps(summary)


def test_worker_join_failure_exits_nonzero(tmp_path: Path) -> None:
    """Session raises mid-pump → stable nonzero."""
    from live_checkpoint import run_checkpoint

    class _ExplodingCapture(CaptureSource):
        _opened = False
        _closed = False

        def open(self, device_id: str) -> None:
            self._opened = True

        def read(self) -> FramePacket | None:
            if self._opened:
                raise RuntimeError("simulated frame pump failure")
            return None

        def close(self) -> None:
            self._closed = True

        @property
        def is_closed(self) -> bool:
            return self._closed

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        capture_factory=lambda dev: _ExplodingCapture(),
    )
    assert code != 0
    assert summary["verdict"] == "FAIL"
    json.dumps(summary)


def test_delete_failure_exits_nonzero(tmp_path: Path) -> None:
    """Store delete failure → nonzero + JSON."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        delete_fn=lambda recorder, sid: (_ for _ in ()).throw(
            OSError("simulated delete failure")
        ),
    )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    assert summary["phases"]["store_lifecycle"]["delete"]["pass"] is False
    json.dumps(summary)


def test_canary_failure_exits_nonzero(tmp_path: Path) -> None:
    """Canary write/read failure → nonzero + JSON."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        canary_fn=lambda recorder, store, key_dir: {
            "pass": False,
            "error": "simulated canary failure",
        },
    )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    assert summary["phases"]["store_lifecycle"]["canary"]["pass"] is False
    json.dumps(summary)


def test_camera_reopen_failure_exits_nonzero(tmp_path: Path) -> None:
    """Camera reopen failure → nonzero + JSON."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        reopen_fn=lambda dev: {
            "pass": False,
            "reopen": False,
            "error": "simulated device reopen failure",
        },
    )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    assert summary["phases"]["camera"]["pass"] is False
    json.dumps(summary)


def test_restart_unreadable_failure_exits_nonzero(tmp_path: Path) -> None:
    """Deleted session remains readable → nonzero + JSON."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
        restart_unreadable_fn=lambda recorder, sid: {
            "pass": False,
            "error": "simulated leak: session was readable after restart",
        },
    )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    assert summary["phases"]["store_lifecycle"]["restart_unreadable"]["pass"] is False
    json.dumps(summary)


def test_staged_errors_surfaced_exits_4(tmp_path: Path) -> None:
    """Staged errors during live session must be surfaced and exit 4."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)

    def _mock_cmd_live(**kwargs: object) -> int:
        sys.stderr.write("research live: staging failed: 1:DiskError;2:Corrupt\n")
        return 4

    with patch("live_checkpoint.cmd_live", side_effect=_mock_cmd_live):
        code, summary = run_checkpoint(
            device="fake",
            profile_path=profile_path,
            record_consent=True,
            image_consent=True,
        )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    live_phase = summary["phases"]["live"]
    assert live_phase["pass"] is False
    assert "staged_errors" in live_phase
    assert "1:DiskError" in live_phase["staged_errors"]
    assert "2:Corrupt" in live_phase["staged_errors"]
    json.dumps(summary)


def test_cli_main_invocation(tmp_path: Path) -> None:
    """CLI main entry point emits single JSON line and exits 0 on fake."""
    from live_checkpoint import main

    profile_path = _write_profile(tmp_path)
    code = main(
        [
            "--device",
            "fake",
            "--profile",
            str(profile_path),
            "--record-consent",
            "--image-consent",
        ]
    )
    assert code == 0


# ---------------------------------------------------------------------------
# Cleanup + safety invariants
# ---------------------------------------------------------------------------


def test_cleanup_runs_on_live_failure(tmp_path: Path) -> None:
    """Cleanup must execute even when cmd_live fails."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    store = tmp_path / "store"
    key_dir = tmp_path / "keys"
    store.mkdir()
    key_dir.mkdir()
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        store=store,
        key_dir=key_dir,
        record_consent=False,
        image_consent=False,
    )
    assert code == 2
    assert summary["phases"].get("teardown", {}).get("cleanup_attempted") is True
    json.dumps(summary)


def test_no_pii_in_summary(tmp_path: Path) -> None:
    """Summary must contain no pixels, embeddings, full paths, or PII."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
    )
    text = json.dumps(summary)
    assert "/Users/" not in text
    assert "/home/" not in text
    assert "embedding" not in text.lower()
    assert "pixel" not in text.lower()


def test_exit_codes_align_with_cmd_live(tmp_path: Path) -> None:
    """Exit codes: 0 ok / 2 usage-consent-profile / 4 runtime."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    code_ok, _ = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
    )
    assert code_ok == 0

    code_usage, _ = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=False,
        image_consent=False,
    )
    assert code_usage == 2


def test_summary_has_manual_checkpoints(tmp_path: Path) -> None:
    """Summary must list unverified manual checkpoints."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    _, summary = run_checkpoint(
        device="fake",
        profile_path=profile_path,
        record_consent=True,
        image_consent=True,
    )
    assert "manual_checkpoints" in summary
    assert len(summary["manual_checkpoints"]) > 0
    assert any(
        "camera" in c.lower() or "permission" in c.lower()
        for c in summary["manual_checkpoints"]
    )


def test_runner_trace_count_mismatch_fails_closed(tmp_path: Path) -> None:
    """Trace entry count != required_min must fail live_passed."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    # Patch read_trace to return an incomplete trace (e.g. 25 entries instead of 26)
    mock_trace = MagicMock()
    mock_trace.entries = tuple(range(25))

    with patch(
        "facecore.research.recorder.ResearchRecorder.read_trace",
        return_value=mock_trace,
    ):
        code, summary = run_checkpoint(
            device="fake",
            profile_path=profile_path,
            record_consent=True,
            image_consent=True,
        )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    live_phase = summary["phases"]["live"]
    assert live_phase["pass"] is False
    assert live_phase.get("trace_entries") == 25


def test_runner_trace_error_surfaces_and_fails_closed(tmp_path: Path) -> None:
    """Trace read error must populate trace_error key and fail live_passed."""
    from live_checkpoint import run_checkpoint

    profile_path = _write_profile(tmp_path)
    with patch(
        "facecore.research.recorder.ResearchRecorder.read_trace",
        side_effect=ValueError("corrupt trace sidecar"),
    ):
        code, summary = run_checkpoint(
            device="fake",
            profile_path=profile_path,
            record_consent=True,
            image_consent=True,
        )
    assert code == 4
    assert summary["verdict"] == "FAIL"
    live_phase = summary["phases"]["live"]
    assert live_phase["pass"] is False
    assert "trace_error" in live_phase
    assert "corrupt trace sidecar" in live_phase["trace_error"]
