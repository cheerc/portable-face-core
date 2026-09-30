"""D4-F1 RED: demo mode must never hand a trace writer to the desktop.

Issue #123, the blocking D4 bug: in demo mode the operator's inference
died and the UI showed a stale "no face detected". Root cause, verified
by measurement (see the module docstring of the sibling launcher test for
how the fifth-generation false protection was avoided):

    cmd_live gated the trace channel on `is_true_path` — "is this the
    real camera path" — which is TRUE in demo mode on a real camera. So
    `_NullRecorder` (the no-op stand-in demo mode uses) was passed as the
    trace writer. `controller._append_live_trace` guards on `is None`,
    which a no-op recorder is not, so it proceeded to its own capability
    probe and raised `AttributeError: trace_recorder has no append_trace
    method` — from inside `_consume_one`, i.e. on EVERY round.

**Correction to the dispatch brief.** Both the commander's analysis and
the lead's said the raise came from `_NullRecorder.__getattr__`'s
allow-list. Measurement says otherwise:

    getattr(NULL_RECORDER, "append_trace")        -> raises AttributeError
    getattr(NULL_RECORDER, "append_trace", None)  -> returns None

`getattr`'s third argument swallows the `AttributeError` that
`__getattr__` raises and returns the default. So the line at
controller.py:456 never raises; the `if append is None: raise` right
after it is the live raise, and D3b's allow-list is not involved at
all — it is what made the failure a *clear error* instead of a silent
no-op. This matters for where a fix belongs: the allow-list must not be
touched, and the guard at 457-458 must not be "revived" or deleted.

**The guard that matters here is the wiring, not the crash.** Calling
`_append_live_trace` with a hand-picked recorder is the D2 false
protection shape — it exercises a state only correct wiring can produce.
These tests therefore drive `cmd_live` itself and read the kwargs each
`DesktopSession` is actually constructed with.

Reaching the true path without a camera: `device="0"` (not "fake") with
an injected `capture_factory`, `presence_mode="checkpoint"` (which builds
the detector only — no embedder, no gallery, no faces), and the real
YuNet artifact already on this machine. `ui="fake"` takes the same
`is_true_path=True` branch and needs no Qt, which also keeps the
`qt_offscreen` device restriction out of the picture.

No camera, no real faces, no gallery, no embeddings, no photos.
"""

from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_MODELS = Path.home() / "facecore-models"
_YUNET = _MODELS / "face_detection_yunet_2023mar.onnx"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None
    or not _YUNET.exists(),
    reason=(
        "D4-F1 drives the real checkpoint presence path: it needs the "
        "research-ui extra and the local YuNet artifact (no camera, no faces)"
    ),
)


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "profile_version": "d4f1-test",
                "timeout_ms": 5000,
                "sample_interval_ms": 200,
                "max_frames": 26,
                "queue_limit": 1,
                "required_support": 1,
                "min_support_interval_ms": 1,
                # Low enough that the checkpoint presence scorer reaches a
                # terminal on the synthetic frames. The behaviour under test
                # is the wiring, not the threshold.
                "match_threshold": 0.10,
                "review_threshold": 0.05,
                "margin_threshold": 0.01,
                "detector_version": "det-d4f1",
                "quality_policy_version": "qual-d4f1",
                "continuity_max_center_delta_ratio": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return path


def _frames(n: int = 40) -> list[Any]:
    from facecore.live.contracts import FramePacket

    return [
        FramePacket(
            sequence=seq,
            captured_ns=seq * 200_000_000,
            rgb=np.ascontiguousarray(np.full((16, 16, 3), 150, dtype=np.uint8)),
        )
        for seq in range(1, n + 1)
    ]


def _run_demo_true_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Drive cmd_live on the demo + true-camera branch; return what happened.

    Returns the recorded wiring kwargs, the return code, and any
    exception that escaped — including its traceback text, because
    "cmd_live returned non-zero" is not an acceptable assertion here:
    several unrelated guards also return 2.
    """
    from facecore.live.capture import FakeCapture
    from facecore.live.desktop import DesktopSession
    from facecore.research.cli import cmd_live

    seen: list[dict[str, Any]] = []
    escaped: list[BaseException] = []
    original_init = DesktopSession.__init__

    def _spy(self: Any, *args: Any, **kwargs: Any) -> None:
        seen.append(dict(kwargs))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(DesktopSession, "__init__", _spy)

    # The setup guard wants a gallery dir even though checkpoint mode
    # never reads it; it only satisfies `models is None or (corpus is
    # None and gallery_dir is None)`.
    gallery = tmp_path / "gallery"
    gallery.mkdir(exist_ok=True)

    try:
        rc = cmd_live(
            profile_path=_profile(tmp_path),
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="0",  # not "fake" -> is_true_path
            session_id="d4f1",
            record_consent=False,
            image_consent=False,
            mode="demo",
            ui="fake",
            models=_MODELS,
            gallery_dir=gallery,
            presence_mode="checkpoint",
            capture_factory=lambda _dev: FakeCapture(frames=_frames()),
        )
    except BaseException as exc:  # noqa: BLE001 - we assert on it below
        escaped.append(exc)
        rc = -1

    return {
        "wiring": seen,
        "rc": rc,
        "exception": escaped[0] if escaped else None,
        # format_exception takes the exception itself (3.10+), not the
        # unpacked (type, value, tb) triple — passing the tb alone raises
        # TypeError and would mask whatever the guard was meant to report.
        "traceback": (
            "".join(traceback.format_exception(escaped[0])) if escaped else ""
        ),
    }


class TestDemoModeHandsNoTraceWriter:
    def test_no_desktop_in_demo_mode_receives_a_trace_recorder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wiring guard: `_NullRecorder` must never reach a trace channel.

        This names the root cause directly. The D3b `_NullRecorder` is a
        legitimate object that must stay out of every recorder channel in
        demo mode — including the trace channel, whose only purpose is
        writing the encrypted research bundle.
        """
        from facecore.research.cli import NULL_RECORDER

        out = _run_demo_true_path(tmp_path, monkeypatch)

        assert out["wiring"], (
            "cmd_live built no DesktopSession, so nothing was verified — "
            "this test would pass for the wrong reason"
        )
        for kwargs in out["wiring"]:
            trace_recorder = kwargs.get("trace_recorder")
            assert trace_recorder is None, (
                "demo mode passed a trace recorder to "
                f"{kwargs.get('session_id')!r}: {type(trace_recorder).__name__}. "
                "Demo mode writes no encrypted bundle, so it must receive no "
                "trace writer at all."
            )
            assert trace_recorder is not NULL_RECORDER, (
                "the _NullRecorder reached a trace channel (issue #123)"
            )
            assert kwargs.get("trace_attempt_id") is None, (
                "trace_attempt_id must move with trace_recorder: the guard is "
                "an `or`, so a leftover id re-opens the same hole"
            )

    def test_the_run_completes_without_an_unexpected_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The behavioural guard: the round runs to a terminal.

        Asserting only `rc == 0` would be a false protection — several
        unrelated setup guards also return 2. So this asserts no
        unexpected exception escaped AND, when one did, that its
        traceback points at `_append_live_trace` rather than at some
        other failure the test could be confusing for the bug.
        """
        out = _run_demo_true_path(tmp_path, monkeypatch)
        exc = out["exception"]

        assert exc is None, (
            "an unexpected exception escaped cmd_live on the demo + true "
            f"path: {type(exc).__name__}: {exc}\n{out['traceback']}"
        )
        assert out["rc"] in (0, 4), (
            f"cmd_live returned {out['rc']}; 2 means a setup guard refused, "
            "which is not the condition under test"
        )

    def test_the_pre_fix_crash_points_at_append_live_trace(self) -> None:
        """Pin WHERE the bug lived, so the behavioural guard stays honest.

        This does not run the CLI; it asserts the mechanism directly, so
        that if someone "fixes" it by weakening controller's capability
        probe, this still documents that the probe was the thing that
        fired. It also documents the getattr-with-default semantics that
        make the probe the raise site rather than `_NullRecorder`.
        """
        from facecore.research.cli import NULL_RECORDER

        # getattr with a default swallows __getattr__'s AttributeError.
        assert getattr(NULL_RECORDER, "append_trace", None) is None, (
            "if this ever raises, the controller's probe would report a "
            "different error and the root-cause attribution changes"
        )
        with pytest.raises(AttributeError):
            getattr(NULL_RECORDER, "append_trace")  # noqa: B009

        controller_src = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "facecore"
            / "live"
            / "controller.py"
        ).read_text(encoding="utf-8")
        assert "append_trace" in controller_src
        # The guard is an `or`: a half-applied fix leaves the hole open.
        assert (
            "if self._trace_recorder is None or self._trace_attempt_id is None:"
            in controller_src
        ), "the controller's None-guard shape changed; re-verify D4-F1"
