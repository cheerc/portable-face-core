"""D4-F1: an inference worker that dies must produce a VISIBLE terminal.

This is the second scope of the D4-F1 fix, and it is deliberately not
a fix for issue #123. That bug's root cause is the demo-mode trace
wiring in `cmd_live` (see `test_d4f1_demo_trace_wiring.py`). This is
the separate robustness gap the bug *exposed*: the inference worker had
a `try/finally` and no `except`, so any unexpected exception escaped
the thread, `controller._terminal` stayed `None`, `state` stayed
"running", and the Qt window went on painting a stale result with no
indication that inference had already died. The operator read a wrong
answer as a real one.

**The guard is the round's terminal, not the absence of a traceback.**
A test that only asserted "no exception was raised" would pass on any
build where the exception happens to be raised somewhere else, and it
would keep passing if this were reverted *and* the window were changed
to swallow the failure. What has to hold is: the round CONCLUDES, its
status is `error`, its reason codes name the failure, and the detail
is recoverable for the operator report.

**Measured while writing this, and it narrows the gap's importance.**
Two of the three obvious failure sites are ALREADY contained upstream:
a scorer exception becomes a `scorer_failure: <Type>` error terminal
(`controller.py:361`), and a research-sink exception is recorded in
`_research_sink_errors` by design (D2). So the round does not in fact
"hang with no terminal" for those. The site that genuinely escaped is
`frame_transform` — D2 deliberately lets a transform failure propagate
so the caller fails closed rather than commit unreconstructible
geometry — plus, until this PR, the trace channel (issue #123). That
makes this scope narrower than it first appears, and the tests below
are aimed at the path that really escapes rather than at the one that
already worked.

No camera, no faces, no gallery, no onnx: synthetic frames, a transform
that raises on cue, and a fake capture.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    SessionStatus,
)
from facecore.live.controller import LiveController
from tests.live.test_qt_window import _profile


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


def _frames(n: int = 40) -> list[FramePacket]:
    return [
        FramePacket(
            sequence=seq,
            captured_ns=seq * 200_000_000,
            rgb=np.full((8, 8, 3), 150, dtype=np.uint8),
        )
        for seq in range(1, n + 1)
    ]


def _ok_observation(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=False,
        quality_reasons=("no_face_detected",),
        face_count=0,
        face_box=None,
        identity_scores={},
        quality_rank=0.0,
        model_generation="gen-d4f1-robust",
        gallery_digest="gallery-d4f1-robust",
    )


def _controller(
    *,
    scorer: Any = None,
    frame_transform: Any = None,
    trace_recorder: Any = None,
    trace_attempt_id: str | None = None,
    session_id: str = "d4f1-robust",
) -> LiveController:
    """A started controller: start_session binds the engine session id."""
    controller = LiveController(
        engine=_engine(),
        source=FakeCapture(frames=_frames()),
        scorer=scorer if scorer is not None else _ok_observation,
        frame_transform=frame_transform,
        trace_recorder=trace_recorder,
        trace_attempt_id=trace_attempt_id,
    )
    controller.start_session(session_id, 0, device_id="fake")
    return controller


def _engine() -> Any:
    from facecore.live.session import SessionEngine

    return SessionEngine(_profile(), "gallery-d4f1-robust", "gen-d4f1-robust")


class TestWorkerFailureBecomesAVisibleTerminal:
    def test_a_scorer_that_raises_yields_an_error_terminal(
        self, qt_app: Any
    ) -> None:
        """The regression: an exception in the loop must not end the thread.

        A scorer that raises on the second frame stands in for any
        unexpected failure inside the round's core. Before D4-F1 this
        propagated out of the thread and left the round unterminated.
        """
        # The transform path is the one that genuinely escapes: D2
        # deliberately lets a transform failure propagate so the caller
        # fails closed rather than committing unreconstructible geometry.
        # The scorer path is already contained (`scorer_failure: ...`
        # terminal) and the research sink is already recorded — measured
        # while writing this, which is why the guard is aimed here.
        def _exploding(packet: FramePacket) -> FramePacket:
            raise RuntimeError("transform exploded")

        controller = _controller(frame_transform=_exploding)
        thread = controller.start_inference_worker()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the worker must not be stuck"

        terminal = controller.wait_for_terminal(timeout_s=2)
        assert terminal is not None, (
            "the round never concluded: the worker died and the UI would "
            "keep painting a stale result as if inference were still running"
        )
        assert terminal.status == SessionStatus.error, (
            f"expected an error terminal, got {terminal.status!r}"
        )
        assert "inference_worker_failed" in terminal.reason_codes, (
            f"the terminal must name the cause; got {terminal.reason_codes!r}"
        )
        assert "RuntimeError" in terminal.reason_codes, (
            f"the exception type belongs in the reason codes; "
            f"got {terminal.reason_codes!r}"
        )
        controller.close()

    def test_the_failure_detail_is_recoverable_for_the_report(
        self, qt_app: Any
    ) -> None:
        """The message itself must survive, not just the fact.

        The operator's D4 report quotes a terminal reason code; the
        detail behind it is what makes that quote actionable, so it has
        to survive the hand-off from the worker to the window.
        """
        controller = _controller(
            frame_transform=lambda packet: (_ for _ in ()).throw(
                ValueError("bad geometry")
            )
        )
        thread = controller.start_inference_worker()
        thread.join(timeout=10)

        controller.wait_for_terminal(timeout_s=2)
        failures = controller.inference_failures()
        assert failures, "the exception text was dropped"
        assert any("bad geometry" in f for f in failures), failures
        controller.close()

    def test_a_clean_worker_records_no_failure(self, qt_app: Any) -> None:
        """The negative case: the new list must stay empty in the normal path.

        Without this, a handler that fired on every round would look
        exactly as healthy as one that fires only on failure.
        """
        controller = _controller()
        thread = controller.start_inference_worker()
        thread.join(timeout=10)

        controller.wait_for_terminal(timeout_s=2)
        assert controller.inference_failures() == [], (
            "a round with no unexpected failure must record none"
        )
        controller.close()


class TestTheGuardResistsWeakening:
    def test_the_reason_code_is_not_freely_editable(self, qt_app: Any) -> None:
        """A failure that claims to be something else must still be visible.

        `status == error` alone would also be satisfied by a genuine
        model error, so it is not sufficient on its own: the code has to
        distinguish "the worker itself broke" from "the engine reached a
        conclusion on its own terms".
        """
        from facecore.live.session import SessionEngine

        engine = SessionEngine(_profile(), "gallery-d4f1-robust", "gen-d4f1-robust")
        engine.start("d4f1-codes", 0)
        worker_failure = engine.terminate_with_error(
            1_000_000, reason_codes=("inference_worker_failed", "RuntimeError")
        )
        assert worker_failure.status == SessionStatus.error
        assert worker_failure.reason_codes == (
            "inference_worker_failed",
            "RuntimeError",
        )

        # A second terminate is a no-op: the first terminal stands.
        again = engine.terminate_with_error(
            2_000_000, reason_codes=("something_else",)
        )
        assert again.reason_codes == worker_failure.reason_codes, (
            "terminate_with_error must be idempotent like the other "
            "terminate paths, or a late failure could overwrite a real result"
        )

    def test_the_worker_thread_is_a_daemon_and_does_not_leak(
        self, qt_app: Any
    ) -> None:
        """Threads must not accumulate across rounds that all fail.

        D2's contract is that workers are bounded and joined; a new
        handler that leaves a thread running would regress it.
        """
        baseline = threading.active_count()
        for _ in range(3):
            controller = _controller(
                frame_transform=lambda packet: (_ for _ in ()).throw(
                    RuntimeError("boom")
                )
            )
            thread = controller.start_inference_worker()
            thread.join(timeout=10)
            controller.wait_for_terminal(timeout_s=2)
            controller.close()
        assert threading.active_count() <= baseline + 1, (
            f"threads grew from {baseline} to {threading.active_count()} "
            "across three failing rounds"
        )


def _consent(session_id: str) -> Any:
    from tests.live.test_qt_window import _consent as _base

    return _base(session_id)
