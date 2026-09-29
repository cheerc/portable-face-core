"""D2b RED: continuous preview + 再辨識／停止相機 (operator decision -10).

Source of truth: decision d-20260929173733098323-10 (operator answered
「B」: the double-clicked App keeps the lens open after a round and
offers 再辨識 / 停止相機), superseding the 終局先關鏡頭 items of
d-20260924080117997753-5.

This supersedes R1 §2-4 for the Qt App path only. The research CLI
path is untouched.

The camera-handle assertions are the load-bearing ones here.
D2, D3a and D3b each shipped a false protection in the same
shape — an assertion aimed at a path the code never walks — so the
scopes below assert against `FakeCapture.open_calls` / `close_calls`
/ `is_closed`, which change exactly when the camera opens and
releases. A passing assertion on those cannot be satisfied by a value
nothing writes.

That property is scoped to the camera-handle assertions, not to
every test in this file. `TestUnlabeledRound` is the counter-example
that keeps it honest: its first guard originally read
`assert not hasattr(EvaluationLabel, "UNLABELED")`, which was
constant-false-and-therefore-meaningless (a dataclass carries no
label-kind attributes) and survived a mutation that genuinely
polluted the research enumeration. It now asserts the enumeration
itself. Read this as "the camera assertions are falsifiable", not as
a claim that this file is immune to that failure mode.

Synthetic frames and an offscreen Qt application only. No camera, no
real faces, no gallery, no photos, no real uids.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FrameObservation, FramePacket
from facecore.live.desktop import DesktopSession
from facecore.live.qt_window import QtResearchWindow
from facecore.live.session import SessionEngine
from facecore.research.cli import G3_DEMO_RESULTS_CSV_COLUMNS
from tests.live.test_qt_window import (
    _AdvancingClock,
    _attempt,
    _consent,
    _manifest,
    _recorder,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="D2b drives the real Qt window; it needs the research-ui extra",
)


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


def _profile() -> Any:
    """The shared Qt-test profile.

    It already carries required_support=1, so one frame ends a round
    immediately and ten consecutive rounds cost no wall clock.
    """
    from tests.live.test_qt_window import _profile as _base_profile

    return _base_profile()


def _face_frames(n: int = 60) -> list[FramePacket]:
    return [
        FramePacket(
            sequence=seq,
            captured_ns=seq * 200_000_000,
            rgb=np.full((16, 16, 3), 150, dtype=np.uint8),
        )
        for seq in range(1, n + 1)
    ]


def _matching_scorer(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(4.0, 4.0, 8.0, 8.0),
        identity_scores={"person-01": 0.50, "person-02": 0.30},
        quality_rank=0.5,
        model_generation="gen-d2b",
        gallery_digest="digest-d2b",
    )


class _CountingSource(FakeCapture):
    """FakeCapture counting opens/closes — the camera-state evidence.

    `is_closed` is the real handle state: the base class flips it in
    open() and close(). Counting the calls additionally proves the
    handle was not torn down and re-created, which is what "不重開鏡頭
    直接開下一輪" means at the source level.
    """

    def __init__(self, frames: list[FramePacket]) -> None:
        super().__init__(frames=frames)
        self.open_calls = 0
        self.read_calls = 0
        self.close_calls = 0

    def open(self, device_id: str) -> None:
        self.open_calls += 1
        super().open(device_id)

    def read(self) -> FramePacket | None:
        self.read_calls += 1
        return super().read()

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _factory(
    tmp_path: Path, source: FakeCapture, *, demo_csv: Path | None = None
) -> Any:
    """Round factory over one shared source, mirroring production."""
    recorder = _recorder(tmp_path)
    manifest = _manifest()
    counter = 0

    def make() -> tuple[DesktopSession, Any, str]:
        nonlocal counter
        counter += 1
        session_id = f"d2b-round-{counter}"
        attempt = _attempt(f"attempt-d2b-{counter}")
        consent = _consent(session_id)
        recorder.begin_attempt(manifest, attempt, consent)
        recorder.begin(session_id, consent)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-d2b", "gen-d2b"),
            source=source,
            scorer=_matching_scorer,
            session_id=session_id,
            release_source_on_terminal=False,
            label_recorder=recorder if demo_csv is None else None,
            label_attempt_id=attempt.attempt_id if demo_csv is None else None,
            demo_label_sink=demo_csv,
        )
        return desktop, consent, attempt.attempt_id

    make.recorder = recorder  # type: ignore[attr-defined]
    return make


def _window(
    qt_app: Any,
    tmp_path: Path,
    source: FakeCapture,
    **kwargs: Any,
) -> tuple[QtResearchWindow, Any]:
    factory = _factory(tmp_path, source, **kwargs)
    first_desktop, first_consent, first_attempt = factory()
    clock = _AdvancingClock()
    window = QtResearchWindow(
        first_desktop,
        consent=first_consent,
        recorder=None if kwargs.get("demo_csv") else factory.recorder,
        attempt_id=None if kwargs.get("demo_csv") else first_attempt,
        offscreen=True,
        clock_ns=clock,
        clock_advance=clock.advance,
        next_session=factory,
        **({"demo_results_csv": kwargs["demo_csv"]} if kwargs.get("demo_csv") else {}),
    )
    window.show()
    return window, factory


# ---------------------------------------------------------------------------
# Scope 1: a finished round keeps the lens open and the preview alive.
# ---------------------------------------------------------------------------


class TestResultKeepsTheCameraOpen:
    def test_result_does_not_release_the_camera(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """The D2b core: `_enter_result` must not release the handle."""
        source = _CountingSource(_face_frames())
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)

        assert window.mode == "result"
        assert source.is_closed is False, (
            "the lens was released at result — D2b's core requirement is "
            "that the camera stays open so the preview keeps running"
        )
        assert source.close_calls == 0, (
            f"close() called {source.close_calls}x before the operator "
            "decided anything"
        )
        window.close()

    def test_stop_camera_releases_and_keeps_the_pick(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Scope 3: 停止相機 shuts the lens and returns to Ready."""
        source = _CountingSource(_face_frames())
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"

        window.stop_camera_clicked()
        assert source.is_closed is True
        assert window.mode == "ready"
        assert window.preview_label.pixmap().isNull(), (
            "停止相機 left the last preview image on screen"
        )
        assert window.result_identity_label.text() == ""
        window.close()

    def test_closing_the_window_always_releases(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Scope 4: whatever the mode, closeEvent releases the handle."""
        source = _CountingSource(_face_frames())
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        assert source.is_closed is False, "precondition: lens is open at result"

        window.close()
        assert source.is_closed is True, (
            "closeEvent left the camera open — this is the one release "
            "path D2b must never break"
        )

    def test_closing_mid_round_releases(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """D2's close_while_reading class: close during a live round."""
        source = _CountingSource(_face_frames(200))
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()
        # Not driven to terminal: the round is still collecting.
        assert window.mode == "running"

        window.close()
        assert source.is_closed is True, (
            "closing the window mid-round left the camera open"
        )


# ---------------------------------------------------------------------------
# Scope 2: 再辨識 starts the next round without re-opening the lens.
# ---------------------------------------------------------------------------


class TestRecognizeAgain:
    def test_ten_consecutive_rounds_keep_one_camera_open(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Acceptance: ≥10 consecutive 再辨識, no worker/state residue.

        The camera must be opened exactly once and closed exactly once —
        once at the end. A re-open per round would show up as
        open_calls == rounds; a leaked handle would show as zero closes.
        """
        source = _CountingSource(_face_frames(400))
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()

        for _round in range(10):
            window.process_until_terminal(max_steps=200)
            assert window.mode == "result", "round did not reach result"
            assert source.is_closed is False, (
                f"round {_round}: the lens was released between rounds"
            )
            window.recognize_again_clicked()
            assert window.mode == "running", (
                f"round {_round}: 再辨識 did not start the next round"
            )

        assert source.open_calls == 1, (
            f"the camera was opened {source.open_calls} times for 10 rounds; "
            "再辨識 must not re-open the lens"
        )
        assert source.close_calls == 0, "the handle was released mid-loop"
        assert source.is_closed is False

        window.close()
        assert source.close_calls >= 1
        assert source.is_closed is True

    def test_previous_round_leaves_no_residue(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Scope 2: no previous result text and no stale countdown."""
        source = _CountingSource(_face_frames(400))
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        first_result_text = window.status_label.text()
        assert first_result_text

        window.recognize_again_clicked()
        assert window.status_label.text() != first_result_text, (
            "the previous round's result text survived into the next round"
        )
        assert window.result_identity_label.text() == "", (
            "the previous round's result panel survived into the next round"
        )
        assert window.frames_label.text() == "", (
            "the previous round's frame counts survived into the next round"
        )
        window.close()

    def test_no_worker_accumulates_across_rounds(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Acceptance: no worker residue after ten rounds."""
        source = _CountingSource(_face_frames(400))
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()

        import threading

        baseline = threading.active_count()
        for _round in range(10):
            window.process_until_terminal(max_steps=200)
            window.recognize_again_clicked()
        window.process_until_terminal(max_steps=200)

        assert threading.active_count() <= baseline + 1, (
            f"threads grew from {baseline} to {threading.active_count()} "
            "across ten rounds"
        )
        window.close()


# ---------------------------------------------------------------------------
# Scope 1 (continued): result UI and preview coexist with D3a panels.
# ---------------------------------------------------------------------------


class TestResultAndPreviewCoexist:
    def test_d3a_panels_survive_at_result_with_camera_open(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """D3a's five panels must still be populated while the lens is open."""
        source = _CountingSource(_face_frames())
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)

        assert window.mode == "result"
        # D1's 3-frame rule is frozen, so one matching frame yields a
        # thin-evidence candidate, not a match. Either way the D3a panels
        # must be populated while the lens is open.
        assert "person-01" in window.result_identity_label.text()
        assert "Top 1" in window.scores_label.text()
        assert "有效幀" in window.frames_label.text()
        window.close()

    def test_recognize_again_clears_the_result_panels(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Scope 1: panels clear when the next round starts, not at result."""
        source = _CountingSource(_face_frames(400))
        window, _factory_obj = _window(qt_app, tmp_path, source)
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        assert window.result_identity_label.text() != ""

        window.recognize_again_clicked()
        assert window.result_identity_label.text() == ""
        assert window.scores_label.text() == ""
        assert window.frames_label.text() == ""
        assert window.failure_reason_label.text() == ""
        window.close()


# ---------------------------------------------------------------------------
# Commander point (二): an unlabeled round is recorded, never dropped.
# ---------------------------------------------------------------------------


class TestUnlabeledRound:
    def test_demo_mode_records_the_skipped_round_as_unlabeled(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Skipping the verdict still writes a demo row, marked 未標註."""
        store = tmp_path / "demo-store"
        store.mkdir()
        demo_csv = store / "demo-results.csv"
        source = _CountingSource(_face_frames(400))
        window, _factory_obj = _window(
            qt_app, tmp_path, source, demo_csv=demo_csv
        )
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)

        # Round 1 reaches result; the operator presses NEITHER verdict and
        # goes straight to 再辨識.
        window.enter_ready()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        window.recognize_again_clicked()

        # Round 2 runs on the same open camera and IS labeled.
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        window.press_correct()

        assert demo_csv.is_file(), (
            "the skipped round was silently dropped instead of recorded"
        )
        rows = list(csv.reader(demo_csv.read_text(encoding="utf-8").splitlines()))
        assert rows[0] == list(G3_DEMO_RESULTS_CSV_COLUMNS)
        assert len(rows) == 3, f"expected header + 2 rounds, got {len(rows)}"
        skipped = dict(zip(rows[0], rows[1], strict=True))
        labeled = dict(zip(rows[0], rows[2], strict=True))
        assert skipped["label_kind"] == "unlabeled", (
            "the skipped round must be marked as unlabeled, not silently "
            "reusing an existing verdict kind"
        )
        assert skipped["label_identity"] == ""
        # The labeled round is unaffected by the new demo-only value. Its
        # kind is 「unenrolled」 rather than 「enrolled」 because the round
        # is a thin-evidence candidate, so display_identity() is None and
        # 正確 means "not enrolled" — D1's contract, untouched here.
        assert labeled["label_kind"] == "unenrolled"
        assert labeled["label_identity"] == ""
        window.close()

    def test_unlabeled_never_enters_the_research_label_kinds(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Decision -11: the research label_kind enumeration is unchanged.

        S1 (reviewer-found): this used to read
        `assert not hasattr(EvaluationLabel, "UNLABELED")`, which was
        the fourth same-shape false protection. `EvaluationLabel` is a
        dataclass whose only public attributes are `from_dict` /
        `schema_version` / `to_dict` — it carries no label-kind members
        at all, so the hasattr was constant-false no matter what the
        code did. Adding "unlabeled" to the real enumeration left this
        test green.

        The authority is the module-level `LABEL_KINDS`, which
        `EvaluationLabel.__post_init__` validates every instance
        against. Assert that directly, and pin the whole set so a
        future addition has to be deliberate here rather than silent.
        """
        from facecore.research.experiment import LABEL_KINDS

        assert "unlabeled" not in LABEL_KINDS, (
            "the demo-only label value leaked into the research "
            "enumeration; report.py/analysis.py would start counting it"
        )
        assert LABEL_KINDS == frozenset({"enrolled", "unenrolled", "uncertain"}), (
            f"the research label vocabulary changed: {sorted(LABEL_KINDS)}"
        )
