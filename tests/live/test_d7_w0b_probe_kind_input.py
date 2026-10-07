"""D7-A W0-b: the operator verdict must reach the demo CSV row.

`operator_verdict` has existed in the demo CSV header since #158, but no
writer ever passed a non-empty value: both Qt writers omit the kwarg, so
every row carries the default empty string. The runbook-adjacent claim
that 「空 verdict = 未進入標註流程」 therefore never meets data.

What these tests defend is the VALUE PATH, not the widget set: pressing
the operator's ✓／✗／再次辨識 buttons must land `correct`／`incorrect`／
`skipped` in the written row. A test that only checks 「a button exists」
or calls `g3_demo_round_row` directly (as the old `:283` test did) passes
on a build where the column stays permanently empty.

The distinction that matters is the same one this file always defended:
「每輪可獨立設定」 is a claim about time. It is kept here in button form
— setting the ground truth via the new annotation state rather than the
removed combos, then requiring different rounds to produce different
cells.

History: the seven tests below that drove `win.probe_kind_combo` /
`win.presenting_identity_combo` were removed when P3 removed the
per-round dropdowns (spec §4.1, commander D2 single-commit). The five
helpers are reused verbatim — they are real wiring (FakeCapture, real
DesktopSession, real _QtResearchWindow), the most valuable part of this
file.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest


def _has_qt() -> bool:
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _has_qt(), reason="needs the research-ui extra (PySide6)"
)


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _gallery(identities: list[str]) -> Any:
    """A gallery stub whose keys are the identities, nothing else.

    The real `ResearchGallery` is a frozen dataclass that validates every
    embedding, so the test uses a minimal duck-typed stand-in. Only the
    `embeddings` mapping is read by the code under test.
    """
    import numpy as np

    class _Gallery:
        def __init__(self) -> None:
            self.embeddings = {
                name: np.zeros(128, dtype="float32") for name in identities
            }

    return _Gallery()


def _profile():
    from facecore.live.contracts import ResearchProfile

    return ResearchProfile(
        schema_version="v1",
        profile_version="w0b-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="yunet-test",
        quality_policy_version="q-test-v1",
        continuity_max_center_delta_ratio=0.50,
    )


def _desktop(session_id: str = "w0b-1", *, matched: bool = False):
    """A DesktopSession wired the way `tests/live/test_desktop.py` does.

    The window reads annotation state at WRITE time and never touches the
    engine for it, so the desktop only has to be constructible — but it
    must be real, because a stub would let the window's own constructor
    fail for an unrelated reason and mask what these tests are about.

    `matched=True` uses a scoring scorer so the round lands `matched`
    (the verdict path works on any terminal, but a matched round is the
    operator's main case); the default all-reject scorer lands
    `invalid_input`.
    """
    import numpy as np

    from facecore.live.capture import FakeCapture
    from facecore.live.desktop import DesktopSession
    from facecore.live.frame_pipeline import FrameObservation, FramePacket
    from facecore.live.session import SessionEngine
    from facecore.research.records import ConsentRecord

    def _packet(seq: int):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[0, 0, 0] = seq % 256
        return FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=rgb)

    def _reject(packet):
        return FrameObservation(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            processed_ns=packet.captured_ns + 1_000_000,
            quality_pass=False,
            quality_reasons=("w0b-synth-reject",),
            face_count=0,
            face_box=None,
            identity_scores={},
            quality_rank=0.0,
            model_generation="test-gen",
            gallery_digest="1" * 64,
        )

    def _match(packet):
        return FrameObservation(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            processed_ns=packet.captured_ns + 1_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(0.0, 0.0, 2.0, 2.0),
            identity_scores={"enroll-23": 0.9, "enroll-07": 0.1},
            quality_rank=0.9,
            model_generation="test-gen",
            gallery_digest="1" * 64,
        )

    consent = ConsentRecord(
        session_id=session_id,
        participant_id="part-w0b-001",
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-10-05T00:00:00Z",
        record_expires_at_utc="2026-11-04T00:00:00Z",
        image_expires_at_utc="2026-10-14T00:00:00Z",
    )
    engine = SessionEngine(_profile(), "1" * 64, "test-gen")
    return DesktopSession(
        engine=engine,
        source=FakeCapture(frames=[_packet(seq) for seq in range(1, 60)]),
        scorer=_match if matched else _reject,
        session_id=session_id,
    ), consent


def _window(app, *, gallery=None):
    from facecore.live.qt_window import _QtResearchWindow

    desktop, consent = _desktop()
    return _QtResearchWindow(
        desktop,
        consent=consent,
        gallery=gallery,
        offscreen=True,
    )


def _round_row(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    """A demo CSV with the real header and zero data rows."""
    from facecore.research.cli import (
        G3_DEMO_RESULTS_CSV_COLUMNS,
        demo_results_csv_path,
    )

    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    path = demo_results_csv_path(store, "2026-10-05T00:00:00+00:00")
    with path.open("w", encoding="utf-8", newline="") as fh:
        import csv as _csv

        w = _csv.writer(fh)
        w.writerow(G3_DEMO_RESULTS_CSV_COLUMNS)
    return path, {len(r) for r in csv.reader(path.open(encoding="utf-8"))}


def _verdict_window(app, tmp_path: Path, **kwargs: Any) -> Any:
    """A demo-mode window whose ✓／✗ path can persist (fail-closed safe).

    Demo mode (plaintext `demo_results_csv`, no recorder) is the sink
    these tests exercise: `has_label_persistence` accepts it, so
    `_press_key` reaches the CSV writer instead of refusing with
    「標註未綁定，無法落盤」. The round is driven to a real terminal via
    `process_until_terminal`, never synthesized.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from facecore.live.qt_window import _QtResearchWindow

    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    from facecore.research.cli import demo_results_csv_path

    demo_csv = demo_results_csv_path(store, "2026-10-05T00:00:00+00:00")

    def _next_session():
        desktop2, consent2 = _desktop(
            session_id=f"w0b-{len(states) + 1}", matched=True
        )
        desktop2.configure_demo_label_persistence(
            demo_csv, actor_ref="qt-operator"
        )
        states.append(desktop2)
        return desktop2, consent2, None

    desktop, consent = _desktop(matched=True)
    states: list[Any] = [desktop]

    win = _QtResearchWindow(
        desktop,
        consent=consent,
        gallery=_gallery(["enroll-23", "enroll-07"]),
        offscreen=True,
        demo_results_csv=demo_csv,
        next_session=_next_session,
        **kwargs,
    )
    win.show()
    win.enter_ready()
    win.start_clicked()
    win.process_until_terminal(max_steps=200)
    return win, demo_csv


def _read_verdicts(demo_csv: Path) -> list[dict[str, str]]:
    with demo_csv.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_press_correct_writes_verdict_correct(app, tmp_path: Path):
    """✓ must land `operator_verdict=correct` in the written row.

    RED on current code: neither Qt writer passes `operator_verdict`,
    so the row carries the default empty string.
    """
    win, demo_csv = _verdict_window(app, tmp_path)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    win.press_correct()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 1, f"exactly one row must be written, got {len(rows)}"
    assert rows[0]["operator_verdict"] == "correct", (
        f"✓ must write correct, got {rows[0]['operator_verdict']!r}"
    )


def test_press_incorrect_writes_verdict_incorrect(app, tmp_path: Path):
    """✗ must land `operator_verdict=incorrect` in the written row.

    RED on current code: same missing kwarg as the ✓ path.
    """
    win, demo_csv = _verdict_window(app, tmp_path)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    win.press_incorrect()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 1, f"exactly one row must be written, got {len(rows)}"
    assert rows[0]["operator_verdict"] == "incorrect", (
        f"✗ must write incorrect, got {rows[0]['operator_verdict']!r}"
    )


def test_rerun_path_writes_verdict_skipped(app, tmp_path: Path):
    """再次辨識 without a verdict must write `operator_verdict=skipped`.

    RED on current code: `_record_unlabeled_round` forwards ground truth
    but not the verdict, so the row is the 「ground truth 有值 ＋ verdict
    空」 tuple the spec declares a data anomaly (§5.1).
    """
    win, demo_csv = _verdict_window(app, tmp_path)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    win.recognize_again_clicked()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 1, f"exactly one row must be written, got {len(rows)}"
    assert rows[0]["operator_verdict"] == "skipped", (
        f"再次辨識 must write skipped, got {rows[0]['operator_verdict']!r}"
    )


def test_each_round_can_set_its_own_verdict(app, tmp_path: Path):
    """「每輪可獨立設定」 in button form: two rounds, two verdicts.

    The time proposition this file always defended, with the setting
    mechanism changed from combos to buttons. A verdict latched at
    startup would make both rows identical.
    """
    win, demo_csv = _verdict_window(app, tmp_path)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    win.press_correct()
    win.start_clicked()
    win.process_until_terminal(max_steps=200)
    assert win._mode == win._MODE_RESULT, "round 2 must reach Result"
    win.press_incorrect()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 2, f"two rows must be written, got {len(rows)}"
    assert (rows[0]["operator_verdict"], rows[1]["operator_verdict"]) == (
        "correct",
        "incorrect",
    ), (
        "round 2 must record its own verdict, not round 1's: "
        f"got {[r['operator_verdict'] for r in rows]}"
    )


# The 「both writers forward the values」 guard used to live here, counting
# the literal `probe_kind=self._probe_kind_value()` and asserting >= 2. It
# moved to tests/live/test_d7_w0b_writer_wiring.py, which parses the AST
# instead, for two reasons that a count cannot handle:
#
#   · a third writer that omits the columns leaves the count at 2, so the
#     guard stayed green while a new path recorded empty ground truth;
#   · one mention inside a comment satisfies the count even when both real
#     writers are broken.
#
# It also lost its PySide6 requirement there, so it runs in `verify` as well.
