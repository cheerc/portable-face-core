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
    from tests.live.test_qt_window import _AdvancingClock

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
    clock = _AdvancingClock()

    win = _QtResearchWindow(
        desktop,
        consent=consent,
        gallery=_gallery(["enroll-23", "enroll-07"]),
        offscreen=True,
        demo_results_csv=demo_csv,
        next_session=_next_session,
        clock_ns=clock,
        clock_advance=clock.advance,
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


def _find_ground_truth(win: Any, pair: tuple[str, str]) -> int:
    """Index of the (probe_kind, identity) option, or -1.

    `QComboBox.findData` does not reliably match tuple payloads
    (QVariant round-trip), so compare the stored pairs directly.
    """
    combo = win.ground_truth_combo
    for i in range(combo.count()):
        if combo.itemData(i) == pair:
            return i
    return -1


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


def test_press_incorrect_discloses_then_writes_incorrect(app, tmp_path: Path):
    """✗ must first disclose the input area, then write `incorrect`.

    Progressive disclosure (spec §4.2): the first ✗ press only reveals
    the ground-truth picker — the operator has judged nothing yet, so
    nothing is written. After picking, confirming writes `incorrect`
    with the picked pair. An accidental ✗ alone must never record a
    wrong identity.
    """
    win, demo_csv = _verdict_window(app, tmp_path)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    win.press_incorrect()
    assert _read_verdicts_len(demo_csv) == 0, "disclosure must not write"
    assert win.disclosure_widget.isVisible(), "input area must appear"
    # Regression guard: showing the parent must show the children too.
    # Qt does not re-show an explicitly hidden child with its parent, so
    # hiding the children at build time left the disclosure area empty
    # with no selectable menu — and the existing tests only asserted the
    # parent, letting it stay green.
    assert win.ground_truth_combo.isVisible(), "menu must be visible"
    assert win.nontarget_dir_button.isVisible(), "folder button must show"
    idx = _find_ground_truth(win, ("target", "enroll-07"))
    assert idx >= 0, "gallery identity must be offered"
    # Route through the real signal: move away first so the pick always
    # fires even when the target is already the current row.
    win.ground_truth_combo.setCurrentIndex(
        (idx + 1) % win.ground_truth_combo.count()
    )
    win.ground_truth_combo.setCurrentIndex(idx)
    win.confirm_incorrect()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 1, f"exactly one row must be written, got {len(rows)}"
    assert rows[0]["operator_verdict"] == "incorrect", (
        f"✗ must write incorrect, got {rows[0]['operator_verdict']!r}"
    )
    assert rows[0]["probe_kind"] == "target", (
        f"✗ must write the picked kind, got {rows[0]['probe_kind']!r}"
    )
    assert rows[0]["presenting_identity"] == "enroll-07", (
        "✗ must write the picked identity, "
        f"got {rows[0]['presenting_identity']!r}"
    )


def _read_verdicts_len(demo_csv: Path) -> int:
    if not demo_csv.is_file():
        return 0
    return len(_read_verdicts(demo_csv))


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
    idx2 = _find_ground_truth(win, ("nontarget", "outsider"))
    assert idx2 >= 0, "outsider must be offered without a folder"
    win.ground_truth_combo.setCurrentIndex(
        (idx2 + 1) % win.ground_truth_combo.count()
    )
    win.ground_truth_combo.setCurrentIndex(idx2)
    win.confirm_incorrect()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 2, f"two rows must be written, got {len(rows)}"
    assert (rows[0]["operator_verdict"], rows[1]["operator_verdict"]) == (
        "correct",
        "incorrect",
    ), (
        "round 2 must record its own verdict, not round 1's: "
        f"got {[r['operator_verdict'] for r in rows]}"
    )
    assert rows[1]["probe_kind"] == "nontarget", (
        "round 2 must not inherit round 1's kind: "
        f"got {rows[1]['probe_kind']!r}"
    )


def test_skip_button_writes_verdict_skipped(app, tmp_path: Path):
    """略過 must write `operator_verdict=skipped` with empty ground truth.

    Skip is a peer of ✓／✗ (spec §4.2), not a disclosure option. Per
    decision, the recommended skip leaves `probe_kind` /
    `presenting_identity` empty and never rewrites stored CSVs — so the
    row carries empty ground truth plus the explicit value.
    """
    win, demo_csv = _verdict_window(app, tmp_path)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    assert win.skip_button.isVisible(), "skip must be offered here"
    win.press_skip()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 1, f"exactly one row must be written, got {len(rows)}"
    assert rows[0]["operator_verdict"] == "skipped", (
        f"略過 must write skipped, got {rows[0]['operator_verdict']!r}"
    )
    assert rows[0]["probe_kind"] == "", "skip must not invent a kind"
    assert rows[0]["presenting_identity"] == "", "skip must not invent a who"


def test_invalid_input_round_has_no_skip(app, tmp_path: Path):
    """`invalid_input` rounds offer only ✓／✗ (spec §4.5).

    There is no identification to decline there — only the missed-person
    flag the two buttons record — so a skip button would invite random
    presses that pollute `operator_verdict`.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from facecore.live.qt_window import _QtResearchWindow
    from tests.live.test_qt_window import _AdvancingClock

    from facecore.research.cli import demo_results_csv_path

    store = tmp_path / "store"
    store.mkdir(parents=True, exist_ok=True)
    demo_csv = demo_results_csv_path(store, "2026-10-05T00:00:00+00:00")
    clock = _AdvancingClock()

    def _next_session():
        desktop2, consent2 = _desktop(session_id="w0b-ii")
        desktop2.configure_demo_label_persistence(
            demo_csv, actor_ref="qt-operator"
        )
        return desktop2, consent2, None

    desktop, consent = _desktop()
    win = _QtResearchWindow(
        desktop,
        consent=consent,
        gallery=_gallery(["enroll-23", "enroll-07"]),
        offscreen=True,
        demo_results_csv=demo_csv,
        next_session=_next_session,
        clock_ns=clock,
        clock_advance=clock.advance,
    )
    win.show()
    win.enter_ready()
    win.start_clicked()
    win.process_until_terminal(max_steps=200)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    assert win.desktop.terminal is not None
    assert win.desktop.terminal.status.value == "invalid_input", (
        "this fixture must land invalid_input, "
        f"got {win.desktop.terminal.status.value}"
    )
    assert not win.skip_button.isVisible(), "skip must be hidden here"
    assert not win.skip_button.isEnabled(), "skip must be disabled here"


def test_auto_discovers_sibling_nontarget_dir(app, tmp_path: Path):
    """Enrollment directory's sibling `non-target` must be auto-discovered.

    When `enrollment_dir` is provided, `_QtResearchWindow` checks its sibling
    `non-target` directory at initialization time and loads its image stems
    into `_nontarget_stems`, setting `_nontarget_dir`.
    """
    sample_root = tmp_path / "face_sample"
    enroll_dir = sample_root / "enrollment"
    nontarget_dir = sample_root / "non-target"
    enroll_dir.mkdir(parents=True)
    nontarget_dir.mkdir(parents=True)
    (nontarget_dir / "nontarget-01.jpg").write_bytes(b"fake-jpg")
    (nontarget_dir / "nontarget-02.png").write_bytes(b"fake-png")
    (nontarget_dir / "ignore.txt").write_text("not-an-image", encoding="utf-8")

    from facecore.live.qt_window import _QtResearchWindow

    desktop, consent = _desktop()
    win = _QtResearchWindow(
        desktop,
        consent=consent,
        gallery=_gallery(["enroll-23"]),
        enrollment_dir=enroll_dir,
        offscreen=True,
    )
    assert win._nontarget_dir == str(nontarget_dir)
    assert win._nontarget_stems == ["nontarget-01", "nontarget-02"]
    win._refresh_ground_truth_options()
    idx = _find_ground_truth(win, ("nontarget", "nontarget-01"))
    assert idx >= 0, "auto-discovered non-target stem must be in disclosure options"
    idx2 = _find_ground_truth(win, ("nontarget", "nontarget-02"))
    assert idx2 >= 0, "second auto-discovered stem must be in disclosure options"


def test_press_incorrect_picks_nontarget_stem_writes_demo_csv(app, tmp_path: Path):
    """Selecting an auto-discovered non-target stem writes to demo CSV.

    When operator clicks ✗, selects a non-target identity (e.g. nontarget-01)
    and confirms, the demo CSV must record:
      operator_verdict=incorrect
      probe_kind=nontarget
      presenting_identity=nontarget-01
    """
    sample_root = tmp_path / "face_sample"
    enroll_dir = sample_root / "enrollment"
    nontarget_dir = sample_root / "non-target"
    enroll_dir.mkdir(parents=True)
    nontarget_dir.mkdir(parents=True)
    (nontarget_dir / "nontarget-01.jpg").write_bytes(b"fake-jpg")

    win, demo_csv = _verdict_window(app, tmp_path, enrollment_dir=enroll_dir)
    assert win._mode == win._MODE_RESULT, "round must reach Result first"
    win.press_incorrect()
    assert win.disclosure_widget.isVisible(), "disclosure area must appear"
    idx = _find_ground_truth(win, ("nontarget", "nontarget-01"))
    assert idx >= 0, "auto-discovered non-target must be offered"
    win.ground_truth_combo.setCurrentIndex(
        (idx + 1) % win.ground_truth_combo.count()
    )
    win.ground_truth_combo.setCurrentIndex(idx)
    win.confirm_incorrect()
    rows = _read_verdicts(demo_csv)
    assert len(rows) == 1, f"exactly one row must be written, got {len(rows)}"
    assert rows[0]["operator_verdict"] == "incorrect"
    assert rows[0]["probe_kind"] == "nontarget"
    assert rows[0]["presenting_identity"] == "nontarget-01"


def test_window_layout_split_and_bottom_controls(app):
    """Layout rework: camera on left, info on right, controls pinned at bottom.

    Defends that the window uses a horizontal split for content (preview left,
    scrollable info right) and places the controls row directly at the bottom
    of the central vertical layout so operator buttons remain on-screen.
    """
    from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout

    win = _window(app)
    central = win.centralWidget()
    assert central is not None
    main_layout = central.layout()
    assert isinstance(main_layout, QVBoxLayout)
    # Bottom item must be the controls layout containing the action buttons
    assert main_layout.count() >= 2
    bottom_item = main_layout.itemAt(main_layout.count() - 1)
    assert isinstance(bottom_item, QHBoxLayout)
    # Verify controls buttons sit in this bottom row
    buttons = [
        bottom_item.itemAt(i).widget()
        for i in range(bottom_item.count())
        if bottom_item.itemAt(i).widget() is not None
    ]
    assert win.start_button in buttons
    assert win.correct_button in buttons
    assert win.incorrect_button in buttons
    assert win.skip_button in buttons

    # Content layout above controls
    content_item = main_layout.itemAt(0)
    assert isinstance(content_item, QHBoxLayout)
    # Left widget contains preview and camera combo
    left_item = content_item.itemAt(0)
    assert left_item.widget() is not None
    left_widgets = [
        left_item.widget().layout().itemAt(i).widget()
        for i in range(left_item.widget().layout().count())
        if left_item.widget().layout().itemAt(i).widget() is not None
    ]
    assert win.camera_combo in left_widgets
    assert win.preview_label in left_widgets

    # Right item is a QScrollArea containing info labels
    right_item = content_item.itemAt(1)
    assert isinstance(right_item.widget(), QScrollArea)
    right_inner = right_item.widget().widget()
    assert right_inner is not None
    right_widgets = [
        right_inner.layout().itemAt(i).widget()
        for i in range(right_inner.layout().count())
        if right_inner.layout().itemAt(i).widget() is not None
    ]
    assert win.status_label in right_widgets
    assert win.disclosure_widget in right_widgets


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
