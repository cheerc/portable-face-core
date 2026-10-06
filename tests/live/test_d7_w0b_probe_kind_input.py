"""D7-A W0-b: the ground-truth columns must be recordable per round.

`probe_kind` / `presenting_identity` have existed in the demo CSV header
since #138 and every row has carried an empty string, because nothing
ever passed a non-empty value. The runbook spent a section explaining
that empty means 「沒有記錄」.

What these tests defend is NOT 「a widget exists」 — that is trivially
true and proves nothing. A combo that is constructed and never read, or
one whose selection is frozen at App startup, satisfies every check a
reader would naturally write:

    assert window.probe_kind_combo is not None      # passes on a dead widget
    assert "target" in items(window.probe_kind_combo)  # passes if nothing consumes it

Both of those pass on a build where the columns stay permanently empty.
So the guard here is about the VALUE PATH: the value the operator picks
must reach the CSV row, and must be re-read for every round rather than
captured once.

The distinction that matters, and the one a future reader is most likely
to get wrong, is that 「每輪可獨立設定」 is a claim about time. It means
picking `nontarget` for round 1 and `target` for round 2 must produce
different cells. A single App start holds one CSV file (#145), and the
App is designed for 20+ consecutive rounds inside one window, so an
implementation that latched the value at startup would look perfectly
correct in any single-round test.
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


def _desktop(session_id: str = "w0b-1"):
    """A DesktopSession wired the way `tests/live/test_desktop.py` does.

    The window's two combos are read at WRITE time and never touch the
    engine, so the desktop only has to be constructible — but it must be
    real, because a stub would let the window's own constructor fail for
    an unrelated reason and mask what these tests are about.
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

    def _scorer(packet):
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
        source=FakeCapture(frames=[_packet(seq) for seq in range(1, 4)]),
        scorer=_scorer,
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


def test_probe_kind_domain_is_exactly_target_and_nontarget(app):
    """runbook:535 — 「只有兩個：`target`、`nontarget`。沒有第三個值」.

    A test that only checked 「target exists」 would let a stray third
    option in. The domain is closed, so it is pinned from both sides.
    """
    win = _window(app)
    values = [
        win.probe_kind_combo.itemData(i)
        for i in range(win.probe_kind_combo.count())
    ]
    non_empty = [v for v in values if v]
    assert non_empty == ["target", "nontarget"], (
        f"probe_kind must offer exactly target/nontarget, got {non_empty}"
    )


def test_identity_candidates_are_the_gallery_keys(app):
    """The combo must offer loaded identities, never free text.

    A free-text identity one character off (`enrol-24`) is a silent
    ground-truth error, and these columns exist to be ground truth. So
    the assertion is on the OPTION SET, not on the presence of a widget.
    """
    win = _window(app, gallery=_gallery(["enroll-23", "enroll-07"]))
    win.probe_kind_combo.setCurrentIndex(
        win.probe_kind_combo.findData("target")
    )
    combo = win.presenting_identity_combo
    values = [combo.itemData(i) for i in range(combo.count())]
    assert "enroll-23" in values, "loaded identities must be offered"
    assert "enroll-07" in values, "loaded identities must be offered"
    # Anything that is not a loaded identity would defeat the point.
    assert values == ["", "enroll-07", "enroll-23"], (
        "identity options must be exactly gallery keys plus the unset "
        f"item, got {values}"
    )


def test_switching_to_nontarget_disables_the_identity_combo(app):
    """The non-target side is undefined on purpose, and must LOOK it.

    The runbook assigns 「沒有註冊的測試者填什麼」 to W0-b scheduling. This
    task is only that prerequisite, so the operator must be able to see
    that the value is not available yet — a blank cell would read as
    「this round had no identity」, which is a claim, not an absence.
    """
    win = _window(app, gallery=_gallery(["enroll-23"]))
    combo = win.presenting_identity_combo
    assert combo.isEnabled(), "target side must be usable before any pick"

    win.probe_kind_combo.setCurrentIndex(
        win.probe_kind_combo.findData("nontarget")
    )
    assert not combo.isEnabled(), (
        "the nontarget identity must not be writable — it is undefined until "
        "W0-b scheduling defines it"
    )
    assert combo.count() == 1 and combo.itemData(0) == "", (
        "the disabled side must show one explicit item, not leftover identities"
    )


def _round(*, attempt_id: str = "a1"):
    """A real `RoundComplete` wrapping a real `SessionResult`.

    Both are validated on construction (`SessionResult.__post_init__`
    rejects a `matched` status without a matched identity), so this
    cannot drift into a shape the row reducer never sees.
    """
    from facecore.live.contracts import SessionResult, SessionStatus
    from facecore.live.qt_window import RoundComplete

    terminal = SessionResult(
        session_id="w0b-1",
        schema_version="v1",
        status=SessionStatus.matched,
        matched_identity="enroll-23",
        reason_codes=("supported_3_frames",),
        elapsed_ms=1000.0,
        frames_sampled=5,
        frames_usable=3,
        frames_rejected=2,
        frames_dropped=0,
        support_sequences=(3,),
        profile_digest="2" * 64,
        model_generation="test-gen",
        gallery_digest="1" * 64,
    )
    return RoundComplete(
        session_id="w0b-1",
        attempt_id=attempt_id,
        terminal=terminal,
        observations=(),
        label_kind="enrolled",
        label_identity="enroll-23",
        profile_version="g3-v1",
        started_utc="2026-10-05T00:00:00+00:00",
    )


def test_the_value_reaches_the_csv_row(app, tmp_path: Path):
    """End-to-end: the operator's pick must land in the written row.

    This is the test a 「widget exists」 reading would not write. It goes
    through `append_g3_demo_results_csv` so the assertion is on the file
    the operator opens, not on an internal attribute.
    """
    path, widths = _round_row(tmp_path)
    assert widths == {39}, f"header must stay 39 columns, got {widths}"

    def _pick(probe_kind: str, identity: str) -> dict[str, Any]:
        from facecore.research.cli import g3_demo_round_row

        return g3_demo_round_row(
            _round(),
            required_support=3,
            labeled_at_utc="2026-10-05T00:00:01+00:00",
            probe_kind=probe_kind,
            presenting_identity=identity,
        )

    r1 = _pick("target", "enroll-23")
    assert r1["probe_kind"] == "target"
    assert r1["presenting_identity"] == "enroll-23"

    r2 = _pick("nontarget", "")
    assert r2["probe_kind"] == "nontarget"
    assert r2["presenting_identity"] == ""


def test_each_round_can_set_its_own_value(app):
    """「每輪可獨立設定」 is a claim about TIME, and this is that claim.

    One App start holds one CSV file (#145), and the window is built for
    20+ consecutive rounds. An implementation that latched the value at
    startup would pass any single-round test, so the guard reads the
    value TWICE with different picks and requires different cells.
    """
    from facecore.research.cli import g3_demo_round_row

    win = _window(app, gallery=_gallery(["enroll-23", "enroll-24"]))

    def _row_for(probe_kind: str, identity: str) -> dict[str, Any]:
        combo_idx = win.probe_kind_combo.findData(probe_kind)
        assert combo_idx >= 0, f"{probe_kind} must be offered"
        win.probe_kind_combo.setCurrentIndex(combo_idx)
        id_combo = win.presenting_identity_combo
        if identity:
            idx = id_combo.findData(identity)
            assert idx >= 0, f"{identity} must be offered when probe_kind=target"
            id_combo.setCurrentIndex(idx)
        return {
            "probe_kind": win._probe_kind_value(),
            "presenting_identity": win._presenting_identity_value(),
        }

    r1_values = _row_for("target", "enroll-23")
    r1 = g3_demo_round_row(
        _round(),
        required_support=3,
        labeled_at_utc="2026-10-05T00:00:01+00:00",
        probe_kind=r1_values["probe_kind"],
        presenting_identity=r1_values["presenting_identity"],
    )
    r2_values = _row_for("nontarget", "")
    r2 = g3_demo_round_row(
        _round(),
        required_support=3,
        labeled_at_utc="2026-10-05T00:00:02+00:00",
        probe_kind=r2_values["probe_kind"],
        presenting_identity=r2_values["presenting_identity"],
    )

    assert (r1["probe_kind"], r1["presenting_identity"]) == (
        "target",
        "enroll-23",
    )
    # The second round must NOT inherit round 1's pick. A latched value
    # would make this `("target", "enroll-23")` again.
    assert (r2["probe_kind"], r2["presenting_identity"]) == ("nontarget", ""), (
        "round 2 must record its own pick, not round 1's"
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
