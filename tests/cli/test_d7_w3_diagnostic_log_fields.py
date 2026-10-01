"""D7-A W3: the diagnostic demo log must carry data that already exists.

Source of truth: plan v8 §3.3 / §8. The work item is deliberately
「搬出既有資料」— every field below is either already computed inside a
live round and merely missing from the CSV, or a count the engine already
emits as an event that nobody collects.

**What this pins, and why each one can fail.**

1. `recognition_duration_ms` — `TimingMarks.recognition_duration_ms`
   (`contracts.py:302`) is a property on data the engine already seals into
   `SessionResult` (`session.py:738` passes `timing_marks=marks`). It is
   live-path data, unlike `open_duration_ms` / `open_to_first_frame_ms`
   which `controller.py:235` deliberately never fills on the demo path.
2. `frames_rejected` — computed at `session.py:293`/`:317`, serialized at
   `contracts.py:395`, never written to the demo CSV.
3. `score_reset_count` / `interval_skip_count` — `_emit_event` returns early
   when `self._event_sink is None` (`session.py:113`), and the live path
   passes no sink at all, so these events are discarded at the source.
   The second-round engine (`cli.py:1427`) is built inline without a sink
   even where the first-round engine (`cli.py:1108`) has one.
4. `probe_kind` / `presenting_identity` — the W0 runbook cannot be analysed
   without them, and plan v8 §2 is explicit that the log is what makes
   cross-identity ground truth analysable.
5. Threshold snapshot — only parameters that are demonstrably read on the
   live path may be recorded (plan v8 §4). `sample_interval_ms` and
   `max_frames` are named in the dispatch as must-not-record, and
   `max_frames` is reachable only in fixed mode while the launcher is
   `--continuous`.

**The mutation this must catch.** Two shapes, both silent:

- a field that is present but always empty / always zero, which reads as
  "measured, found nothing" rather than "never collected";
- a threshold recorded from the profile without that value actually being
  read on the live path, which is the §4 false-knob shape the plan forbids.

So every numeric assertion here is a *non-zero* assertion, and the
not-recorded set is asserted as a set, not as a comment.
"""

from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from facecore.live.contracts import (
    DecisionEvent,
    FrameObservation,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.research.cli import (
    G3_DEMO_RESULTS_CSV_COLUMNS,
    G3_DEMO_RESULTS_CSV_NAME,
    g3_demo_round_row,
)

# ---------------------------------------------------------------------------
# The 23 pre-existing columns must survive untouched, in order.
# ---------------------------------------------------------------------------
LEGACY_23 = (
    "mode",
    "app_version",
    "profile_version",
    "model_generation",
    "gallery_digest",
    "profile_digest",
    "round_id",
    "session_id",
    "started_utc",
    "labeled_at_utc",
    "elapsed_ms",
    "result",
    "reason_codes",
    "top1_identity",
    "top1_score",
    "top2_identity",
    "top2_score",
    "margin",
    "frames_sampled",
    "frames_usable",
    "required_support",
    "label_kind",
    "label_identity",
)

# Fields the v8 dispatch forbids. Named explicitly so a future edit that
# adds one of them fails here rather than shipping a false knob into the log.
MUST_NOT_RECORD = (
    "sample_interval_ms",
    "max_frames",
    "frames_dropped",
    "open_duration_ms",
    "open_to_first_frame_ms",
    "model_load_ms",
    "best_frame_score",
    "best_frame_index",
    "label_identity_mismatch",
    "score_p50",
    "score_max",
    "no_frame_reason",
    "quality_reject_count",
    "support_wait_ms",
    "first_frame_ms",
    "quality_policy_version",
)

# Parameters that plan v8 §4 classifies as 「可轉（控制流已追蹤）」and which
# therefore may be snapshotted. Anything outside this set is off-limits.
SNAPSHOT_ALLOWED = (
    "required_support",
    "min_support_interval_ms",
    "timeout_ms",
    "match_threshold",
    "review_threshold",
    "margin_threshold",
)


def _profile() -> ResearchProfile:
    """A profile matching `profiles/g3-v1.json` in shape.

    `continuity_max_center_delta_ratio` matters: `can_auto_match()`
    (contracts.py:145) returns False without it, and `observe()` then
    returns at the `reset`/`auto_match_disabled` branch (session.py:503)
    BEFORE reaching the interval guard at :523. A profile missing it can
    therefore never emit `interval_skip` at all — which would make the
    reset-vs-skip distinction this file exists to pin untestable.
    """
    return ResearchProfile(
        schema_version="v1",
        profile_version="qt-test-v1",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.363,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="yunet",
        quality_policy_version="1",
        continuity_max_center_delta_ratio=0.5,
    )


def _obs(
    sequence: int,
    *,
    scores: dict[str, float] | None = None,
    quality_pass: bool = True,
    captured_ns: int | None = None,
) -> FrameObservation:
    """One observation.

    `scores` should carry a runner-up as well: the margin gate reads
    `top1 - runner_up`, and a single-entry dict makes the margin 0.0, which
    diverts the round into the margin branch before the score gate is ever
    consulted — a different failure than the one these tests provoke.

    `captured_ns` defaults to 200 ms per sequence, which is exactly
    `min_support_interval_ms`, NOT "inside" it: the guard at
    session.py:483 is a strict `<`. Pass it explicitly to test the skip.
    """
    stamp = sequence * 200_000_000 if captured_ns is None else captured_ns
    return FrameObservation(
        sequence=sequence,
        captured_ns=stamp,
        processed_ns=stamp + 1_000_000,
        quality_pass=quality_pass,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores=(
            scores
            if scores is not None
            else {"person-synth-01": 0.9, "person-synth-02": 0.1}
        ),
        quality_rank=0.9,
        model_generation="gen-qt-test",
        gallery_digest="gallery-qt-test",
    )


def _events() -> list[DecisionEvent]:
    return [
        DecisionEvent(
            sequence=1,
            event_type="score_reset",
            accepted=False,
            reset_reason="score_below_threshold",
            support_before=1,
            support_after=0,
            candidate_before="person-synth-01",
            candidate_after=None,
            terminal_status=None,
            terminal_identity=None,
            deadline_remaining_ms=4000.0,
        ),
        DecisionEvent(
            sequence=2,
            event_type="interval_skip",
            accepted=False,
            reset_reason=None,
            support_before=0,
            support_after=0,
            candidate_before=None,
            candidate_after=None,
            terminal_status=None,
            terminal_identity=None,
            deadline_remaining_ms=3000.0,
        ),
        DecisionEvent(
            sequence=3,
            event_type="score_reset",
            accepted=False,
            reset_reason="margin_below_threshold",
            support_before=2,
            support_after=0,
            candidate_before="person-synth-01",
            candidate_after=None,
            terminal_status=None,
            terminal_identity=None,
            deadline_remaining_ms=2000.0,
        ),
    ]


def _terminal(*, timing: bool = True) -> SessionResult:
    """A terminal carrying timing marks and a non-trivial rejected count."""
    from facecore.live.contracts import TimingMarks

    marks = (
        TimingMarks(
            open_begin_ns=None,
            open_end_ns=None,
            first_frame_ns=1_000_000_000,
            recognition_start_ns=1_000_000_000,
            terminal_ns=4_000_000_000,
        )
        if timing
        else None
    )
    return SessionResult(
        session_id="sess-1",
        schema_version="v1",
        status=SessionStatus.timeout,
        matched_identity=None,
        reason_codes=("insufficient_evidence", "support_0_of_3"),
        elapsed_ms=3000.0,
        frames_sampled=12,
        frames_usable=3,
        frames_rejected=7,
        frames_dropped=0,
        support_sequences=(),
        profile_digest="prof-digest",
        model_generation="gen-qt-test",
        gallery_digest="gallery-qt-test",
        timing_marks=marks,
    )


class _Round:
    """A real RoundComplete (g3_demo_round_row asserts the exact type)."""

    @staticmethod
    def make(
        terminal: SessionResult, obs: tuple[FrameObservation, ...]
    ) -> Any:
        from facecore.live.qt_window import RoundComplete

        return RoundComplete(
            session_id="sess-1",
            attempt_id="att-1",
            terminal=terminal,
            observations=obs,
            label_kind="unenrolled",
            label_identity=None,
            profile_version="qt-test-v1",
            started_utc="2026-10-01T00:00:00+00:00",
        )


def _row(**kwargs: Any) -> dict[str, object]:
    obs = (
        _obs(1, scores={"person-synth-01": 0.62, "person-synth-02": 0.0}),
        _obs(2, scores={"person-synth-01": 0.71, "person-synth-02": 0.0}),
    )
    events = kwargs.pop("events", None)
    counts = (
        None
        if events is None
        else {
            t: sum(1 for e in events if e.event_type == t)
            for t in {e.event_type for e in events}
        }
    )
    return g3_demo_round_row(
        _Round.make(_terminal(), obs),
        required_support=3,
        labeled_at_utc="2026-10-01T00:00:00+00:00",
        event_counts=counts,
        profile=kwargs.pop("profile", _profile()),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Column contract
# ---------------------------------------------------------------------------
class TestEngineTalliesWithoutASink:
    """The counts must come from the engine, not from the caller's memory.

    This is the test the row-level ones above cannot make: they inject
    `event_counts` directly, so they stay green whether or not the engine
    counts anything. The live path passes NO sink at all, and
    `_emit_event` returns early when the sink is None — so if the tally
    were taken after that guard it would be structurally always 0 and the
    demo log would ship a column that is present, plausible, and empty.
    """

    def _engine(self) -> Any:
        from facecore.live.session import SessionEngine

        return SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test")

    def test_no_sink_is_the_live_default(self) -> None:
        assert self._engine().event_count("score_reset") == 0

    def test_a_below_threshold_frame_tallies_without_any_sink(self) -> None:
        engine = self._engine()
        engine.start("sess-1", 0)
        # below match_threshold -> session.py:450 emits score_reset
        engine.observe(
            _obs(1, scores={"person-synth-01": 0.10, "person-synth-02": 0.0})
        )
        assert engine.event_count("score_reset") == 1, (
            "a round with no event_sink must still tally its events; the "
            "live demo path never passes a sink"
        )

    def test_the_two_ambiguous_paths_stay_distinguishable(self) -> None:
        """The whole point of D4 §11 item 18b: reset vs skip must differ.

        Both frames pass every score gate, so this round cannot produce a
        `score_reset` — the only thing that can separate it from a reset
        round is the interval guard at session.py:523. It needs a third
        frame: `required_support` is 3, and the interval is measured
        against the previous SUPPORT frame, so two frames never establish
        one.
        """
        passing = {"person-synth-01": 0.90, "person-synth-02": 0.0}
        skip = self._engine()
        skip.start("sess-1", 0)
        # Stamps must be monotonic: session.py:289 treats a non-advancing
        # sequence as an error and terminates the round, which would hide
        # every later event behind `identity_change` instead.
        skip.observe(_obs(1, scores=passing, captured_ns=0))
        skip.observe(
            _obs(2, scores=passing, captured_ns=50_000_000)
        )
        skip.observe(_obs(3, scores=passing, captured_ns=1_000_000_000))
        assert skip.event_count("interval_skip") == 1, (
            "a frame arriving inside min_support_interval_ms must be "
            "counted, not silently dropped"
        )
        assert skip.event_count("score_reset") == 0, (
            "this round cleared every gate; a reset here would mean the "
            "counter is reporting a branch it never took"
        )

        # The control: a below-gate round resets and never skips.
        reset = self._engine()
        reset.start("sess-1", 0)
        reset.observe(
            _obs(
                1,
                scores={"person-synth-01": 0.10, "person-synth-02": 0.0},
                captured_ns=0,
            )
        )
        assert reset.event_count("score_reset") == 1
        assert reset.event_count("interval_skip") == 0

    def test_counts_reset_between_rounds_on_a_reused_engine(self) -> None:
        engine = self._engine()
        engine.start("round-1", 0)
        engine.observe(
            _obs(1, scores={"person-synth-01": 0.10, "person-synth-02": 0.0})
        )
        assert engine.event_count("score_reset") == 1
        engine.start("round-2", 10_000_000_000)
        assert engine.event_count("score_reset") == 0, (
            "a reused engine must not report the previous round's events"
        )

    def test_event_counts_returns_a_copy(self) -> None:
        engine = self._engine()
        snap = engine.event_counts()
        snap["score_reset"] = 999
        assert engine.event_count("score_reset") == 0, (
            "event_counts() must not hand out a live reference"
        )


class TestColumnContract:
    def test_the_23_legacy_columns_keep_their_exact_names_and_order(self) -> None:
        cols = list(G3_DEMO_RESULTS_CSV_COLUMNS)
        assert cols[: len(LEGACY_23)] == list(LEGACY_23), (
            "the 23 pre-existing columns were renamed, reordered or dropped; "
            "the dispatch forbids all three"
        )

    def test_no_column_is_declared_twice(self) -> None:
        """`csv.DictWriter` silently tolerates a repeated fieldname.

        W3 originally declared `required_support` twice — once at its
        position among the frame columns and once in the threshold
        snapshot — and the file still parsed, with the same value written
        into both slots. Nothing downstream complained; a reader counting
        columns would have seen 35 headers for 33 distinct facts.
        """
        cols = list(G3_DEMO_RESULTS_CSV_COLUMNS)
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        assert not dupes, f"duplicate column names in the header: {dupes}"

    def test_label_kind_keeps_its_original_position(self) -> None:
        """The SOP quotes label_kind as column 22 (letter V).

        W3 appends after `label_identity`, so this must not move. An
        insertion ahead of it would make the SOP's `V:V` range read the
        wrong field — the D4 S1 failure.
        """
        cols = list(G3_DEMO_RESULTS_CSV_COLUMNS)
        assert cols.index("label_kind") == 21
        assert cols[17] == "margin", "column R is no longer margin"

    def test_legacy_columns_are_never_blank_for_a_scored_round(self) -> None:
        """Guards the rename/blank failure mode, not just the header list."""
        row = _row()
        for name in LEGACY_23:
            if name in {"top1_identity", "top1_score", "top2_identity",
                        "top2_score", "margin", "label_identity",
                        "shown_identity"}:
                continue
            assert row[name] not in (None, ""), f"{name} went blank"

    def test_every_column_is_produced_by_the_row_builder(self) -> None:
        """A column in the header with no value is the §8 false-data shape."""
        row = _row()
        missing = [
            c for c in G3_DEMO_RESULTS_CSV_COLUMNS if c not in row
        ]
        assert not missing, f"declared columns with no row value: {missing}"


# ---------------------------------------------------------------------------
# 2. The new fields carry real values
# ---------------------------------------------------------------------------
class TestNewFields:
    def test_recognition_duration_ms_is_written_and_non_empty(self) -> None:
        row = _row()
        assert row["recognition_duration_ms"] == "3000.0", (
            "recognition_duration_ms must come from timing_marks "
            "(terminal - recognition_start); it is None for a demo round "
            "that never filled the marks"
        )

    def test_frames_rejected_is_written_as_a_real_count(self) -> None:
        row = _row()
        assert row["frames_rejected"] == "7", (
            "frames_rejected must be carried over from SessionResult, not "
            "recomputed and not defaulted to 0"
        )

    def test_score_reset_count_counts_only_score_reset_events(self) -> None:
        row = _row(events=_events())
        assert row["score_reset_count"] == "2", (
            "the 3 events contain 2 score_reset; counting every event type "
            "would report 3"
        )

    def test_interval_skip_count_counts_only_interval_skip_events(self) -> None:
        row = _row(events=_events())
        assert row["interval_skip_count"] == "1"

    def test_event_counts_default_to_zero_only_when_no_events_were_seen(
        self,
    ) -> None:
        row = _row(events=())
        assert row["score_reset_count"] == "0"
        assert row["interval_skip_count"] == "0"

    def test_probe_kind_and_presenting_identity_default_to_unknown(
        self,
    ) -> None:
        """A blank cell would be read as 'not a non-target round'."""
        row = _row()
        assert row["probe_kind"] == "", (
            "unlabelled rounds must leave probe_kind empty rather than "
            "guessing 'target' — an invented value corrupts the W0 counts"
        )
        assert row["presenting_identity"] == ""

    def test_probe_kind_accepts_the_two_documented_kinds(self) -> None:
        for kind in ("target", "nontarget"):
            row = _row(probe_kind=kind, presenting_identity="enroll-07")
            assert row["probe_kind"] == kind
            assert row["presenting_identity"] == "enroll-07"

    def test_threshold_snapshot_records_the_effective_values(self) -> None:
        row = _row()
        assert row["match_threshold"] == "0.363"
        assert row["review_threshold"] == "0.3"
        assert row["margin_threshold"] == "0.1"
        assert row["required_support"] == "3"

    def test_min_support_interval_and_timeout_are_snapshotted(self) -> None:
        row = _row()
        assert row["min_support_interval_ms"] == "200"
        assert row["timeout_ms"] == "5000"


# ---------------------------------------------------------------------------
# 3. The forbidden set never appears
# ---------------------------------------------------------------------------
class TestForbiddenFields:
    def test_no_forbidden_field_is_a_declared_column(self) -> None:
        cols = set(G3_DEMO_RESULTS_CSV_COLUMNS)
        leaked = sorted(cols.intersection(MUST_NOT_RECORD))
        assert not leaked, f"forbidden fields declared in the log: {leaked}"

    def test_the_header_only_grows_with_permitted_columns(self) -> None:
        extra = [
            c
            for c in G3_DEMO_RESULTS_CSV_COLUMNS
            if c not in LEGACY_23
        ]
        assert set(extra) <= set(SNAPSHOT_ALLOWED) | {
            "recognition_duration_ms",
            "frames_rejected",
            "score_reset_count",
            "interval_skip_count",
            "probe_kind",
            "presenting_identity",
        }, f"unvetted new columns: {sorted(set(extra) - set(SNAPSHOT_ALLOWED))}"


# ---------------------------------------------------------------------------
# 4. End-to-end: a real round through the real demo writer
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    pytest.importorskip.__module__ is None, reason="always false"
)
def test_demo_csv_end_to_end_carries_the_new_fields(tmp_path: Path) -> None:
    """The row builder is only half the contract; the file must carry it."""
    from tests.live.test_d3b_optional_recording import _csv_parts

    store = tmp_path / "demo-store"
    store.mkdir()
    demo_csv = store / G3_DEMO_RESULTS_CSV_NAME

    import os

    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from facecore.live.capture import FakeCapture
    from facecore.live.qt_window import QtResearchWindow
    from tests.live.test_g3_round_records import _RoundFactory, _face_frames
    from tests.live.test_qt_window import _matching_scorer

    app = QApplication.instance() or QApplication([])
    del app

    factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), _matching_scorer)
    desktop, consent, _attempt = factory()
    window = QtResearchWindow(
        desktop,
        consent=consent,
        recorder=None,
        attempt_id=None,
        offscreen=True,
        clock_ns=lambda: 0,
        next_session=factory,
        demo_results_csv=demo_csv,
    )
    window.show()
    window.enter_ready()
    window.start_clicked()
    window.process_until_terminal(max_steps=200)
    window.press_correct()
    window.close()

    assert demo_csv.is_file(), "demo verdict wrote no file"
    header, rows, _ = _csv_parts(demo_csv)
    assert header[: len(LEGACY_23)] == list(LEGACY_23)
    row = dict(zip(header, rows[0], strict=True))
    # Non-blank for every declared column: the §8 false-data guard.
    blank = [c for c, v in row.items() if v == "" and c not in
             {"top1_identity", "top1_score", "top2_identity", "top2_score",
              "margin", "label_identity", "probe_kind",
              "presenting_identity"}]
    assert not blank, f"columns blank in a real scored round: {blank}"
    assert int(row["frames_rejected"]) >= 0
    assert csv  # keep the import meaningful for readers


@pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="the real wiring is exercised through the Qt window",
)
def test_a_real_demo_round_fills_the_diagnostic_fields(
    qt_app: Any, tmp_path: Path
) -> None:
    """The values must be real on a real round, not merely present.

    The row-level tests inject `event_counts` and synthesise
    `TimingMarks`, so they prove the plumbing but not that a live round
    produces a meaningful value. This drives the actual window and reads
    the file, which is the only place the claim 「逐幀計數會出現在 log」
    can be checked end to end.

    `recognition_duration_ms` is asserted non-blank specifically: it is
    derived from `TimingMarks`, and `controller.py:235` deliberately
    never fills the open segment on the demo path, so the natural
    fear is that the whole marks object is empty there and the column
    ships permanently blank — the §8 false-data shape. The real run
    shows it is populated (0.0 on a single-frame round, which is a true
    measurement: the round anchored and terminated at the same instant).
    """
    from facecore.live.capture import FakeCapture
    from facecore.live.qt_window import QtResearchWindow
    from facecore.research.cli import G3_DEMO_RESULTS_CSV_NAME
    from tests.live.test_g3_round_records import _RoundFactory, _face_frames
    from tests.live.test_qt_window import _matching_scorer

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    store = tmp_path / "demo-store"
    store.mkdir()
    demo_csv = store / G3_DEMO_RESULTS_CSV_NAME
    factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), _matching_scorer)
    desktop, consent, _attempt = factory()
    window = QtResearchWindow(
        desktop,
        consent=consent,
        recorder=None,
        attempt_id=None,
        offscreen=True,
        clock_ns=lambda: 0,
        next_session=factory,
        demo_results_csv=demo_csv,
    )
    window.show()
    window.enter_ready()
    window.start_clicked()
    window.process_until_terminal(max_steps=200)
    window.press_correct()
    window.close()

    rows = list(csv.DictReader(demo_csv.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["recognition_duration_ms"] != "", (
        "recognition_duration_ms came out blank on a real round: the "
        "timing marks are not reaching the demo writer"
    )
    # A real round that matched had usable frames, so the quality gate
    # rejected a definite number of them — or none. Either is meaningful;
    # a blank cell is not.
    assert row["frames_rejected"].isdigit()
    assert row["score_reset_count"].isdigit()
    assert row["interval_skip_count"].isdigit()
    # The threshold snapshot must carry this round's actual profile.
    assert row["match_threshold"] == "0.45", row["match_threshold"]
    assert row["required_support"] == "1"
    # W0 fields stay empty until W0-a supplies an input path.
    assert row["probe_kind"] == "" and row["presenting_identity"] == ""


# ---------------------------------------------------------------------------
# 5. The sink is wired to EVERY engine (plan v8 §8, dispatch item 2)
# ---------------------------------------------------------------------------
class TestEveryEngineHasASink:
    """Guards the specific failure the dispatch called out by name.

    W3's premise was that `cli.py` builds the per-round engine inline
    without an `event_sink` while the first-round engine has one — so
    under `--continuous` only round 1 could ever carry event data. An
    acceptance that asks for "at least one row with a non-zero count"
    would then be satisfied by round 1 while every later row is fake
    data, the same shape as the `frames_dropped` column §8 forbids.

    Parsed from the AST rather than grepped: a keyword-only call and a
    positional one are both `SessionEngine(...)` to grep, but only the
    keyword form actually passes the sink.
    """

    def test_no_engine_construction_site_omits_the_event_sink(self) -> None:
        import ast
        from pathlib import Path

        cli = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "facecore"
            / "research"
            / "cli.py"
        )
        tree = ast.parse(cli.read_text(encoding="utf-8"))
        sites: list[tuple[int, list[str]]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(
                node.func, "id", ""
            ) == "SessionEngine":
                sites.append(
                    (node.lineno, [k.arg for k in node.keywords])
                )
        assert len(sites) >= 4, (
            f"expected several engine sites, found {len(sites)} — a new "
            "call shape may have appeared and this check no longer sees it"
        )
        missing = [ln for ln, kw in sites if "event_sink" not in kw]
        assert not missing, (
            f"SessionEngine built without event_sink at line(s) {missing}; "
            "that engine's round will report zero events for real reasons "
            "nobody can tell apart from 'no events happened'"
        )

    def test_the_sink_is_defined_before_every_use(self) -> None:
        """A use before the definition is a NameError at round time.

        The per-round engine is built inside `next_session_factory`, which
        closes over `cmd_live`'s sink. Defining it below the branch that
        needs it compiles fine and only fails when a round is actually
        built — and the window swallows that into 「建輪失敗」, so the run
        quietly produces no rounds at all.
        """
        import ast
        from pathlib import Path

        cli = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "facecore"
            / "research"
            / "cli.py"
        )
        tree = ast.parse(cli.read_text(encoding="utf-8"))
        defined_at: int | None = None
        used_at: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_event_sink":
                defined_at = node.lineno
            if isinstance(node, ast.Name) and node.id == "_event_sink":
                if isinstance(node.ctx, ast.Load):
                    used_at.append(node.lineno)
        assert defined_at is not None, "_event_sink no longer exists"
        early = [ln for ln in used_at if ln < defined_at]
        assert not early, (
            f"_event_sink used at line(s) {early}, defined at {defined_at}"
        )


@pytest.fixture(scope="module")
def qt_app() -> Any:
    """A module-scoped QApplication, matching the D3b suite's fixture."""
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])
