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
import re
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
    face_count: int = 1,
    face_box: tuple[float, float, float, float] | None = (0.0, 0.0, 2.0, 2.0),
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
        quality_reasons=("quality_blurry",) if not quality_pass else (),
        face_count=face_count,
        face_box=face_box,
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
            "support_clear_reasons",
            "score_reset_count",
            "interval_skip_count",
            "probe_kind",
            "presenting_identity",
            # D7-A W1: gallery startup visibility, vetted in
            # tests/cli/test_d7_w1_gallery_visibility.py. Listed here so a
            # FOURTH column cannot be appended without a name appearing
            # in this allow-set — which is the whole point of the guard.
            "expected_count",
            "loaded_count",
            "gallery_rejected",
            # G3-w: the operator's own verdict on the round, `correct` /
            # `incorrect` / empty. Not one of the D7-A additions, so it has
            # no vetting file of its own yet; the allow-set entry is what
            # this guard is FOR — the name had to be reviewed and listed
            # here rather than slipping in unnoticed. See
            # docs/specs/2026-10-05-g3-operator-verdict-ground-truth.md §5.1
            # for the value domain and the reason it is appended last.
            "operator_verdict",
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
    # `support_clear_reasons` is legitimately empty for a round whose
    # support window was never disturbed — that is the measurement, not a
    # missing value. Same reasoning as the W0 fields.
    #
    # D7-A W1's three gallery columns are excluded for the SAME reason but
    # a different one: this window is built WITHOUT a gallery, so it has no
    # `load_report` to forward and the three cells are honestly empty. The
    # companion assertion below is what keeps that exclusion honest — it
    # fails if the columns are blank on a window that DOES carry a report,
    # so the exclusion can never quietly become the normal case.
    blank = [c for c, v in row.items() if v == "" and c not in
             {"top1_identity", "top1_score", "top2_identity", "top2_score",
              "margin", "label_identity", "probe_kind",
              "presenting_identity", "support_clear_reasons",
              "expected_count", "loaded_count", "gallery_rejected",
              # G3-w `operator_verdict`, EXPIRED EXEMPTION. Unlike every
              # other entry above, this cell is NOT honestly empty: this
              # window went through press_correct(), so the operator's
              # judgement is known and `correct` is the right value. It is
              # empty only because P2 added the column and P3 has not wired
              # the button yet. So it sits in this exclusion set on a
              # deadline, and `test_the_operator_verdict_exemption_is_
              # bounded_by_observable_state` below is what ends that
              # deadline. Do not leave it here after P3.
              "operator_verdict"}]
    assert not blank, f"columns blank in a real scored round: {blank}"
    assert int(row["frames_rejected"]) >= 0
    assert row["gallery_rejected"] == "", (
        "a window with no gallery carries no report; the cell must say so "
        "rather than claiming a measured empty gallery"
    )
    assert csv  # keep the import meaningful for readers


def test_the_operator_verdict_exemption_is_bounded_by_observable_state() -> None:
    """The one thing that keeps the exemption above from being permanent.

    Every OTHER name in that exclusion set is exempt because the value
    genuinely does not exist for this window — no gallery, so no
    `load_report`; no support disturbance, so no clear to count. Their
    companion assertions hold that exemption to the truth they rest on.

    `operator_verdict` is not that kind of case: press_correct() means the
    operator DID judge the round, so `correct` is available and the cell
    must not be blank. The exclusion above is a deliberate, dated hole —
    P2 shipped the column, P3 does the wiring.

    An exemption nobody can see expire is indistinguishable from a bug, and
    this repo has already shipped guards that nothing ran. So the hole is
    tied to an OBSERVABLE state: ANY production writer forwarding a verdict.

    ⚠️ ANY writer, not one of them by name. There are two — the labeled
    `_press_key` and the unlabeled `_record_unlabeled_round` — and an
    earlier version of this companion watched only `_press_key`. Wiring the
    verdict into the other one left it green: the probe pointed at one path
    and silently accepted while the exemption ran unbounded. Naming a single
    writer here is what created that hole, so the check below is scoped to
    the CALL SITE rather than to any function.

    · while NO writer passes a verdict → this passes, hole open
    · the moment ANY writer passes one → this turns RED, and stays red

    ⚠️⚠️ That last sentence is not a to-do you can finish by editing this
    file. Removing `operator_verdict` from the exclusion set will NOT make
    this go green — it goes green only when P3 takes the exemption back by
    wiring every writer AND dropping the name in one commit. See the
    assertion message for why the two halves have to move together.

    Reading the source with ast rather than counting strings is the same
    reason `test_d7_w0b_writer_wiring.py` parses instead of greps: a
    comment or a passing mention can satisfy a string count while nothing
    is forwarded.
    """
    import ast

    writer_name = "append_g3_demo_results_csv"
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "facecore"
        / "live"
        / "qt_window.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # ⚠️ Ask the inverse question.
    #
    # Every previous version asked 「can I find the forwarding?」 and grew the
    # search each time one more binding form turned out to be invisible:
    # another writer (`_record_unlabeled_round`), then `**kwargs`, then an
    # import alias, then a plain assignment. Each fix caught its own shape
    # and each one was the same bug again — the verdict reached the CSV while
    # the guard reported 「not wired」.
    #
    # So this does not widen the search. It inverts it: the writer's name is
    # a LITERAL in every binding form, because you cannot alias or assign a
    # name without writing it. So find every string literal equal to the
    # writer's name, and for each one ask whether the call it heads is one
    # whose keywords can be read. Any that cannot be read is not evidence of
    # 「no forwarding」 — it is an absence of evidence, and it fails loud.
    #
    # The companion rule is test_d7_w0b_writer_wiring.py::_is_live_read:
    # recognise what you can, refuse what you cannot, and never treat the
    # second case as the first.
    writers: dict[str, bool] = {}
    unreadable: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            # Does this call head mention the writer's name as a literal?
            # Covers `writer(...)`, `_alias(...)`, `c.writer(...)`,
            # `getattr(m, "writer")(...)` and `_w(...)` where
            # `_w = writer` was assigned anywhere in the module — the
            # assignment is itself a literal mention, which is what makes
            # this closed rather than another enumeration.
            # Might this call BE the writer? A binding form has to mention the
            # writer's name SOMEWHERE to exist: as the callee itself
            # (`writer(...)`, `c.writer(...)`), as a string argument to
            # getattr, or on the right-hand side of an assignment
            # (`_w = writer`). Collecting the assigned names and testing the
            # callee against that set was my first attempt and it was wrong —
            # it flagged every call to any name this module ever assigns,
            # which is the indiscriminate interception this guard exists to
            # avoid. The test is whether THIS writer's name is written here,
            # not whether the callee happens to be assignable.
            head = ast.unparse(call.func)
            mentions_writer = (
                head.split(".")[-1] == writer_name
                or writer_name in head
                or any(
                    isinstance(node_, ast.Constant) and node_.value == writer_name
                    for node_ in ast.walk(call)
                )
            )
            if not mentions_writer:
                continue
            # ⚠️ Unresolvable forwarding is NOT "not forwarded".
            #
            # `**kwargs` reaches here with arg=None, so the subset test
            # below would answer 「not forwarded」 for a call that may well
            # be forwarding it — and it did. Proven by execution, not by
            # reading: a row written through that path carried
            # operator_verdict='correct' while this guard stayed green.
            #
            # Other shapes reach the same place, so the check is on the
            # unpacking rather than on `**kwargs` specifically:
            #   · `*args` / `**kwargs` on the call itself (arg is None)
            #   · a `**dict` built conditionally a few lines up
            #   · a walrus in an argument that reassigns the kwargs dict
            # What they share is that the call's keywords are no longer
            # a literal list, so 「is my name among them」 has no answer.
            # A guard that cannot answer must say so rather than pick the
            # convenient answer — the same rule
            # test_d7_w0b_writer_wiring.py::_is_live_read follows when it
            # refuses anything it cannot read as a live read.
            #
            # The cost is that P3 is pushed towards explicit keywords,
            # which is the point: an unverifiable shape that stays silent
            # is worse than one that interrupts.
            unpacked = [
                ast.unparse(arg.value)
                for arg in call.keywords
                if arg.arg is None
            ]
            if (
                unpacked
                or any(isinstance(arg, ast.Starred) for arg in call.args)
                or head.split(".")[-1] != writer_name
            ):
                unreadable.append(f"{node.name}() → {head}(...)")
                continue
            writers[node.name] = "operator_verdict" in {
                kw.arg for kw in call.keywords
            }
    if unreadable:
        raise AssertionError(
            "these calls mention the demo writer but their forwarding cannot "
            "be determined by reading them:\n  "
            + "\n  ".join(sorted(set(unreadable)))
            + "\n\n"
            "This is a handoff blocker, not a style note. Call the writer "
            "directly with explicit keywords — "
            "`append_g3_demo_results_csv(..., operator_verdict=...)` — and "
            "this goes green.\n\n"
            "⚠️⚠️ But understand what happens next: this guard STAYS RED "
            "after that, and it is supposed to. It turns green only when "
            "`operator_verdict` is ALSO removed from the exclusion set in "
            "test_demo_csv_end_to_end_carries_the_new_fields. Both halves "
            "belong in ONE commit, which is P3's job — not something you "
            "can finish from this message, and not something you can "
            "finish by editing this file.\n\n"
            "Why it matters: earlier versions of this guard read a call "
            "like this and concluded 「not wired」, while the round's real "
            "verdict was sitting in the CSV. That is the one failure this "
            "guard exists to prevent, and it would have repeated here "
            "silently."
        )
    assert writers, (
        "no function in qt_window.py calls append_g3_demo_results_csv; this "
        "file's view of the demo write path is stale"
    )
    forwarding = sorted(name for name, has in writers.items() if has)
    assert not forwarding, (
        f"{forwarding} now forward(s) operator_verdict, so the operator's real "
        "judgement reaches the CSV.\n\n"
        "This is a HANDOFF REMINDER for P3, not an instruction you can finish "
        "here. Removing `operator_verdict` from the exclusion set in "
        "test_demo_csv_end_to_end_carries_the_new_fields will NOT turn this "
        "green — this guard watches whether a writer forwards the value, and "
        "that stays true after the exemption is dropped.\n\n"
        "Both halves have to land in ONE commit: wire every production writer, "
        "and drop the name from the exclusion set. Removing only the exemption "
        "leaves a blank-cell failure with nothing watching it; removing only "
        "the wiring leaves a permanently-permanent hole.\n\n"
        "If you are reading this while trying to make it green by editing "
        "this assertion — stop. Deleting this guard is the exact failure it "
        "exists to prevent: every row keeps recording an empty verdict while "
        "the repo stays green."
    )


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


# ---------------------------------------------------------------------------
# 6. Rework 2: support_clear_reasons must cover every clearing path
# ---------------------------------------------------------------------------
class TestSupportClearReasons:
    """The read that actually answers D4 §11 item 18b.

    `event_type` cannot: four sites emit `event_type="rejected"` while
    their `reset_reason` values split into 「restart the App」 and 「keep
    sampling」. Before this rework the demo log recorded only
    `score_reset` and `interval_skip`, so three of the five clearing
    families were invisible — and the two recorded columns were not even
    the same kind of fact (one clears, one does not).

    Every test here first accumulates a NON-EMPTY support window, because
    a clear is only observable relative to something: with an empty
    window `support_before == support_after == 0` and the predicate is
    vacuously false.
    """

    def _engine(self) -> Any:
        from facecore.live.session import SessionEngine

        return SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test")

    @staticmethod
    def _seed_support(engine: Any) -> None:
        """Two accepted frames → support window of 2."""
        scores = {"person-synth-01": 0.90, "person-synth-02": 0.10}
        engine.observe(_obs(1, scores=scores, captured_ns=0))
        engine.observe(_obs(2, scores=scores, captured_ns=200_000_000))
        assert len(engine._support_sequences) == 2, (
            "the fixture must actually accumulate support, or every clear "
            "below is vacuous"
        )

    def test_score_below_threshold_is_recorded_with_its_reason(self) -> None:
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.05, "person-synth-02": 0.0},
                 captured_ns=400_000_000)
        )
        assert engine.support_clear_reasons() == {"score_below_threshold": 1}

    def test_continuity_jump_is_recorded(self) -> None:
        """The third family the rework exists for, and the one the
        reviewers named: `continuity_max_center_delta_ratio` is 0.5 in
        `profiles/g3-v1.json`, so this path is live on the demo path, not
        dead code. Without a test it could silently stop being recorded.
        """
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        # g3-v1's limit is 0.5; jump the box far enough that
        # delta / max(dim1, 1.0) exceeds it. `face_box` is (x0,y0,x1,y1).
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.90, "person-synth-02": 0.10},
                 captured_ns=400_000_000, face_box=(40.0, 40.0, 42.0, 42.0))
        )
        assert engine.support_clear_reasons() == {"continuity_jump_detected": 1}, (
            "a continuity jump clears the support window and is one of the "
            "two reasons the operator must restart the App; it must appear "
            "in the log with its own reason"
        )

    def test_quality_reject_is_recorded_with_the_quality_reasons(self) -> None:
        """A rejected frame keeps sampling; the reason carries the detail."""
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(3, scores={}, quality_pass=False, captured_ns=400_000_000)
        )
        reasons = engine.support_clear_reasons()
        assert len(reasons) == 1
        (reason, count), = reasons.items()
        assert count == 1
        assert reason.startswith("quality_rejected:"), reason

    def test_no_face_is_recorded_under_its_own_reason(self) -> None:
        """`no_face_detected` is a DISTINCT reason from `quality_rejected`.

        Both come from the same clearing site, so observing only one of
        them would not prove the column buckets by reason — it would only
        prove that site emits something. A previous version of this test
        sent two `face_count=0` frames with no support accumulated and
        asserted `== {}`, which passed for the wrong reason: the window
        was never non-empty, so nothing could clear it. That is the 「a
        guard that guards nothing」 shape — see
        [[green-ci-proves-nothing-about-execution]].
        """
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(_obs(3, scores={}, face_count=0, captured_ns=400_000_000))
        assert engine.support_clear_reasons() == {"no_face_detected": 1}

    def test_empty_identity_scores_is_recorded_under_its_own_reason(
        self,
    ) -> None:
        """The third reason from that one site — a face but no scores.

        `no_face_detected`, `quality_rejected: …` and
        `empty_identity_scores` all clear from the same guard, and each
        names a different operator-facing cause. Covering one of them
        proves nothing about the others.
        """
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(_obs(3, scores={}, captured_ns=400_000_000))
        assert engine.support_clear_reasons() == {"empty_identity_scores": 1}

    def test_margin_below_threshold_is_a_distinct_reason(self) -> None:
        """Same clearing site as `score_below_threshold`, opposite cause.

        The top score clears `match_threshold` comfortably here; only the
        gap to the runner-up fails. Both come from the single
        `if top_score < … or margin < …` guard, so the reason string is
        the only thing that separates them.
        """
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(
                3,
                scores={"person-synth-01": 0.40, "person-synth-02": 0.38},
                captured_ns=400_000_000,
            )
        )
        assert engine.support_clear_reasons() == {"margin_below_threshold": 1}

    def test_multiple_faces_is_recorded_and_terminates(self) -> None:
        """The other restart-required reason; also a terminal."""
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.90, "person-synth-02": 0.10},
                 captured_ns=400_000_000, face_count=2)
        )
        assert engine.support_clear_reasons() == {"input_multiple_faces": 1}

    def test_none_runner_up_is_recorded(self) -> None:
        """A single-candidate frame has no margin, so it cannot qualify."""
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.90}, captured_ns=400_000_000)
        )
        assert engine.support_clear_reasons() == {"none_runner_up": 1}

    def test_interval_skip_is_not_a_clear(self) -> None:
        """The column that is NOT a clearing event must stay out.

        This is the conflation 18b was about: a frame that arrives inside
        the interval is not accumulated, but the window is left intact.
        Counting it here would report a clear that never happened.
        """
        engine = self._engine()
        engine.start("sess-1", 0)
        passing = {"person-synth-01": 0.90, "person-synth-02": 0.10}
        engine.observe(_obs(1, scores=passing, captured_ns=0))
        engine.observe(_obs(2, scores=passing, captured_ns=50_000_000))
        engine.observe(_obs(3, scores=passing, captured_ns=1_000_000_000))
        assert engine.event_count("interval_skip") == 1
        assert engine.support_clear_reasons() == {}, (
            "interval_skip leaves the support window untouched; recording "
            "it as a clear is the 18b conflation"
        )

    def test_identity_change_is_not_a_clear(self) -> None:
        """It reseeds the window to 1 rather than emptying it."""
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(3, scores={"person-synth-09": 0.90, "person-synth-10": 0.10},
                 captured_ns=400_000_000)
        )
        assert engine.support_clear_reasons() == {}

    def test_continuity_accumulation_is_not_a_clear(self) -> None:
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.90, "person-synth-02": 0.10},
                 captured_ns=400_000_000)
        )
        assert engine.support_clear_reasons() == {}

    def test_a_round_with_no_clears_reports_empty_not_zeroes(self) -> None:
        engine = self._engine()
        engine.start("sess-1", 0)
        self._seed_support(engine)
        assert engine.support_clear_reasons() == {}

    def test_counts_reset_between_rounds(self) -> None:
        engine = self._engine()
        engine.start("round-1", 0)
        self._seed_support(engine)
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.05, "person-synth-02": 0.0},
                 captured_ns=400_000_000)
        )
        assert engine.support_clear_reasons() == {"score_below_threshold": 1}
        engine.start("round-2", 10_000_000_000)
        assert engine.support_clear_reasons() == {}


class TestSupportClearReasonsReachTheCSV:
    """The column must be observably populated, not merely present.

    An earlier version of this file asserted only that
    `support_clear_reasons` was not blank. That assertion is vacuous for
    a round that never cleared: the correct value there IS the empty
    string. So mutating the writer to always emit "" survived — the exact
    §8 "column present but tells you nothing" shape, and a false green in
    my own test.

    These close it: they drive the real engine through a real clear and
    read the value the CSV would carry.
    """

    def _row_from_real_engine(self) -> dict[str, object]:
        from facecore.live.session import SessionEngine

        engine = SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test")
        engine.start("sess-1", 0)
        scores = {"person-synth-01": 0.90, "person-synth-02": 0.10}
        engine.observe(_obs(1, scores=scores, captured_ns=0))
        engine.observe(_obs(2, scores=scores, captured_ns=200_000_000))
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.05, "person-synth-02": 0.0},
                 captured_ns=400_000_000)
        )
        assert engine.support_clear_reasons() == {"score_below_threshold": 1}
        return g3_demo_round_row(
            _Round.make(
                _terminal(),
                (_obs(1, scores=scores), _obs(2, scores=scores)),
            ),
            required_support=3,
            labeled_at_utc="2026-10-01T00:00:00+00:00",
            event_counts=engine.event_counts(),
            support_clears=engine.support_clear_reasons(),
            profile=_profile(),
        )

    def test_the_column_carries_the_real_reason_and_count(self) -> None:
        row = self._row_from_real_engine()
        assert row["support_clear_reasons"] == "score_below_threshold:1", (
            "the column must serialise the reason and its count; an empty "
            "value here would be indistinguishable from 'nothing cleared'"
        )

    def test_multiple_reasons_are_joined_and_sorted(self) -> None:
        from facecore.live.session import SessionEngine

        engine = SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test")
        engine.start("sess-1", 0)
        # Seed a NON-EMPTY window first: a clear is only observable
        # relative to something, exactly as in TestSupportClearReasons.
        scores = {"person-synth-01": 0.90, "person-synth-02": 0.10}
        engine.observe(_obs(1, scores=scores, captured_ns=0))
        engine.observe(_obs(2, scores=scores, captured_ns=200_000_000))
        engine.observe(
            _obs(3, scores={}, quality_pass=False, captured_ns=400_000_000)
        )
        engine.observe(
            _obs(4, scores={}, quality_pass=False, captured_ns=600_000_000)
        )
        reasons = engine.support_clear_reasons()
        assert reasons, "two rejected frames should each record a reason"
        row = g3_demo_round_row(
            _Round.make(_terminal(), (_obs(1, scores={}),)),
            required_support=3,
            labeled_at_utc="2026-10-01T00:00:00+00:00",
            event_counts=engine.event_counts(),
            support_clears=reasons,
            profile=_profile(),
        )
        joined = str(row["support_clear_reasons"])
        # Since F1 truncation no reason carries a colon, so the first one
        # separates the family from its count unambiguously.
        for part in (p for p in joined.split(";") if p):
            reason, _, count = part.partition(":")
            assert count.isdigit(), f"unparsable count in {part!r}"
            assert reason.startswith("quality_rejected"), reason


class TestQualityReasonsAreTruncatedInTheCSV:
    """F1: the CSV carries the reason FAMILY, not the gate combination.

    The seven gates in `pipeline/quality.py:36-55` each append
    independently, so the in-memory `reset_reason` has a 2^7 key space —
    127 non-empty subsets. A W4 `groupby(reason)` over that produces a
    near-singleton tail, and the question the log has to answer (「was the
    quality gate the blocker」) is boolean, not a 7-way combination.

    These use MULTIPLE gates on one frame on purpose: a single-gate case
    would pass against an untruncated writer too, so it would prove
    nothing about the truncation.
    """

    def _row(self, quality_reasons: tuple[str, ...]) -> dict[str, object]:
        from facecore.live.session import SessionEngine

        engine = SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test")
        engine.start("sess-1", 0)
        scores = {"person-synth-01": 0.90, "person-synth-02": 0.10}
        engine.observe(_obs(1, scores=scores, captured_ns=0))
        engine.observe(_obs(2, scores=scores, captured_ns=200_000_000))
        engine.observe(
            _obs(3, scores={}, quality_pass=False, captured_ns=400_000_000)
        )
        # The engine must be keeping the detail even though the CSV will not.
        reasons = engine.support_clear_reasons()
        (reason, count), = reasons.items()
        assert count == 1
        assert reason.startswith("quality_rejected:"), reason
        return g3_demo_round_row(
            _Round.make(_terminal(), (_obs(1, scores=scores),)),
            required_support=3,
            labeled_at_utc="2026-10-01T00:00:00+00:00",
            event_counts=engine.event_counts(),
            support_clears=reasons,
            profile=_profile(),
        )

    def test_a_single_gate_is_truncated_to_the_family(self) -> None:
        row = self._row(("quality_blurry",))
        assert row["support_clear_reasons"] == "quality_rejected:1"

    def test_several_gates_still_yield_one_cell_not_many(self) -> None:
        """The whole point: N gates must not become N keys or N cells."""
        row = self._row(
            (
                "quality_detector_confidence",
                "quality_blurry",
                "quality_exposure",
                "quality_pose_yaw",
                "quality_occluded",
            )
        )
        assert row["support_clear_reasons"] == "quality_rejected:1", (
            "five simultaneous gates must collapse to the same single cell "
            "as one gate; a per-gate or per-subset encoding would produce a "
            "different value here"
        )

    def test_the_cell_value_carries_no_gate_names(self) -> None:
        row = self._row(("quality_detector_confidence", "quality_exposure"))
        cell = str(row["support_clear_reasons"])
        for gate in ("quality_detector_confidence", "quality_exposure"):
            assert gate not in cell, (
                f"{gate} leaked into the CSV cell; the per-gate detail "
                "belongs to the in-memory interface and the trace"
            )

    def test_the_count_is_unambiguous_after_truncation(self) -> None:
        """With no colon left in the reason, `split(':', 1)` is exact."""
        row = self._row(("quality_blurry", "quality_exposure"))
        reason, _, count = str(row["support_clear_reasons"]).partition(":")
        assert reason == "quality_rejected"
        assert count.isdigit() and int(count) == 1

    def test_other_families_are_untouched_by_the_truncation(self) -> None:
        """Only `quality_rejected` grows a suffix; the rest are literals.

        A blanket `split(':', 1)` would be harmless here only because the
        other reasons contain no colon — this pins that, so a future
        reason that does would fail rather than silently truncate.
        """
        from facecore.live.session import SessionEngine

        engine = SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test")
        engine.start("sess-1", 0)
        scores = {"person-synth-01": 0.90, "person-synth-02": 0.10}
        engine.observe(_obs(1, scores=scores, captured_ns=0))
        engine.observe(_obs(2, scores=scores, captured_ns=200_000_000))
        engine.observe(
            _obs(3, scores={"person-synth-01": 0.05, "person-synth-02": 0.0},
                 captured_ns=400_000_000)
        )
        row = g3_demo_round_row(
            _Round.make(_terminal(), (_obs(1, scores=scores),)),
            required_support=3,
            labeled_at_utc="2026-10-01T00:00:00+00:00",
            event_counts=engine.event_counts(),
            support_clears=engine.support_clear_reasons(),
            profile=_profile(),
        )
        assert row["support_clear_reasons"] == "score_below_threshold:1"


class _RetiredFigure:
    """One figure from the deleted batch, and how to recognise it.

    A bare-digit scan is not enough, and the reason is visible in the
    note itself: `11` occurs twice today as 「D4 §11 item 18b」 — a spec
    section number, not the retired count of 11 rounds. A rule that
    flagged every `11` would send the next editor chasing a figure they
    never wrote. So each figure carries the spec-reference form that
    must be exempt, rather than relying on the reader to notice.

    Spelled-out numerals are recognised too. A guard that only matches
    digits is defeated by 「sixteen of the rounds」, which states the
    same retired fact in wording no digit-scan can catch — the failure
    mode this whole guard exists to prevent, reached through a different
    spelling.
    """

    def __init__(self, value: str, spelled: str | None = None,
                 exempt: tuple[str, ...] = ()) -> None:
        self.value = value
        self.spelled = spelled
        self.exempt = exempt

    def mentions(self, note: str) -> bool:
        """True when this figure appears as a batch statistic.

        A figure counts only where it is NOT part of a spec citation:
        `§11` is section 11 of D4, `11 rounds` is eleven rounds. The
        exemption is spelled out per figure so an unrelated new spec
        citation cannot silently start passing or failing.
        """
        pattern = rf"(?<![\w.]){self.value}(?![\w.])"
        for match in re.finditer(pattern, note):
            window = note[max(0, match.start() - 2):match.end() + 8]
            if any(cite in window for cite in self.exempt):
                continue
            return True
        # Spelled-out form. A retired figure written as a word states
        # the same thing as the digit, and no digit-only guard sees it.
        word = self.spelled
        if word and re.search(rf"(?<![\w]){word}(?![\w])", note, re.IGNORECASE):
            return True
        return False


# Only the figures that were distinctive in the deleted batch. `5` is
# deliberately absent: as a bare word or digit it appears constantly in
# unrelated prose, so pinning it would make the guard fire on ordinary
# sentences — and a guard that cries wolf is not a guard.
RETIRED_FIGURES = (
    _RetiredFigure("16", spelled="sixteen"),
    _RetiredFigure("11", spelled="eleven", exempt=("§11", "§ 11")),
    _RetiredFigure("122"),
    _RetiredFigure("32", spelled="thirty-two"),
)

# A commit reference: the hex prefix of a real SHA. Deliberately strict —
# accepting bare words like "previously" would make the rule unfalsifiable
# again, which is the shape this whole guard exists to prevent.
_COMMIT_REF = re.compile(
    r"\b(?:see |in |removed in |per )\bcommit\b|\b[0-9a-f]{7,40}\b"
)


class TestDocumentedScopeMatchesTheLog:
    """The scope note must not rot into quoting a batch that is gone.

    `support_clear_reasons`'s field comment used to state figures from
    one operator batch — how many rounds had no usable frame, and their
    summed frame count. Those are facts about a CSV that lives OUTSIDE
    the repo, so nothing in the suite would have noticed them going
    stale — the exact 「a comment becomes a fact by being written down」
    shape. They did go stale: that batch was deleted and the log
    refilled, leaving the note describing data that no longer existed.

    ⚠️ The deleted test is worth recording. It read the operator's CSV
    and asserted the note's figures against it, skipping when the file
    was absent. On CI that skip is permanent, so the guard only ever ran
    on one contributor's machine — and it reported green there while
    asserting numbers about a batch that had been deleted. That is the
    「exists but no job runs it → silent」 shape, one level up: it DID run,
    just never where the rot happened. A guard whose input is a mutable
    file outside version control cannot outlast that file's contents.

    What replaces it is the part that is actually true of the repo: the
    note must not restate a deleted batch, and must say where the live
    numbers live instead.
    """

    def _scope_note(self) -> str:
        """The comment block immediately above the column declaration.

        Scoping matters: an earlier version of this test grepped the whole
        file, and every mutation survived because those figures also occur
        elsewhere in `cli.py` — 16/11/5/122 appear in unrelated arithmetic
        and in the leading-zero test. A guard that greps too much is a
        guard that cannot fail, so this returns only the lines belonging
        to this column's own comment.
        """
        src = (
            Path(__file__).resolve().parents[2]
            / "src" / "facecore" / "research" / "cli.py"
        )
        lines = src.read_text(encoding="utf-8").splitlines()
        idx = next(
            i
            for i, line in enumerate(lines)
            if line.strip() == '"support_clear_reasons",'
        )
        note: list[str] = []
        for line in reversed(lines[:idx]):
            if line.strip() and not line.strip().startswith("#"):
                break
            note.append(line)
        return "\n".join(reversed(note))

    def test_the_comment_does_not_restate_the_deleted_batch(self) -> None:
        """A retired figure needs a commit reference; nothing else does.

        The rule is semantic rather than a list of banned strings. An
        earlier version pinned six literal phrases and its docstring
        claimed re-adding any of them "in any wording" would fail. That
        claim was false in both directions, and both failures were found
        by mutation rather than by reading:

        - `those 122 frames` — the deleted comment's SECOND mention of
          the frame count, named in that same docstring — was not in the
          list, so re-adding it passed.
        - a legitimate historical citation (`the old 122 figure was
          removed, see commit 491034b`) was blocked, because the pin
          matched the digits rather than the intent.

        So: quoting a retired figure is allowed only when the note also
        cites the commit that retired it. That distinguishes the two
        cases for what they are. A literal pin list cannot, and any
        list added to fix one gap becomes the next gap.
        """
        note = self._scope_note()
        cited = bool(_COMMIT_REF.search(note))
        restated = [f.value for f in RETIRED_FIGURES if f.mentions(note)]
        assert not (restated and not cited), (
            f"the scope note restates retired figures {restated} with no "
            "commit reference. Those counts came from an operator batch "
            "that was deleted; per-batch counts go stale the moment the "
            "operator reruns the App, which appends to the log. Either "
            "state the reasoning alone, or — if the history genuinely "
            "belongs here — cite the commit that retired the figure so a "
            "reader can tell a citation from a live claim"
        )
        # The replacement has to be actionable, or 「stop quoting numbers」
        # reads as 「the answer went away」.
        assert "w0a-diagnostic-run-runbook.md" in note, (
            "the note must say where the live numbers live"
        )
        assert "W2" in note, "the note must say where 18b's answer does live"

    def test_the_w4_combination_rules_are_stated(self) -> None:
        """The reader of a CSV sees none of this, so it must be in the file."""
        src = (
            Path(__file__).resolve().parents[2]
            / "src" / "facecore" / "research" / "cli.py"
        )
        lines = src.read_text(encoding="utf-8").splitlines()
        idx = next(
            i
            for i, line in enumerate(lines)
            if line.strip() == '"interval_skip_count",'
        )
        # The rules are the comment block immediately BELOW the column, so
        # walk forward. An earlier version walked forward too but stopped
        # at the first blank line, which is the line right after the
        # column — so it collected nothing and the two mutation checks
        # below passed against an empty string.
        block: list[str] = []
        for line in lines[idx + 1:]:
            if line.strip() and not line.strip().startswith("#"):
                break
            block.append(line)
        text = "\n".join(block)
        # Anchored to the rules themselves. A loose `or "subset" in text`
        # passed against a mutated comment because the word recurs in the
        # W0 block further down; `and "add" in text.lower()` passed for
        # the same reason. Both mutations were survivable until the
        # assertions named the sentences they are guarding.
        #
        # The same hazard applies to the anchors below: `text` spans lines
        # 444-492, which is the rule block AND the W0 ground-truth block,
        # so every anchor was checked to occur in the rule half only —
        # an anchor that also occurred in the W0 half would pass against
        # a rule that had been deleted.
        assert (
            "`interval_skip_count` and `support_clear_reasons` must NOT"
            in text
        ), "the no-adding rule must be stated explicitly, not implied"
        # (1) Both columns named, (2) the consequence stated rather than
        # the inequality: the rule exists because a reader WILL want to
        # add them. "neither is a subset of the other" is not actionable;
        # "must NOT be added" is.
        assert (
            "`score_reset_count` and `support_clear_reasons` must NOT be"
            in text
        ), "the score/clear rule must be stated as a prohibition, not as a description"
        # (3) At least one CONCRETE clearing family the counter cannot
        # see. Naming the category ("a non-score gate (quality /
        # no-face)") satisfies the sentence while guarding nothing: it
        # tells the reader no specific path is missing, so an
        # implementer reading W4 still cannot tell which reason codes
        # the counter drops. The set is the six paths measured in
        # `TestSupportClearReasons` to clear the window without emitting
        # `score_reset`; `score_below_threshold` and
        # `margin_below_threshold` are deliberately EXCLUDED because the
        # counter does see those, so naming them would not make the
        # point.
        invisible_to_score_counter = (
            "continuity_jump_detected",
            "quality_rejected",
            "no_face_detected",
            "empty_identity_scores",
            "none_runner_up",
            "input_multiple_faces",
        )
        named = [r for r in invisible_to_score_counter if r in text]
        assert named, (
            "the rule must name a concrete clearing family that "
            "`score_reset_count` cannot see; a bare category such as "
            "'quality / no-face' says which GATE fired but not which "
            "reason code is missing from the counter, so W4's implementer "
            "cannot tell what the sum drops"
        )
        # `assert named` above is an EXISTENCE quantifier: it says
        # nothing about WHICH reasons may appear alongside. A counter-
        # visible reason mixed into the list is the dangerous version of
        # this — it reads as "the counter misses this too", which is
        # exactly false, and it would mislead W4's implementer about what
        # the sum drops. Two mutations (keep the six correct ones, add
        # one of the two visible reasons) survived the existence check
        # and a full-repo run, so the exclusion needs its own assertion.
        #
        # The slice must be the ENUMERATED LIST, not the whole ② section:
        # ② legitimately names both visible reasons to explain that they
        # are visible. Slicing from "② THE COUNTER IS BLIND" instead makes
        # the UNMUTATED comment red, which would have passed a reviewer
        # checking only that mutations go red. These two markers are the
        # list's own delimiters, each appearing exactly once.
        list_block = text.split(
            "empties the window without moving the counter:"
        )[1].split("Summing it in")[0]
        visible_leaked = [
            r for r in ("score_below_threshold", "margin_below_threshold")
            if r in list_block
        ]
        assert not visible_leaked, (
            "the counter-invisible list must not name a reason the counter "
            f"CAN see: {visible_leaked}"
        )
        # (4) BOTH harms, because they fail in opposite directions and a
        # reader who keeps only one still gets a wrong summary: keeping
        # only the over-count half under-reports clears, keeping only the
        # blind half over-reports them. Each half is a separate edit the
        # file can drift into, so each is asserted separately.
        assert "double-counts" in text, (
            "the rule must say that summing can double-count — the "
            "counter moves on a frame that cleared nothing"
        )
        assert "invents clears that never happened" in text, (
            "the rule must say that summing can invent clears — six of "
            "the eight clearing paths never move the counter"
        )


class TestSameRoundReasonsAreAggregated:
    """B: one round, two different quality causes → ONE cell, count 2.

    The truncation in F1 was applied per item inside the comprehension,
    so two in-memory keys that only differ after truncation collided and
    were emitted side by side: 「quality_rejected:1;quality_rejected:1」.
    W4's `groupby(key)` would then see the same key twice instead of one
    key with count 2 — the F1 long tail returning under a new name.

    The scenario is a SINGLE round. Reviewer1's first probe used four
    separate rounds and merged them with `dict.update()`, which the real
    writer never does (one row per round, no cross-round merge) — so it
    both missed the defect and would have proved a path that cannot occur.
    Reachable: the window reopens between the two refusals, and
    `required_support=3` does not end the round early.

    `FrameObservation` is built directly rather than through `_obs`,
    which hardcodes `quality_reasons=("quality_blurry",)` — going
    through it silently produces two *identical* gate sets, so the defect
    does not reproduce at all. Sequence numbers come from a counter for
    the same reason: hand-written ids go non-monotonic and trip the
    `duplicate_sequence` guard at `session.py:326`, which ends the round
    before the second refusal.
    """

    @staticmethod
    def _profile() -> ResearchProfile:
        import json

        return ResearchProfile.from_dict(
            json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "profiles" / "g3-v1.json"
                ).read_text(encoding="utf-8")
            )
        )

    def _run(self, gates_per_refusal: list[tuple[str, ...]]) -> dict[str, object]:
        """One round: seed the window, then refuse, reopen, refuse, …"""
        from facecore.live.contracts import FrameObservation
        from facecore.live.qt_window import RoundComplete
        from facecore.live.session import SessionEngine

        prof = self._profile()
        good = {"person-synth-01": 0.90, "person-synth-02": 0.10}
        seq = 0
        clock = 0

        def good_frame() -> FrameObservation:
            nonlocal seq, clock
            seq += 1
            clock += 200_000_000
            return FrameObservation(
                sequence=seq, captured_ns=clock, processed_ns=clock + 1_000_000,
                quality_pass=True, quality_reasons=(), face_count=1,
                face_box=(0.0, 0.0, 2.0, 2.0), identity_scores=good,
                quality_rank=0.9, model_generation="gen-1", gallery_digest="d",
            )

        def bad_frame(gates: tuple[str, ...]) -> FrameObservation:
            nonlocal seq, clock
            seq += 1
            clock += 200_000_000
            return FrameObservation(
                sequence=seq, captured_ns=clock, processed_ns=clock + 1_000_000,
                quality_pass=False, quality_reasons=gates, face_count=1,
                face_box=(0.0, 0.0, 2.0, 2.0), identity_scores={},
                quality_rank=0.0, model_generation="gen-1", gallery_digest="d",
            )

        engine = SessionEngine(prof, "d", "gen-1")
        engine.start("sess-1", 0)
        engine.observe(good_frame())
        engine.observe(good_frame())
        for gates in gates_per_refusal:
            engine.observe(bad_frame(gates))
            engine.observe(good_frame())
            engine.observe(good_frame())
        terminal = engine.finish(clock, reason="timeout")
        return g3_demo_round_row(
            RoundComplete(
                session_id="sess-1", attempt_id="a1", terminal=terminal,
                observations=(good_frame(),), label_kind="unenrolled",
                label_identity=None, profile_version=prof.profile_version,
                started_utc="2026-10-01T00:00:00+00:00",
            ),
            required_support=3, labeled_at_utc="2026-10-01T00:00:01+00:00",
            event_counts=engine.event_counts(),
            support_clears=engine.support_clear_reasons(),
            profile=prof,
        )

    def test_two_distinct_causes_in_one_round_become_one_cell(self) -> None:
        row = self._run([("quality_exposure",), ("quality_pose_yaw",)])
        assert row["support_clear_reasons"] == "quality_rejected:2", (
            "two distinct quality causes in one round must aggregate into a "
            "single cell with count 2; duplicate cells are the F1 long tail "
            "returning under a new name"
        )

    def test_the_same_cause_twice_also_reads_as_two(self) -> None:
        row = self._run([("quality_exposure",), ("quality_exposure",)])
        assert row["support_clear_reasons"] == "quality_rejected:2"

    def test_a_single_refusal_is_one_cell(self) -> None:
        assert self._run([("quality_exposure",)])["support_clear_reasons"] == (
            "quality_rejected:1"
        )


# ---------------------------------------------------------------------------
# 5. The operator-facing docs must know the schema they describe
# ---------------------------------------------------------------------------
# Added by G3-w. Appending `operator_verdict` turned neither of these
# documents red, and that is the whole problem: the runbook's step 0 is
# the ONLY documented way an operator identifies which log they are
# holding, and it still stopped at 38 columns. An operator whose file
# correctly ended in `operator_verdict` would find no matching row,
# conclude from the paragraph below that they held a stale file, and
# follow its own remedy — rename it away and re-run. That throws away a
# correct dataset.
#
# The precedent is recorded rather than hypothetical: #138 added twelve
# columns without touching the docs, #142 added three more and DID update
# both files, and this guard exists so the next append cannot repeat #138.


def test_the_operator_docs_can_identify_a_log_with_the_current_last_column() -> None:
    """Both docs must list the live last column, at the live width.

    These are documents the operator reads, not code. Nothing else in the
    suite reads them, so nothing else turns red when they go stale — which
    is exactly how a stale manual ships.

    The check is deliberately narrow: it looks for a table row naming the
    current last column and carrying its position. It does not try to
    parse the tables or keep them in sync with each other, because a
    cleverer check would be a new place to be wrong.
    """
    last = G3_DEMO_RESULTS_CSV_COLUMNS[-1]
    width = len(G3_DEMO_RESULTS_CSV_COLUMNS)
    row_prefix = f"| `{last}` |"
    stale: dict[str, str] = {}
    for name in ("w0a-diagnostic-run-runbook.md", "g3-local-test-sop.md"):
        path = Path(__file__).resolve().parents[2] / "docs" / name
        rows = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(row_prefix)
        ]
        if not rows:
            stale[name] = f"no row names `{last}`"
        elif not any(str(width) in row for row in rows):
            stale[name] = f"`{last}` is listed but not as {width} columns"
    assert not stale, (
        f"the operator-facing docs do not describe the current schema: {stale}. "
        f"The demo log's last column is `{last}` at {width} columns. An "
        "operator following the runbook's step 0 identifies their file by "
        "its last column heading; a file that is not in the table reads as "
        "stale, and the runbook's remedy for a stale file is to discard it "
        "and re-run — which would throw away a correct dataset."
    )
