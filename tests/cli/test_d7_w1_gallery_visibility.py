"""D7-A W1 RED: gallery visibility (expected/loaded/rejected) on every demo row.

Three columns the plan mandates (v8 §3.1 W1-a):
  expected_count  int
  loaded_count    int
  gallery_rejected  "filename（reason）" joined by ";" — EMPTY means all loaded.

These tests drive the PRODUCTION entry point (the window's labeled and
unlabeled RoundComplete paths -> append_g3_demo_results_csv) rather than
injecting a pre-built row, so a test cannot pass while the wiring is
absent. Every assertion below is expected to FAIL before the change.
"""

from __future__ import annotations

import csv
from pathlib import Path

from facecore.live.frame_pipeline import EnrollmentFailure, GalleryLoadReport
from facecore.research.cli import (
    G3_DEMO_RESULTS_CSV_COLUMNS,
    append_g3_demo_results_csv,
)

EXPECTED_NEW = ("expected_count", "loaded_count", "gallery_rejected")


class _Report:
    """Wraps the REAL GalleryLoadReport.

    The Qt window also reads `load_report.source_dir`, so a hand-rolled
    double breaks the UI path with AttributeError. Building the real
    dataclass keeps this a test of production behaviour rather than of my
    guess about which attributes the window touches.
    """

    def __init__(self, *, expected, loaded, failures):
        self._real = GalleryLoadReport(
            expected_count=expected,
            loaded_count=loaded,
            failures=tuple(failures),
            source_dir=None,
        )

    def __getattr__(self, name):
        return getattr(self._real, name)


def _report(*, expected=23, loaded=21, failures=()):
    return _Report(expected=expected, loaded=loaded, failures=failures)


def _failure(filename: str, reason: str) -> EnrollmentFailure:
    return EnrollmentFailure(filename=filename, reason=reason)


class _DummyGallery:
    """Carries only a load_report, the way the real gallery does."""

    def __init__(self, *, load_report):
        self.load_report = load_report


class TestColumnsExist:
    def test_three_new_columns_are_declared(self) -> None:
        missing = [c for c in EXPECTED_NEW if c not in G3_DEMO_RESULTS_CSV_COLUMNS]
        assert not missing, f"missing demo columns: {missing}"

    def test_new_columns_are_appended_after_the_existing_35(self) -> None:
        assert G3_DEMO_RESULTS_CSV_COLUMNS[-3:] == EXPECTED_NEW, (
            "W1 must append, never insert: operator's spreadsheet formulas "
            "depend on the existing positions"
        )

    def test_the_23_legacy_columns_keep_their_exact_names_and_order(self) -> None:
        legacy = G3_DEMO_RESULTS_CSV_COLUMNS[:23]
        assert legacy[17] == "margin"  # column 18
        assert legacy[20] == "required_support"  # column 21
        assert legacy[21] == "label_kind"  # column 22
        assert legacy[22] == "label_identity"  # column 23
        assert len(legacy) == 23


class TestRejectedFormatting:
    def test_filename_and_reason_are_both_present(self) -> None:
        row = _row(_report(failures=[_failure("bad-01.jpg", "quality_blurry")]))
        assert "bad-01.jpg" in row["gallery_rejected"]
        assert "quality_blurry" in row["gallery_rejected"]

    def test_multiple_failures_join_with_semicolon(self) -> None:
        row = _row(
            _report(
                failures=[
                    _failure("bad-01.jpg", "quality_blurry"),
                    _failure("bad-02.jpg", "no_face"),
                ]
            )
        )
        assert row["gallery_rejected"].count(";") == 1, (
            "two failures must be two entries, not one merged blob"
        )
        assert "bad-01.jpg" in row["gallery_rejected"]
        assert "bad-02.jpg" in row["gallery_rejected"]

    def test_all_success_is_the_empty_string(self) -> None:
        row = _row(_report(expected=23, loaded=23, failures=()))
        assert row["gallery_rejected"] == "", (
            "empty means every photo loaded; it must not become '0' or 'none'"
        )

    def test_the_plan_mandated_bracket_format_is_used_exactly(self) -> None:
        """Plan v8 §3.1: 「建議存 filename（reason）形式」.

        Asserted as an exact cell, not as "filename in cell and reason in
        cell" — that weaker form survived the M4 mutation, which swapped
        the full-width brackets for a colon. A reader parses this cell by
        eye; the brackets are what make the boundary legible.
        """
        row = _row(_report(failures=[_failure("bad-01.jpg", "quality_blurry")]))
        assert row["gallery_rejected"] == "bad-01.jpg（quality_blurry）"

    def test_two_failures_keep_the_exact_delimited_form(self) -> None:
        row = _row(
            _report(
                failures=[
                    _failure("bad-01.jpg", "quality_blurry"),
                    _failure("bad-02.jpg", "no_face"),
                ]
            )
        )
        assert row["gallery_rejected"] == (
            "bad-01.jpg（quality_blurry）;bad-02.jpg（no_face）"
        )

    def test_no_report_yields_empty_counts_not_a_crash(self) -> None:
        """A caller with no load_report must still produce a row."""
        row = _row(None)
        assert row["gallery_rejected"] == ""
        assert row["expected_count"] == ""
        assert row["loaded_count"] == ""


class TestCounts:
    def test_expected_and_loaded_are_written_as_text(self) -> None:
        row = _row(_report(expected=23, loaded=21))
        assert row["expected_count"] == "23"
        assert row["loaded_count"] == "21"

    def test_counts_are_carried_on_every_round(self) -> None:
        """The same report must reach row 1 and row 2 — not just the first."""
        rows = _two_rows()
        assert rows[0]["expected_count"] == rows[1]["expected_count"] == "23"
        assert rows[0]["loaded_count"] == rows[1]["loaded_count"] == "21"

    def test_counts_are_not_recomputed_per_round(self) -> None:
        """The gallery loads once at startup; the report is a fixed value."""
        rows = _two_rows()
        assert rows[0]["gallery_rejected"] == rows[1]["gallery_rejected"]


class TestAppendPathEndToEnd:
    def test_a_real_append_writes_the_three_columns(self, tmp_path: Path) -> None:
        target = tmp_path / "demo-results.csv"
        report = _report(
            expected=23,
            loaded=21,
            failures=[_failure("bad-01.jpg", "quality_blurry")],
        )
        append_g3_demo_results_csv(
            target, _round(), required_support=3, labeled_at_utc="2026-10-03T00:00:00Z",
            gallery_load_report=report,
        )
        with target.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            rows = list(reader)
        assert header is not None
        for column in EXPECTED_NEW:
            assert column in header
        assert rows[0]["expected_count"] == "23"
        assert rows[0]["loaded_count"] == "21"
        assert "bad-01.jpg" in rows[0]["gallery_rejected"]

    def test_header_is_written_once_and_existing_rows_are_preserved(
        self, tmp_path: Path
    ) -> None:
        """Existing append semantics (#140) must not change."""
        target = tmp_path / "demo-results.csv"
        for _ in range(3):
            append_g3_demo_results_csv(
                target, _round(), required_support=3,
                labeled_at_utc="2026-10-03T00:00:00Z", gallery_load_report=_report(),
            )
        with target.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            rows = list(reader)
        assert header is not None
        assert len(rows) == 3
        assert header == list(G3_DEMO_RESULTS_CSV_COLUMNS)
        assert all(r["gallery_rejected"] == "" for r in rows)


class TestProductionWiring:
    def test_the_window_forwards_its_load_report_to_both_write_paths(
        self,
    ) -> None:
        """A real window carrying a load_report must reach the CSV.

        A wiring assertion: it parses the module and reads every call to
        `append_g3_demo_results_csv`, so a test cannot pass while the
        argument is never passed. Parsed with ast rather than string
        slicing — the first version split the call text on ")" and read
        zero call sites, because an earlier `getattr(..., "x", 0)`
        argument closed the "block" long before gallery_load_report.
        """
        import ast

        source = (
            Path(__file__).resolve().parents[2]
            / "src" / "facecore" / "live" / "qt_window.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        forwarding: list[str] = []
        total_calls = 0
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "append_g3_demo_results_csv"
            ):
                continue
            total_calls += 1
            keywords = {kw.arg for kw in node.keywords}
            if "gallery_load_report" in keywords:
                forwarding.append(ast.unparse(node))
        assert total_calls == 2, (
            "both demo write paths (labeled 正確／錯誤 and unlabeled "
            "再次辨識) write demo rows; a third or missing call site means "
            "this file's view of the write path is stale"
        )
        assert len(forwarding) == total_calls, (
            "every append_g3_demo_results_csv call must forward "
            "gallery_load_report"
        )
        assert all("self.load_report" in call for call in forwarding), (
            "the forwarded value must be the window's startup report, not a "
            "recomputed per-round one"
        )

    def test_round_complete_carries_the_report(self) -> None:
        from facecore.live.qt_window import RoundComplete

        fields = RoundComplete.__dataclass_fields__
        assert "gallery_load_report" in fields, (
            "RoundComplete is the demo write path's carrier; without a "
            "gallery field the report cannot reach the CSV"
        )

    def test_the_three_columns_are_blank_only_when_no_report_exists(self) -> None:
        """Guards the W3 §8 exemption this feature had to take.

        `test_demo_csv_end_to_end_carries_the_new_fields` excludes these
        three columns from its blank-cell guard, because the window it
        builds has no gallery. That exclusion is only honest while a window
        WITH a report fills them — this assertion is the other half, and it
        fails if the exemption becomes the normal case.
        """
        import importlib.util
        import os

        import pytest

        if importlib.util.find_spec("PySide6") is None:
            pytest.skip("needs the Qt demo path")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from facecore.live.capture import FakeCapture
        from facecore.live.qt_window import QtResearchWindow
        from tests.live.test_g3_round_records import _RoundFactory, _face_frames
        from tests.live.test_qt_window import _matching_scorer

        app = QApplication.instance() or QApplication([])
        del app

        import csv as _csv
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            base = _Path(tmp)
            factory = _RoundFactory(base, FakeCapture(_face_frames()), _matching_scorer)
            desktop, consent, _attempt = factory()
            demo_csv = base / "demo-results.csv"
            gallery = _DummyGallery(
                load_report=_report(
                    expected=23, loaded=21,
                    failures=[_failure("bad-01.jpg", "quality_blurry")],
                )
            )
            window = QtResearchWindow(
                desktop,
                consent=consent,
                recorder=None,
                attempt_id=None,
                offscreen=True,
                clock_ns=lambda: 0,
                next_session=factory,
                demo_results_csv=demo_csv,
                gallery=gallery,
            )
            window.show()
            window.enter_ready()
            window.start_clicked()
            window.process_until_terminal(max_steps=200)
            window.press_correct()
            window.close()

            assert demo_csv.is_file(), "the labeled path wrote no demo row"
            with demo_csv.open(encoding="utf-8") as fh:
                row = next(iter(_csv.DictReader(fh)))
            assert row["expected_count"] == "23", (
                "a window carrying a load_report must fill these columns — "
                "the W3 blank-cell exemption must not become the norm"
            )
            assert row["loaded_count"] == "21"
            assert row["gallery_rejected"] == "bad-01.jpg（quality_blurry）"


# ---------------------------------------------------------------- fixtures


def _round():
    """A real RoundComplete, built the way the window builds it."""
    from facecore.live.contracts import (
        FrameObservation,
        SessionResult,
        SessionStatus,
    )
    from facecore.live.qt_window import RoundComplete

    observation = FrameObservation(
        sequence=1, captured_ns=0, processed_ns=1, quality_pass=True,
        quality_reasons=(), face_count=1,
        face_box=(100.0, 100.0, 200.0, 200.0),
        identity_scores={"synthetic-a": 0.8, "synthetic-b": 0.1},
        quality_rank=0.9, model_generation="synthetic-gen",
        gallery_digest="synthetic-digest",
    )
    terminal = SessionResult(
        session_id="w1-synthetic", schema_version="v1",
        status=SessionStatus.matched, matched_identity="synthetic-a",
        reason_codes=("supported_3_frames",), elapsed_ms=1200.0,
        frames_sampled=3, frames_usable=3, frames_rejected=0,
        frames_dropped=0, support_sequences=(),
        profile_digest="synthetic-profile",
        model_generation="synthetic-gen",
        gallery_digest="synthetic-digest",
    )
    return RoundComplete(
        session_id="w1-synthetic", attempt_id=None, terminal=terminal,
        observations=(observation,), label_kind="unlabeled",
        label_identity=None, profile_version="w1-test",
        started_utc="2026-10-03T00:00:00Z",
    )


def _row(report):
    from facecore.research.cli import g3_demo_round_row

    return g3_demo_round_row(
        _round(), required_support=3, labeled_at_utc="2026-10-03T00:00:00Z",
        gallery_load_report=report,
    )


def _two_rows():
    return [_row(_report()), _row(_report())]