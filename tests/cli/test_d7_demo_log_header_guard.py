"""D7-A #141 (丙-新): demo log header guard + per-App-execution file.

Governing decision `d-20261003071803476368-1`. Two mechanisms:

  1. Header guard — before appending, compare the file's ACTUAL first row
     against `G3_DEMO_RESULTS_CSV_COLUMNS` as a whole tuple. Mismatch
     refuses with zero writes. A length-only check is explicitly NOT
     enough: a header can be renamed or reordered without changing its
     length, and that drift still makes every later column land in the
     wrong cell.

  2. Per-App-execution file — the demo log is named after when the App
     started, so a new App run opens a new file and never appends to an
     older one. That is what makes (1) a backstop rather than a daily
     obstacle: new App == new code == new header == new file.

Every test drives the PRODUCTION entry point (`append_g3_demo_results_csv`
and the path builder `cmd_live` itself) rather than a hand-built row, so a
test cannot pass while the wiring is absent.

The third header case is the one this whole task exists for: same column
COUNT, different names/order. `TestSameCountDifferentOrder` asserts it
twice — once that the call refuses, and once that the file is byte
identical afterwards.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import pytest

from facecore.research.cli import (
    G3_DEMO_RESULTS_CSV_COLUMNS,
    append_g3_demo_results_csv,
    cmd_live,
    demo_results_csv_path,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_header(path: Path, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(columns)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def qt_app() -> Any:
    """Only for the tests that build a real window.

    Kept so a future Qt-backed guard has a fixture; a test that merely
    drives `cmd_live` must NOT ask for it. `importorskip` makes a
    PySide6-dependent test vanish from the `verify` job (which installs
    .[dev] only) instead of failing there — and a guard that no CI job
    runs is a guard that proves nothing while the run stays green.
    """
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


# ------------------------------------------------------------------ fixtures


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
        session_id="guard-synthetic", schema_version="v1",
        status=SessionStatus.matched, matched_identity="synthetic-a",
        reason_codes=("supported_3_frames",), elapsed_ms=1200.0,
        frames_sampled=3, frames_usable=3, frames_rejected=0,
        frames_dropped=0, support_sequences=(),
        profile_digest="synthetic-profile",
        model_generation="synthetic-gen",
        gallery_digest="synthetic-digest",
    )
    return RoundComplete(
        session_id="guard-synthetic", attempt_id=None, terminal=terminal,
        observations=(observation,), label_kind="unlabeled",
        label_identity=None, profile_version="guard-test",
        started_utc="2026-10-03T00:00:00Z",
    )


def _append(target: Path) -> None:
    append_g3_demo_results_csv(
        target, _round(), required_support=3,
        labeled_at_utc="2026-10-03T00:00:00Z",
    )


# ------------------------------------------------------------- case 1: match


class TestMatchingHeaderAppends:
    def test_an_existing_matching_header_is_appended_to(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "demo-results.csv"
        _write_header(target, G3_DEMO_RESULTS_CSV_COLUMNS)
        _append(target)
        assert len(_rows(target)) == 1, "a matching header must not be refused"

    def test_a_missing_file_is_created_with_the_current_header(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "demo-results.csv"
        _append(target)
        with target.open(encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == list(G3_DEMO_RESULTS_CSV_COLUMNS)

    def test_repeated_appends_do_not_duplicate_the_header(self, tmp_path: Path) -> None:
        target = tmp_path / "demo-results.csv"
        for _ in range(3):
            _append(target)
        assert len(_rows(target)) == 3
        with target.open(encoding="utf-8") as fh:
            first = next(csv.reader(fh))
        assert tuple(first) == G3_DEMO_RESULTS_CSV_COLUMNS, (
            "one header row, not one per append. Compared as a tuple rather "
            "than by counting a substring: 'mode' also occurs inside "
            "'model_generation', so a substring count reads 2 here."
        )


# ------------------------------------------------- case 2: different length


class TestDifferentLengthHeaderRefuses:
    def test_it_refuses(self, tmp_path: Path) -> None:
        target = tmp_path / "demo-results.csv"
        # The 23-column shape the operator may still have on disk.
        _write_header(target, G3_DEMO_RESULTS_CSV_COLUMNS[:23])
        with pytest.raises(OSError):
            _append(target)

    def test_the_file_is_byte_identical_afterwards(self, tmp_path: Path) -> None:
        target = tmp_path / "demo-results.csv"
        _write_header(target, G3_DEMO_RESULTS_CSV_COLUMNS[:23])
        before = _digest(target)
        with pytest.raises(OSError):
            _append(target)
        assert _digest(target) == before, (
            "a refusal must leave the file untouched — proven by hash, "
            "because an unchanged row COUNT does not prove the bytes held"
        )


# ------------------------------- case 3: same length, different names/order


class TestSameCountDifferentOrder:
    """The case a length check cannot catch — and this task's whole point."""

    def _renamed_header(self) -> tuple[str, ...]:
        """Same count and same order except one column is RENAMED."""
        cols = list(G3_DEMO_RESULTS_CSV_COLUMNS)
        cols[0] = "modes"  # `mode` -> `modes`
        return tuple(cols)

    def test_a_renamed_column_refuses(self, tmp_path: Path) -> None:
        target = tmp_path / "demo-results.csv"
        _write_header(target, self._renamed_header())
        with pytest.raises(OSError):
            _append(target)

    def test_a_reordered_header_refuses(self, tmp_path: Path) -> None:
        target = tmp_path / "demo-results.csv"
        swapped = list(G3_DEMO_RESULTS_CSV_COLUMNS)
        swapped[3], swapped[4] = swapped[4], swapped[3]
        _write_header(target, tuple(swapped))
        with pytest.raises(OSError):
            _append(target)

    def test_the_fixture_is_what_case_two_cannot_catch(
        self, tmp_path: Path,
    ) -> None:
        """States the premise the refusal above depends on — as two claims
        that can each go red.

        The first version of this asserted only
        `len(_renamed_header()) == len(G3_DEMO_RESULTS_CSV_COLUMNS)`, and
        that equation is a tautology: `_renamed_header` is built by
        copying COLUMNS and changing one element, so the two can never
        differ in length. It passed under every mutation, including the
        one this file exists to prevent.

        So assert the two properties a length check cannot supply:
        the header really is the SAME LENGTH (a length check would wave it
        through) and really is DIFFERENT (only a tuple comparison stops
        it). Each clause fails on its own if a future edit drops a column
        from the fixture or lets it drift back into equality — which is
        what makes this a guard rather than a restatement.
        """
        renamed = self._renamed_header()
        assert len(renamed) == len(G3_DEMO_RESULTS_CSV_COLUMNS), (
            "no longer the same count: case 2's length check would now "
            "catch this, so the tests would be passing for the wrong reason"
        )
        assert renamed != G3_DEMO_RESULTS_CSV_COLUMNS, (
            "the fixture drifted back into equality; nothing is being "
            "refused any more"
        )
        assert renamed[0] != G3_DEMO_RESULTS_CSV_COLUMNS[0]

    def test_the_renamed_file_is_byte_identical_afterwards(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "demo-results.csv"
        _write_header(target, self._renamed_header())
        before = _digest(target)
        with pytest.raises(OSError):
            _append(target)
        assert _digest(target) == before


# ------------------------------------------------------------ the message


class TestRefusalMessage:
    """Slot 2 asks: can the operator act on this alone, with no agent?"""

    def _message(self, target: Path) -> str:
        _write_header(target, G3_DEMO_RESULTS_CSV_COLUMNS[:23])
        with pytest.raises(OSError) as exc:
            _append(target)
        return str(exc.value)

    def test_it_states_the_actual_and_expected_column_counts(
        self, tmp_path: Path,
    ) -> None:
        msg = self._message(tmp_path / "demo-results.csv")
        assert "23" in msg
        assert str(len(G3_DEMO_RESULTS_CSV_COLUMNS)) in msg

    def test_it_names_the_last_column_of_each_header(self, tmp_path: Path) -> None:
        """The runbook's step 0 tells the operator to identify their log by
        the LAST column's heading. The message must carry both so they can
        match it against that rule without opening a spreadsheet."""
        msg = self._message(tmp_path / "demo-results.csv")
        assert "label_identity" in msg
        assert G3_DEMO_RESULTS_CSV_COLUMNS[-1] in msg

    def test_it_points_at_the_offending_file(self, tmp_path: Path) -> None:
        target = tmp_path / "demo-results.csv"
        assert str(target) in self._message(target)

    def test_it_tells_the_operator_what_to_do_next(self, tmp_path: Path) -> None:
        msg = self._message(tmp_path / "demo-results.csv")
        assert "demo-results-" in msg, (
            "the fix is a per-App-execution file; naming it is what lets the "
            "operator move on alone"
        )


# -------------------------------------------------- per-App-execution file


class TestPerAppExecutionFileName:
    def test_two_app_starts_get_two_different_files(self, tmp_path: Path) -> None:
        first = demo_results_csv_path(tmp_path, "2026-10-03T00:00:00Z")
        second = demo_results_csv_path(tmp_path, "2026-10-03T01:00:00Z")
        assert first != second

    def test_the_same_app_start_is_stable(self, tmp_path: Path) -> None:
        stamp = "2026-10-03T00:00:00Z"
        assert demo_results_csv_path(tmp_path, stamp) == demo_results_csv_path(
            tmp_path, stamp
        )

    def test_it_lives_in_the_store_root(self, tmp_path: Path) -> None:
        path = demo_results_csv_path(tmp_path, "2026-10-03T00:00:00Z")
        assert path.parent == tmp_path

    def test_the_stamp_is_filesystem_safe(self, tmp_path: Path) -> None:
        name = demo_results_csv_path(tmp_path, "2026-10-03T07:11:07+00:00").name
        assert "/" not in name and ":" not in name, name

    def test_rounds_of_one_app_share_one_file(self, tmp_path: Path) -> None:
        """The decision's whole reason for choosing (a) over (c): one App run
        is one file, no matter how many rounds it commits."""
        stamp = "2026-10-03T00:00:00Z"
        target = demo_results_csv_path(tmp_path, stamp)
        for _ in range(3):
            _append(target)
        assert len(_rows(target)) == 3

    def test_the_name_carries_no_session_id(self, tmp_path: Path) -> None:
        """`session_id` means per-ROUND in this codebase (round_session_id
        at cli.py:1601). Using it here would split one App run into dozens
        of files, which is exactly what decision (丙-新) ruled out."""
        path = demo_results_csv_path(tmp_path, "2026-10-03T00:00:00Z")
        assert "session" not in path.name.lower()


    def test_this_guard_runs_where_ci_runs_it(self) -> None:
        """The wiring guard must not depend on PySide6.

        The `verify` CI job installs `.[dev]` only — no PySide6. The
        `qt_app` fixture uses `importorskip`, so any test requesting it
        there is silently skipped rather than failed: the job stays
        green and the guard proves nothing. `qt-smoke` does install
        PySide6 but never runs `tests/cli/`, so nothing else picks it up.

        That is how this wiring guard shipped in its first form — it was
        written, it passed locally, and no CI job ever executed it.

        Asserted structurally, by inspecting the guard's own signature:
        adding `qt_app` back turns this red without needing a second
        environment.
        """
        import inspect

        from tests.cli.test_d7_demo_log_header_guard import (
            TestSecondPrecisionFileName as cls,
        )

        params = inspect.signature(
            cls.test_cmd_live_passes_a_microsecond_instant_and_gets_a_second_precision_name
        ).parameters
        assert "qt_app" not in params, (
            "cmd_live defaults to ui='fake' and needs no Qt — requesting "
            "qt_app makes importorskip drop this test in the verify job, "
            "where a green run would mean the guard never ran"
        )


class TestSecondPrecisionFileName:
    """#145 follow-up: the name carries seconds, not microseconds.

    `datetime.now().isoformat()` almost always carries six microsecond
    digits, and the earlier implementation only deleted the decimal point
    — so a real App produced a name twelve digits long while every example
    in the runbook showed six. An operator comparing his Finder window
    against the manual had no way to reconcile the two.
    """

    SECOND = "2026-10-03T15:30:00+00:00"
    MICROS = "2026-10-03T15:30:00.123456+00:00"

    def test_second_precision_input(self, tmp_path: Path) -> None:
        assert demo_results_csv_path(tmp_path, self.SECOND).name == (
            "demo-results-2026-10-03T153000_0000.csv"
        )

    def test_microsecond_input_is_dropped_to_seconds(self, tmp_path: Path) -> None:
        assert demo_results_csv_path(tmp_path, self.MICROS).name == (
            "demo-results-2026-10-03T153000_0000.csv"
        )

    def test_both_precisions_name_the_same_file(self, tmp_path: Path) -> None:
        """The property the fix exists for: one instant, one filename.

        Not a length assertion — a length check would also pass if the
        name kept the microseconds and merely dropped a different pair of
        characters. Equality between the two inputs is what a reader
        actually depends on.
        """
        assert demo_results_csv_path(tmp_path, self.SECOND) == (
            demo_results_csv_path(tmp_path, self.MICROS)
        )

    def test_a_utc_z_suffix_is_accepted(self, tmp_path: Path) -> None:
        assert demo_results_csv_path(tmp_path, "2026-10-03T15:30:00Z").name == (
            "demo-results-2026-10-03T153000_0000.csv"
        )

    def test_the_offset_survives(self, tmp_path: Path) -> None:
        """Truncating the string would have eaten the timezone with the
        microseconds; a non-UTC instant must keep its offset."""
        assert demo_results_csv_path(
            tmp_path, "2026-10-03T15:30:00+08:00"
        ).name == "demo-results-2026-10-03T153000_0800.csv"

    def test_different_instants_still_differ(self, tmp_path: Path) -> None:
        """Second precision must not collapse distinct App runs."""
        first = demo_results_csv_path(tmp_path, "2026-10-03T15:30:00+00:00")
        second = demo_results_csv_path(tmp_path, "2026-10-03T15:30:01+00:00")
        assert first != second

    def test_it_matches_the_name_the_runbook_shows(self, tmp_path: Path) -> None:
        """The manual's example has to be a name the App can produce.

        Read out of the runbook rather than restated here, so that editing
        the manual without touching the code turns this red — the drift
        this follow-up exists to close cannot come back unnoticed.
        """
        runbook = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "w0a-diagnostic-run-runbook.md"
        )
        assert runbook.is_file(), runbook
        examples = set(re.findall(r"demo-results-[\dT:_-]+\.csv", runbook.read_text()))
        per_app = {n for n in examples if n != "demo-results-old.csv"}
        assert per_app, "the runbook no longer shows a per-App filename example"
        produced = demo_results_csv_path(tmp_path, self.SECOND).name
        assert produced in per_app, (
            f"the App produces {produced} but the runbook shows {sorted(per_app)}"
        )

    def test_cmd_live_passes_a_microsecond_instant_and_gets_a_second_precision_name(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Wiring guard: what `cmd_live` actually passes in, and what comes out.

        The original bug was in the CALLER's input, not in the helper:
        `cmd_live` hands over `now.isoformat()`, which carries six
        microsecond digits. A test that calls `demo_results_csv_path`
        with tidy inputs cannot see that at all.

        So run a real `cmd_live` demo session and capture the argument it
        passes to the path builder. Asserting on the captured value
        (rather than on a file that only appears after an operator
        verdict) keeps the assertion on the wire between the two, which
        is where the defect lived.

        Deliberately takes NO `qt_app` fixture: `cmd_live` defaults to
        `ui="fake"` and never builds a window. An earlier version asked
        for Qt, and because the fixture uses `importorskip` the whole
        test vanished from the `verify` job — green run, no guard. The
        CI config has the same blind spot documented in
        `test_this_guard_runs_where_ci_runs_it`.
        """
        import json

        profile = tmp_path / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "schema_version": "v1",
                    "profile_version": "filename-fix",
                    "timeout_ms": 5000,
                    "sample_interval_ms": 200,
                    "max_frames": 26,
                    "queue_limit": 1,
                    "required_support": 1,
                    "min_support_interval_ms": 1,
                    "match_threshold": 0.10,
                    "review_threshold": 0.05,
                    "margin_threshold": 0.01,
                    "detector_version": "det-fix",
                    "quality_policy_version": "qual-fix",
                    "continuity_max_center_delta_ratio": 0.5,
                }
            ),
            encoding="utf-8",
        )

        captured: list[tuple[str, str]] = []
        real = demo_results_csv_path

        def spy(store_root: Path, started_utc: str) -> Path:
            produced = real(store_root, started_utc)
            captured.append((started_utc, produced.name))
            return produced

        monkeypatch.setattr(
            "facecore.research.cli.demo_results_csv_path", spy, raising=True
        )

        rc = cmd_live(
            profile_path=profile,
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="filename-fix",
            record_consent=False,
            image_consent=False,
            mode="demo",
        )

        assert rc == 0, rc
        assert captured, "cmd_live never asked for a demo log path"
        started_utc, name = captured[0]
        assert re.search(r"\.\d{6}[+-]\d{2}:\d{2}$", started_utc), (
            f"expected a microsecond-bearing instant, got {started_utc!r}; if "
            "this stops holding, the premise of the test moved and the "
            "wiring it guards may no longer be the real one"
        )
        stamp = name[len("demo-results-") : -len(".csv")]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{6}_\d{4}", stamp), (
            f"cmd_live passed {started_utc} and got {name} — the microsecond "
            "digits are still in the filename the operator sees"
        )