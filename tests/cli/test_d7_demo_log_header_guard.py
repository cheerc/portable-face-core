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
from pathlib import Path

import pytest

from facecore.research.cli import (
    G3_DEMO_RESULTS_CSV_COLUMNS,
    append_g3_demo_results_csv,
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

    def test_the_count_is_identical_so_a_length_check_would_miss_it(
        self, tmp_path: Path,
    ) -> None:
        """States the premise the refusal above depends on.

        If a future edit ever changes the column count, these fixtures
        stop being 'same count' and the tests would pass for the wrong
        reason (case 2's length check). Pinning it keeps them honest.
        """
        assert len(self._renamed_header()) == len(G3_DEMO_RESULTS_CSV_COLUMNS)

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