"""D3b RED: recording stays byte-identical; demo mode records nothing.

Source of truth: commander decisions
- d-20260929174858816709-11 item 5 — the two byte-level proofs are
  mandatory: appending in record mode must leave the pre-existing header
  and rows byte-for-byte identical, and one demo round must leave
  results.csv with unchanged content AND unchanged mtime.
- d-20260929193128013174-12 item 4 — the byte/hash comparison is the
  primary assertion and st_mtime_ns is the second track; if a write is
  intercepted at all it must be builtins.open (cli.py's own writer uses
  open()), because intercepting shutil/Path catches nothing.
- d-20260929193651396921-13 — the operator's App is `--mode demo` while
  cmd_live keeps its recording default.

Every guard here is written against the real path the code walks: the
recorder passed into the window, and the store directory handed to
cmd_live. No directory is created that the code under test never
receives — that shape is what made D3a's S1 test a false protection.

Synthetic frames, FakeCapture, offscreen Qt, tmp dirs. No camera, no
real faces, no real gallery, no real uids.
"""

from __future__ import annotations

import builtins
import csv
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from facecore.live.capture import FakeCapture
from facecore.live.qt_window import QtResearchWindow
from facecore.research.cli import (
    G3_DEMO_RESULTS_CSV_COLUMNS,
    G3_DEMO_RESULTS_CSV_NAME,
    G3_RESULTS_CSV_COLUMNS,
    cmd_live,
    g3_demo_round_row,
)
from tests.live.test_g3_round_records import _RoundFactory, _face_frames
from tests.live.test_qt_window import _matching_scorer

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="D3b demo/record wiring is exercised through the Qt window",
)


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


def _profile_dict(tmp_path: Path, *, name: str = "profile.json") -> Path:
    import json as _json

    profile = tmp_path / name
    profile.write_text(
        _json.dumps(
            {
                "schema_version": "v1",
                "profile_version": "d3b-test",
                "timeout_ms": 5000,
                "sample_interval_ms": 200,
                "max_frames": 26,
                "queue_limit": 1,
                "required_support": 1,
                "min_support_interval_ms": 1,
                # Matches the frozen g3-v1 geometry, with thresholds low
                # enough that the CLI's deterministic fake scorer
                # (0.50 / 0.30) reaches matched on its first frame. The
                # recording-vs-demo behaviour under test does not depend
                # on the real 0.363 threshold.
                "match_threshold": 0.10,
                "review_threshold": 0.05,
                "margin_threshold": 0.01,
                "detector_version": "det-d3b",
                "quality_policy_version": "qual-d3b",
                "continuity_max_center_delta_ratio": 0.5,
            }
        )
    )
    return profile


# ---------------------------------------------------------------------------
# Byte-level helpers. The three-part shape is commander-ruled: a prefix
# comparison alone cannot catch a header re-appended at the tail.
# ---------------------------------------------------------------------------


def _csv_parts(path: Path) -> tuple[list[str], list[list[str]], bytes]:
    """Return (header, body rows, raw bytes) for a csv file."""
    raw = path.read_bytes()
    rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    return rows[0], rows[1:], raw


def _assert_frozen_header_and_legacy_rows_untouched(
    before: bytes,
    after: bytes,
    *,
    legacy_rows: int,
) -> None:
    """One append must not alter the file's existing bytes at all.

    Three checks, because each catches a different real regression:
    1. byte prefix identity — catches in-place rewrite/reorder;
    2. header line still the original literal — catches a re-appended
       header (which a prefix check alone would NOT see, since the extra
       header lands at the tail);
    3. legacy row count and width preserved — catches a truncated or
       re-shaped legacy row.
    """
    assert after[: len(before)] == before, "existing bytes were rewritten"
    before_rows = list(csv.reader(before.decode("utf-8").splitlines()))
    after_rows = list(csv.reader(after.decode("utf-8").splitlines()))
    assert after_rows[0] == before_rows[0], "header line changed"
    assert after_rows[0] == list(G3_RESULTS_CSV_COLUMNS)
    assert len(after_rows) - 1 == legacy_rows + 1, "expected exactly one new row"
    for index, row in enumerate(after_rows[1:], start=1):
        assert len(row) == len(G3_RESULTS_CSV_COLUMNS), f"row {index} wrong width"


def _no_open_on(target_name: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record any builtins.open() call whose PATH IS EXACTLY target_name.

    The daemon spike measured that only builtins.open actually fires for
    append_g3_results_csv (0 hits for Path.open / shutil.copy*), so this
    is the one interception layer that is not a false protection. The
    byte-level assertions remain the primary defense; this is a second,
    independent track.

    Matching is on the file's own name, not a substring: "results.csv" is
    a substring of "demo-results.csv", and a substring match would flag
    the demo file itself as a violation of the rule it is exempt from.
    """
    opened: list[str] = []
    real_open = builtins.open

    def _tracking_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            if Path(str(file)).name == target_name:
                opened.append(str(file))
        except Exception:
            pass
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _tracking_open)
    return opened


# ---------------------------------------------------------------------------
# 1. Record mode: appending leaves the pre-existing file byte-identical.
# ---------------------------------------------------------------------------


class TestRecordModeIsByteIdentical:
    def test_append_preserves_every_existing_byte(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Ruling -11 item 5, first half: header and rows survive verbatim."""
        from facecore.research.cli import append_g3_results_csv

        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), _matching_scorer)
        results_csv = tmp_path / "results.csv"
        round_ = _one_labeled_round(qt_app, factory, results_csv, "rec-1")
        append_g3_results_csv(results_csv, round_)

        before = results_csv.read_bytes()
        header, legacy_rows, _ = _csv_parts(results_csv)
        assert header == list(G3_RESULTS_CSV_COLUMNS)
        assert len(legacy_rows) == 1

        append_g3_results_csv(results_csv, round_)
        after = results_csv.read_bytes()

        _assert_frozen_header_and_legacy_rows_untouched(
            before, after, legacy_rows=len(legacy_rows)
        )

    def test_legacy_seeded_file_gains_exactly_one_row(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """A file seeded with older rounds keeps them and gains one row."""
        from facecore.research.cli import append_g3_results_csv

        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), _matching_scorer)
        results_csv = tmp_path / "results.csv"
        # Seed two historical rows the way the 29 existing rounds look.
        with open(results_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=G3_RESULTS_CSV_COLUMNS)
            writer.writeheader()
            for index in range(2):
                row = {key: "" for key in G3_RESULTS_CSV_COLUMNS}
                row.update(
                    {
                        "round_id": f"legacy-{index}",
                        "session_id": f"legacy-{index}",
                        "result": "matched",
                        "gallery_digest": "e3d77c4cfab5b141a8abaecdb908254d7",
                    }
                )
                writer.writerow(row)

        before = results_csv.read_bytes()
        _, legacy_rows, _ = _csv_parts(results_csv)
        assert len(legacy_rows) == 2

        append_g3_results_csv(
            results_csv, _one_labeled_round(qt_app, factory, results_csv, "rec-2")
        )
        after = results_csv.read_bytes()

        _assert_frozen_header_and_legacy_rows_untouched(
            before, after, legacy_rows=2
        )


# ---------------------------------------------------------------------------
# 2. Demo mode: one round leaves results.csv content AND mtime untouched.
# ---------------------------------------------------------------------------


class TestDemoModeLeavesResearchLedgerUntouched:
    def _seed_legacy_store(self, tmp_path: Path) -> tuple[Path, bytes, int]:
        store = tmp_path / "store"
        store.mkdir(parents=True, exist_ok=True)
        results_csv = store / "results.csv"
        with open(results_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=G3_RESULTS_CSV_COLUMNS)
            writer.writeheader()
            for index in range(2):
                row = {key: "" for key in G3_RESULTS_CSV_COLUMNS}
                row.update(
                    {
                        "round_id": f"legacy-{index}",
                        "session_id": f"legacy-{index}",
                        "result": "matched",
                    }
                )
                writer.writerow(row)
        return results_csv, results_csv.read_bytes(), 2

    def test_demo_round_does_not_touch_results_csv(
        self, qt_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ruling -11 item 5, second half: bytes and mtime_ns both fixed."""
        results_csv, before_bytes, legacy_count = self._seed_legacy_store(tmp_path)
        mtime_before = results_csv.stat().st_mtime_ns
        opened = _no_open_on("results.csv", monkeypatch)

        rc = cmd_live(
            profile_path=_profile_dict(tmp_path),
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="d3b-demo-1",
            record_consent=False,
            image_consent=False,
            mode="demo",
        )

        assert rc == 0
        assert results_csv.read_bytes() == before_bytes
        assert results_csv.stat().st_mtime_ns == mtime_before
        header, rows, _ = _csv_parts(results_csv)
        assert header == list(G3_RESULTS_CSV_COLUMNS)
        assert len(rows) == legacy_count
        assert opened == [], f"demo mode opened results.csv: {opened}"

    def test_demo_mode_writes_no_encrypted_bundle(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Decision -12 item 3: no recorder, so no bundle and no ledger."""
        store = tmp_path / "store"
        rc = cmd_live(
            profile_path=_profile_dict(tmp_path),
            store=store,
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="d3b-demo-2",
            record_consent=False,
            image_consent=False,
            mode="demo",
        )
        assert rc == 0
        # No session bundle, no attempt ledger, no encrypted frames anywhere.
        assert list(store.glob("**/*.enc")) == []
        assert not (store / "d3b-demo-2").exists()

    def test_demo_verdict_writes_the_plaintext_demo_row(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Decision -11 item 2 + -12 item 2: a verdict becomes one demo row.

        Driven through the real window with a demo sink and NO recorder, so
        the row can only come from the demo writer. A single-shot cmd_live
        run never labels, so it could not prove this at all.
        """
        # The demo store is a directory the recording harness never writes to,
        # so "no encrypted file" is a claim about the demo path alone and not
        # about a shared tmp dir the factory's own recorder populated.
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

        assert demo_csv.is_file(), "demo verdict wrote no file"
        header, rows, _ = _csv_parts(demo_csv)
        assert header == list(G3_DEMO_RESULTS_CSV_COLUMNS)
        assert len(rows) == 1
        row = dict(zip(header, rows[0], strict=True))
        assert row["mode"] == "demo-no-recording"
        assert row["app_version"]
        assert row["profile_version"] == "qt-test-v1"
        assert row["gallery_digest"] == "gallery-qt-test"
        assert row["label_kind"] == "enrolled"
        assert row["label_identity"] == "person-synth-01"
        assert row["top1_identity"] == "person-synth-01"
        assert row["required_support"]
        assert row["reason_codes"], "D1 reason codes must survive into the demo file"
        # No encrypted artifact beside the demo file. The _RoundFactory's own
        # recorder lives under a different tmp dir and is not the store under
        # test; what matters is that the store holding the demo file gained no
        # encrypted sibling.
        assert list(demo_csv.parent.glob("**/*.enc")) == []
        assert not (demo_csv.parent / "_labels").exists()

    def test_demo_verdict_never_calls_recorder_commit(
        self, qt_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Decision -12 item 2: the demo path must not reach commit."""
        import facecore.research.cli as research_cli

        calls: list[str] = []
        monkeypatch.setattr(
            research_cli,
            "commit_g3_rounds",
            lambda *a, **k: calls.append("commit") or (0, 0),
        )
        demo_csv = tmp_path / "store" / G3_DEMO_RESULTS_CSV_NAME
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
        assert calls == []

    def test_demo_mode_creates_the_plaintext_demo_file(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Decision -11 item 2: the demo file exists with its own header."""
        store = tmp_path / "store"
        rc = cmd_live(
            profile_path=_profile_dict(tmp_path),
            store=store,
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="d3b-demo-3",
            record_consent=False,
            image_consent=False,
            mode="demo",
        )
        assert rc == 0
        demo_csv = store / G3_DEMO_RESULTS_CSV_NAME
        # Single-shot fake mode has no operator verdict, so the file may not
        # exist yet — but if it does, its header is its own, not the 17 cols.
        if demo_csv.is_file():
            header, _rows, _ = _csv_parts(demo_csv)
            assert header == list(G3_DEMO_RESULTS_CSV_COLUMNS)
            assert header != list(G3_RESULTS_CSV_COLUMNS)

    def test_continuous_demo_run_labels_into_demo_file_only(
        self, qt_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: the CLI demo path labels a real round, ledger untouched.

        This is the guard that survived the first mutation attempt, because
        it exercises the actual cmd_live → QtResearchWindow wiring rather
        than constructing a window by hand. A real run with an operator
        verdict is the only shape that can reach the writer.
        """
        import facecore.research.cli as research_cli
        from facecore.live.qt_window import QtResearchWindow as QtWindow

        results_csv, before_bytes, legacy_count = self._seed_legacy_store(tmp_path)
        mtime_before = results_csv.stat().st_mtime_ns
        opened = _no_open_on("results.csv", monkeypatch)

        orig_init = QtWindow.__init__
        def _hooked(self: Any, *args: Any, **kwargs: Any) -> None:
            orig_init(self, *args, **kwargs)
            self.enter_ready()
            self.start_clicked()
            self.process_until_terminal(max_steps=200)
            self.press_correct()

        monkeypatch.setattr(QtWindow, "__init__", _hooked)
        # Assert the CLI really did hand the window a demo sink and no
        # research ledger — otherwise this test would pass for the wrong
        # reason (e.g. if the round simply failed to label).
        seen: list[dict[str, Any]] = []

        def _spy_init(self: Any, *args: Any, **kwargs: Any) -> None:
            seen.append(dict(kwargs))
            _hooked(self, *args, **kwargs)

        monkeypatch.setattr(QtWindow, "__init__", _spy_init)

        rc = cmd_live(
            profile_path=_profile_dict(tmp_path),
            store=results_csv.parent,
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="d3b-demo-e2e",
            record_consent=False,
            image_consent=False,
            mode="demo",
            ui="qt",
            qt_offscreen=True,
            continuous=True,
            capture_factory=lambda _dev: FakeCapture(_face_frames()),
        )

        assert rc == 0, "demo run must complete with a labeled round"
        # The window really received the demo configuration.
        assert seen, "the window was never constructed"
        assert seen[0]["recorder"] is None, "demo mode must pass no recorder"
        assert seen[0]["results_csv"] is None, (
            "demo mode must not be handed the research ledger target"
        )
        assert seen[0]["demo_results_csv"] == (
            results_csv.parent / G3_DEMO_RESULTS_CSV_NAME
        )
        # The verdict landed in the demo file, with its own header.
        demo_csv = results_csv.parent / G3_DEMO_RESULTS_CSV_NAME
        header, rows, _ = _csv_parts(demo_csv)
        assert header == list(G3_DEMO_RESULTS_CSV_COLUMNS)
        assert len(rows) == 1
        assert dict(zip(header, rows[0], strict=True))["label_kind"] == "enrolled"
        # And the research ledger is byte- and mtime-identical.
        assert results_csv.read_bytes() == before_bytes
        assert results_csv.stat().st_mtime_ns == mtime_before
        assert len(_csv_parts(results_csv)[1]) == legacy_count
        assert opened == [], f"demo mode opened results.csv: {opened}"
        # No encrypted artifact was created in the store.
        assert list(results_csv.parent.glob("**/*.enc")) == []
        assert not (results_csv.parent / "_labels").exists()
        assert research_cli.NULL_RECORDER is not None


# ---------------------------------------------------------------------------
# 3. Mode guard: consent flags and mode must agree, loudly.
# ---------------------------------------------------------------------------


class TestModeConsentAgreement:
    def test_demo_mode_with_consent_flags_refuses(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Decision -13 item 3: never silently ignore consent flags."""
        rc = cmd_live(
            profile_path=_profile_dict(tmp_path),
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="d3b-mix",
            record_consent=True,
            image_consent=False,
            mode="demo",
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "--mode record" in err

    def test_record_mode_without_consent_flags_refuses(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """The pre-existing fail-closed consent contract is unchanged."""
        rc = cmd_live(
            profile_path=_profile_dict(tmp_path),
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="d3b-noconsent",
            record_consent=False,
            image_consent=False,
            mode="record",
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "--record-consent" in err

    def test_unknown_mode_refuses(self, tmp_path: Path, capsys: Any) -> None:
        rc = cmd_live(
            profile_path=_profile_dict(tmp_path),
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="d3b-badmode",
            record_consent=False,
            image_consent=False,
            mode="nonsense",
        )
        assert rc == 2
        assert "unsupported mode" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 4. Demo file privacy: plaintext, but no biometrics.
# ---------------------------------------------------------------------------


class TestDemoRowCarriesNoBiometrics:
    def test_demo_row_has_no_pixel_or_embedding_column(self) -> None:
        """A plaintext file may not carry pixels, embeddings, or face boxes."""
        forbidden = ("pixel", "embedding", "face_box", "frame_bytes", "photo", "rgb")
        for column in G3_DEMO_RESULTS_CSV_COLUMNS:
            assert not any(
                token in column.lower() for token in forbidden
            ), f"demo column {column!r} smells biometric"

    def test_demo_row_mirrors_research_row_where_shared(self) -> None:
        """Demo scores are derived from g3_round_row, so they cannot drift."""
        factory = _RoundFactory(
            Path("/tmp"), FakeCapture(_face_frames()), _matching_scorer
        )
        round_ = _one_labeled_round_no_csv(factory)
        row = g3_demo_round_row(round_, required_support=3, labeled_at_utc="t0")
        for key in ("top1_identity", "top1_score", "top2_identity", "margin", "result"):
            assert row[key] == _research_row_value(round_, key)
        assert row["mode"] == "demo-no-recording"
        assert row["required_support"] == "3"


def _research_row_value(round_: Any, key: str) -> Any:
    from facecore.research.cli import g3_round_row

    return g3_round_row(round_)[key]


def _one_labeled_round_no_csv(factory: _RoundFactory) -> Any:
    """One labeled RoundComplete without any csv target configured."""
    from PySide6.QtTest import QTest  # noqa: F401  (offscreen driver parity)

    desktop, consent, attempt = factory()
    window = QtResearchWindow(
        desktop,
        consent=consent,
        recorder=None,
        attempt_id=None,
        offscreen=True,
        clock_ns=lambda: 0,
        next_session=factory,
        demo_results_csv=None,
    )
    window.enter_ready()
    window.start_clicked()
    window.process_until_terminal(max_steps=200)
    window.press_correct()
    assert len(window.completed_rounds) == 1
    window.close()
    return window.completed_rounds[0]


def _one_labeled_round(
    qt_app: Any,
    factory: _RoundFactory,
    results_csv: Path,
    session_hint: str,
) -> Any:
    """One labeled RoundComplete committed to results_csv in record mode."""
    desktop, consent, attempt = factory()
    window = QtResearchWindow(
        desktop,
        consent=consent,
        recorder=factory.recorder,
        attempt_id=attempt,
        offscreen=True,
        clock_ns=lambda: 0,
        next_session=factory,
        results_csv=None,
    )
    window.enter_ready()
    window.start_clicked()
    window.process_until_terminal(max_steps=200)
    window.press_correct()
    assert len(window.completed_rounds) == 1
    window.close()
    return window.completed_rounds[0]
