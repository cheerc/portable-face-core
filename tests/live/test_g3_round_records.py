"""G3 W4 RED: per-round records queue + results.csv + 30-day retention.

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md §3 (one
record per round, results.csv without images/embeddings, 30-day
retention) and §7 item 4 (labeled records, data survives restart,
results.csv consistent with the store rounds).

All tests use synthetic frames, FakeCapture, an offscreen Qt
application, and a tmp ResearchRecorder. No camera, real faces,
gallery, embeddings, or photos.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import FramePacket
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.live.qt_window import QtResearchWindow, RoundComplete
from facecore.research.recorder import ResearchRecorder
from tests.live.test_qt_window import (
    _AdvancingClock,
    _attempt,
    _consent,
    _manifest,
    _matching_scorer,
    _profile,
    _recorder,
)


@pytest.fixture(scope="module")
def qt_app() -> Any:
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication as ActualQApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return ActualQApplication.instance() or ActualQApplication([])


def _face_frames(n: int = 60) -> list[FramePacket]:
    frames = []
    for seq in range(1, n + 1):
        frame = np.full((8, 8, 3), 150, dtype=np.uint8)
        frames.append(
            FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)
        )
    return frames


class _RoundFactory:
    """Recorder-bound round sessions over one shared source (W3 pattern)."""

    def __init__(
        self,
        tmp_path: Path,
        source: FakeCapture,
        scorer: Any,
        prefix: str = "g3w4",
    ) -> None:
        self.recorder = _recorder(tmp_path)
        self.manifest = _manifest()
        self.source = source
        self.scorer = scorer
        self.prefix = prefix
        self.counter = 0
        self.attempt_ids: list[str] = []

    def __call__(self) -> tuple[DesktopSession, Any, str]:
        self.counter += 1
        session_id = f"{self.prefix}-round-{self.counter}"
        attempt = _attempt(f"attempt-{self.prefix}-{self.counter}")
        consent = _consent(session_id)
        self.recorder.begin_attempt(self.manifest, attempt, consent)
        self.recorder.begin(session_id, consent)
        self.attempt_ids.append(attempt.attempt_id)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-qt-test", "gen-qt-test"),
            source=self.source,
            scorer=self.scorer,
            session_id=session_id,
            release_source_on_terminal=False,
            label_recorder=self.recorder,
            label_attempt_id=attempt.attempt_id,
        )
        return desktop, consent, attempt.attempt_id


def _open_window(qt_app: Any, factory: _RoundFactory) -> QtResearchWindow:
    first_desktop, first_consent, first_attempt = factory()
    clock = _AdvancingClock()
    window = QtResearchWindow(
        first_desktop,
        consent=first_consent,
        recorder=factory.recorder,
        attempt_id=first_attempt,
        offscreen=True,
        clock_ns=clock,
        clock_advance=clock.advance,
        next_session=factory,
    )
    window.show()
    window.enter_ready()
    window.start_clicked()
    return window


def _commit_round(
    recorder: ResearchRecorder, results_csv: Path, round_: RoundComplete
) -> None:
    """Production commit path under test (bundle + attempt + csv row)."""
    from facecore.research.cli import commit_g3_rounds

    committed, failed = commit_g3_rounds(recorder, results_csv, [round_])
    assert (committed, failed) == (1, 0)


class TestG3RoundRecords:
    def test_key_queues_round_complete_material(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """A key press queues full round material for the CLI tail."""
        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), _matching_scorer)
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        round_attempt = factory.attempt_ids[-1]
        window.press_correct()
        assert window.mode == "ready"
        assert len(window.completed_rounds) == 1
        queued = window.completed_rounds[0]
        assert queued.session_id.startswith("g3w4-round-")
        assert queued.attempt_id == round_attempt
        assert queued.terminal.status.value == "matched"
        assert queued.label_kind == "enrolled"
        assert queued.label_identity == "person-synth-01"
        assert queued.profile_version == "qt-test-v1"
        assert queued.started_utc != ""
        assert len(queued.observations) > 0
        window.close()

    def test_two_rounds_csv_matches_store(self, qt_app: Any, tmp_path: Path) -> None:
        """Two labeled rounds → two commits + two csv rows, no images."""
        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), _matching_scorer)
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        window.press_correct()
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        window.press_incorrect()
        assert len(window.completed_rounds) == 2
        results_csv = tmp_path / "results.csv"
        for round_ in window.completed_rounds:
            _commit_round(factory.recorder, results_csv, round_)
        with open(results_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["label_kind"] == "enrolled"
        assert rows[1]["label_kind"] == "uncertain"
        assert rows[0]["top1_identity"] == "person-synth-01"
        assert float(rows[0]["top1_score"]) == pytest.approx(0.9)
        assert float(rows[0]["margin"]) == pytest.approx(0.8)
        assert rows[0]["profile_version"] == "qt-test-v1"
        assert rows[0]["model_generation"] == "gen-qt-test"
        # No images or embeddings in the csv surface.
        blob = results_csv.read_text()
        assert "embedding" not in blob.lower()
        assert rows[0]["session_id"] != ""
        # Store holds both committed bundles.
        for round_ in window.completed_rounds:
            record = factory.recorder.read_record(round_.session_id)
            assert record.result.status.value == round_.terminal.status.value
        window.close()


def _g3_profile_json(tmp_path: Path) -> Path:
    import json as _json

    profile = {
        "schema_version": "v1",
        "profile_version": "g3-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.363,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet",
        "quality_policy_version": "1",
        "continuity_max_center_delta_ratio": 0.50,
    }
    path = tmp_path / "g3-profile.json"
    path.write_text(_json.dumps(profile))
    return path


class TestG3ContinuousCloseout:
    def test_close_without_key_is_rc4_and_cancelled(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """No labeled round at close → rc4, staged sessions aborted."""
        from datetime import datetime, timezone

        from facecore.research.cli import cmd_live
        from facecore.research.recorder import ResearchRecorder

        store = tmp_path / "store"
        keys = tmp_path / "keys"
        rc = cmd_live(
            profile_path=_g3_profile_json(tmp_path),
            store=store,
            key_dir=keys,
            device="fake",
            session_id="g3w4-smoke-nokey",
            record_consent=True,
            image_consent=True,
            ui="qt",
            qt_offscreen=True,
            continuous=True,
        )
        assert rc == 4
        rec = ResearchRecorder(
            store_root=store,
            key_dir=keys,
            clock=lambda: datetime.now(timezone.utc),
        )
        attempts = rec.list_attempts(experiment_id="exp-cli-e3")
        assert attempts
        for attempt in attempts:
            assert attempt.operational_status == "cancelled"

    def test_retention_is_30_days(self, qt_app: Any, tmp_path: Path) -> None:
        """G3 consent TTLs are 30d record + 30d image (spec §3)."""
        import json as _json
        from datetime import datetime

        from facecore.research.cli import cmd_live

        store = tmp_path / "store"
        keys = tmp_path / "keys"
        rc = cmd_live(
            profile_path=_g3_profile_json(tmp_path),
            store=store,
            key_dir=keys,
            device="fake",
            session_id="g3w4-smoke-ttl",
            record_consent=True,
            image_consent=True,
        )
        assert rc == 0
        manifest = _json.loads((store / "g3w4-smoke-ttl" / "manifest.json").read_text())
        created = datetime.fromisoformat(manifest["created_at_utc"])
        record_exp = datetime.fromisoformat(manifest["record_expires_at_utc"])
        image_exp = datetime.fromisoformat(manifest["image_expires_at_utc"])
        assert (record_exp - created).days == 30
        assert (image_exp - created).days == 30


class TestG3ReopenContinuity:
    def test_reopen_appends_without_losing_rows(
        self, qt_app: Any, tmp_path: Path
    ) -> None:
        """Reopening continues results.csv; store rounds stay consistent."""
        from facecore.research.cli import commit_g3_rounds

        results_csv = tmp_path / "results.csv"
        first_rows: list[dict[str, str]] = []
        factory = _RoundFactory(tmp_path, FakeCapture(_face_frames()), _matching_scorer)
        window = _open_window(qt_app, factory)
        window.process_until_terminal(max_steps=200)
        window.press_correct()
        for round_ in window.completed_rounds:
            committed, failed = commit_g3_rounds(
                factory.recorder, results_csv, [round_]
            )
            assert (committed, failed) == (1, 0)
        with open(results_csv, newline="") as handle:
            first_rows = list(csv.DictReader(handle))
        assert len(first_rows) == 1
        window.close()
        # Reopen: a fresh window over the same store continues the csv.
        factory2 = _RoundFactory(
            tmp_path,
            FakeCapture(_face_frames()),
            _matching_scorer,
            prefix="g3w4-reopen",
        )
        window2 = _open_window(qt_app, factory2)
        window2.process_until_terminal(max_steps=200)
        window2.press_incorrect()
        for round_ in window2.completed_rounds:
            committed, failed = commit_g3_rounds(
                factory2.recorder, results_csv, [round_]
            )
            assert (committed, failed) == (1, 0)
        with open(results_csv, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0] == first_rows[0]
        assert rows[1]["label_kind"] == "uncertain"
        window2.close()
