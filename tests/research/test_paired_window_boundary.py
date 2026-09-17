"""Paired-window closing-frame boundary alignment (evaluator side).

RED contract:
- realistic provenance: the ``CollectionWindow`` and the trace come from the
  REAL collector path (``cmd_live`` fixed-window run over a fake camera and
  test-double detector/embedder), so the last sampled frame is STRICTLY
  beyond ``collection_deadline_ns`` — exactly what a real camera produces.
  Before the fix that trace is downgraded to ``incomplete`` via
  ``frame_beyond_deadline`` + ``frame_count_mismatch``, making
  ``collection_extent == "full"`` — and therefore ``paired_complete`` —
  structurally unreachable for every real session.
- positive control: MULTIPLE beyond-deadline frames must STILL be refused.
  The fix touches the deadline check itself, so without this control a
  working fix and a disabled check are indistinguishable.
- ``frame_beyond_window`` (``collection_end_ns``) is a separate gate and
  must keep behaving exactly as before.

Existing paired tests (``test_paired_analysis.py:154``/``:828``,
``test_replay_clock.py``) hand-build windows whose deadline sits AFTER the
last frame — provenance a real collector never produces. They stay valid for
fixing the arithmetic, but only the realistic-provenance test below can keep
this defect from coming back.

Repo holds synthetic only; no camera is opened.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from facecore.live.capture import CaptureSource
from facecore.live.contracts import FramePacket
from facecore.research.analysis import analyze_batch
from facecore.research.cli import _load_profile, cmd_live
from facecore.research.experiment import EvaluationLabel
from facecore.research.recorder import ResearchRecorder
from facecore.research.replay import evaluate_arms

EXPERIMENT_ID = "exp-window-boundary"


def _clock() -> datetime:
    return datetime.now(timezone.utc)


def _profile_path(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "window-boundary-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 25,
        "queue_limit": 1,
        "required_support": 3,
        "min_support_interval_ms": 200,
        "match_threshold": 0.45,
        "review_threshold": 0.30,
        "margin_threshold": 0.10,
        "detector_version": "yunet-test",
        "quality_policy_version": "q-test-v1",
        "continuity_max_center_delta_ratio": 0.50,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


def _synthetic_png(path: Path, seed: int) -> None:
    from PIL import Image

    arr = np.full((200, 200, 3), 120, dtype=np.uint8)
    arr[::2, ::2] = 160
    arr[1::2, 1::2] = 80
    arr[20:80, 20:80, 0] = (150 + seed) % 256
    Image.fromarray(arr, mode="RGB").save(path)


def _corpus_manifest(tmp_path: Path) -> Path:
    p1 = tmp_path / "enroll_01.png"
    p2 = tmp_path / "enroll_02.png"
    _synthetic_png(p1, 1)
    _synthetic_png(p2, 2)
    manifest = {
        "files": [
            {
                "path": str(p1),
                "role": "enrollment",
                "identity": "person-01",
                "conditions": [],
            },
            {
                "path": str(p2),
                "role": "enrollment",
                "identity": "person-02",
                "conditions": [],
            },
        ]
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(manifest))
    return path


def _mock_detector() -> MagicMock:
    from facecore.pipeline.detect import DetectedFace

    landmarks = (
        (50.0, 60.0),
        (110.0, 60.0),
        (80.0, 90.0),
        (60.0, 115.0),
        (100.0, 115.0),
    )
    face = DetectedFace(
        box=(20.0, 20.0, 120.0, 120.0), landmarks=landmarks, confidence=0.99
    )
    detector = MagicMock()
    detector.detect.return_value = [face]
    return detector


class _ScriptedEmbedder:
    """Gallery build consumes the first two embeds; live frames get the rest.

    Orthogonal unit vectors make the cosine score exactly the component
    along each enrolled identity, so the terminal is chosen by arithmetic
    rather than by image content.
    """

    model_version = "sface_2021dec"

    def __init__(self, live_vec: tuple[float, float, float]) -> None:
        self._calls = 0
        self._live = live_vec

    def embed(self, crop: Any) -> tuple[np.ndarray, str]:
        self._calls += 1
        if self._calls == 1:
            vec = (1.0, 0.0, 0.0)
        elif self._calls == 2:
            vec = (0.0, 1.0, 0.0)
        else:
            vec = self._live
        arr = np.array(vec, dtype=np.float32)
        return arr / float(np.linalg.norm(arr)), "sface_2021dec"


class _SteppedCamera(CaptureSource):
    """Fake camera anchored on first read, advancing 200ms per frame.

    The anchor is taken lazily so stamps land after the session start the
    way a real device does — which is precisely why the closing frame ends
    up marginally beyond the deadline.
    """

    def __init__(self, count: int) -> None:
        self._lock = threading.Lock()
        self._count = count
        self._seq = 0
        self._anchor: int | None = None
        self._opened = False
        self._closed = False

    def open(self, device_id: str) -> None:
        with self._lock:
            self._opened = True
            self._closed = False
            self._seq = 0
            self._anchor = None

    def read(self) -> FramePacket | None:
        with self._lock:
            if self._closed or not self._opened or self._seq >= self._count:
                return None
            if self._anchor is None:
                self._anchor = time.monotonic_ns()
            self._seq += 1
            rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
            rgb[::2, ::2] = 160
            rgb[1::2, 1::2] = 80
            rgb[0, 0, 0] = self._seq % 256
            return FramePacket(
                sequence=self._seq,
                captured_ns=self._anchor + self._seq * 200_000_000,
                rgb=rgb,
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._opened = False

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


def _run_real_fixed_window_session(
    tmp_path: Path, session_id: str
) -> tuple[ResearchRecorder, Any, Any, Any]:
    """Drive the REAL chain: CLI -> fake camera -> real adapter -> engine.

    Returns (recorder, profile, window, trace) with provenance produced by
    the production collector, never hand-built.
    """
    profile_path = _profile_path(tmp_path)
    corpus = _corpus_manifest(tmp_path)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    store = tmp_path / "store"
    key_dir = tmp_path / "research_keys"

    rc = cmd_live(
        profile_path=profile_path,
        store=store,
        key_dir=key_dir,
        device="0",
        session_id=session_id,
        record_consent=True,
        image_consent=True,
        fixed_seconds=True,
        models=models,
        corpus=corpus,
        capture_factory=lambda _device: _SteppedCamera(30),
        detector_factory=lambda _models: _mock_detector(),
        embedder_factory=lambda _models: _ScriptedEmbedder((0.97, 0.20, 0.10)),
        experiment_id=EXPERIMENT_ID,
        attempt_id=f"att-{session_id}",
    )
    assert rc == 0

    recorder = ResearchRecorder(
        store_root=store, key_dir=key_dir, clock=_clock
    )
    window = recorder.read_record(session_id).collection_window
    trace = recorder.read_trace(f"att-{session_id}")
    profile = _load_profile(profile_path)
    return recorder, profile, window, trace


class TestRealisticProvenanceClosingFrame:
    """The collector's own closing frame must not void the paired window."""

    def test_real_fixed_window_run_yields_full_extent(
        self, tmp_path: Path
    ) -> None:
        _rec, profile, window, trace = _run_real_fixed_window_session(
            tmp_path, "sess-window-full"
        )

        # Provenance guard: this is the real collector's output, and its
        # closing frame is STRICTLY beyond the deadline. If this ever stops
        # holding, the test has lost the property it exists to protect.
        assert window.collection_complete is True
        assert window.collection_stop_reason == "deadline_reached"
        last_ns = max(e.captured_ns for e in trace.entries)
        assert last_ns > window.collection_deadline_ns
        assert len(trace.entries) == window.frames_sampled

        arm_a, arm_b = evaluate_arms(trace, profile, window=window)

        for arm in (arm_a, arm_b):
            assert "frame_beyond_deadline" not in arm.decision_codes
            assert "frame_count_mismatch" not in arm.decision_codes
            assert arm.collection_extent == "full"
            # The closing frame stays in the evaluated set.
            assert arm.frames_scored == window.frames_sampled

    def test_real_run_counts_toward_paired_complete(
        self, tmp_path: Path
    ) -> None:
        recorder, profile, window, trace = _run_real_fixed_window_session(
            tmp_path, "sess-window-paired"
        )
        attempt_id = "att-sess-window-paired"
        arm_a, arm_b = evaluate_arms(trace, profile, window=window)

        recorder.write_label(
            EvaluationLabel(
                attempt_id=attempt_id,
                revision=1,
                kind="enrolled",
                identity_id="person-01",
                actor_ref="boundary-evaluator",
                labeled_at=_clock().isoformat(),
            )
        )
        attempts = [
            a
            for a in recorder.list_attempts(EXPERIMENT_ID)
            if a.attempt_id == attempt_id
        ]
        assert len(attempts) == 1

        report = analyze_batch(
            attempts,
            [arm_a, arm_b],
            [recorder.read_label(attempt_id)],
            profile=profile,
        )
        assert report.attempted == 1
        assert report.paired_complete == 1


class TestBeyondDeadlinePositiveControls:
    """Only the single closing frame is exempt; the gate still holds."""

    def test_multiple_beyond_deadline_frames_still_refused(
        self, tmp_path: Path
    ) -> None:
        _rec, profile, window, trace = _run_real_fixed_window_session(
            tmp_path, "sess-window-multi"
        )
        deadline = window.collection_deadline_ns
        beyond = [e for e in trace.entries if e.captured_ns > deadline]
        assert len(beyond) == 1, "real collector emits exactly one closing frame"

        # Drag the penultimate frame past the deadline too: now TWO frames
        # are beyond it, so the closing-frame exemption must not apply.
        entries = list(trace.entries)
        ordered = sorted(entries, key=lambda e: e.captured_ns)
        second_last = ordered[-2]
        bumped = replace(second_last, captured_ns=deadline + 1)
        entries[entries.index(second_last)] = bumped
        tampered = replace(trace, entries=tuple(entries))

        arm_a, arm_b = evaluate_arms(tampered, profile, window=window)
        for arm in (arm_a, arm_b):
            assert "frame_beyond_deadline" in arm.decision_codes
            assert arm.collection_extent == "incomplete"

    def test_beyond_window_gate_is_unchanged(self, tmp_path: Path) -> None:
        _rec, profile, window, trace = _run_real_fixed_window_session(
            tmp_path, "sess-window-end"
        )
        ordered = sorted(trace.entries, key=lambda e: e.captured_ns)
        # collection_end_ns is a separate gate: a closing frame past the
        # actual end of collection stays a violation, exemption or not.
        bounded = replace(window, collection_end_ns=ordered[-2].captured_ns)

        arm_a, arm_b = evaluate_arms(trace, profile, window=bounded)
        for arm in (arm_a, arm_b):
            assert "frame_beyond_window" in arm.decision_codes
            assert arm.collection_extent == "incomplete"


class TestIncompleteWindowsKeepTheGate:
    """The exemption is tied to proven deadline_reached provenance."""

    @pytest.mark.parametrize(
        ("complete", "stop_reason"),
        [
            (False, "max_frames_reached"),
            (False, "source_exhausted"),
            (True, "max_frames_reached"),
        ],
    )
    def test_non_deadline_windows_still_flag_beyond_deadline(
        self, tmp_path: Path, complete: bool, stop_reason: str
    ) -> None:
        _rec, profile, window, trace = _run_real_fixed_window_session(
            tmp_path, f"sess-window-{stop_reason}-{int(complete)}"
        )
        downgraded = replace(
            window, collection_complete=complete, collection_stop_reason=stop_reason
        )

        arm_a, arm_b = evaluate_arms(trace, profile, window=downgraded)
        for arm in (arm_a, arm_b):
            assert "frame_beyond_deadline" in arm.decision_codes
            assert arm.collection_extent == "incomplete"


def test_record_ttl_unaffected_by_boundary_alignment(tmp_path: Path) -> None:
    """Guard: the fix touches evaluation only, never retention."""
    recorder, _profile, _window, _trace = _run_real_fixed_window_session(
        tmp_path, "sess-window-ttl"
    )
    future = _clock() + timedelta(days=31)
    purged = recorder.purge_expired(future)
    assert "sess-window-ttl" in purged
