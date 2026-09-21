"""E3 RED: fixed-window collector separation + two real-entrypoint wirings.

Source of truth:
    - spec §4.2 (fixed window + two arms) + §5 (attempt before camera open);
    - plan §12 E3 + §10 gap table; ADR 0010.

RED contract (must fail on current base, behavioural not import errors):
    - LiveController has no fixed-window separation: B terminal currently
      stops the collector (early-stop only). Fixed mode must keep collecting
      to the original deadline after B locks, without re-sending to B.
    - research/cli.py has no attempt pre-placement: begin_attempt is absent
      from the true caller path, so camera-open failure leaves no attempt.
    - E2 sinks have no production caller: score_frame diagnostic_sink and
      SessionEngine event_sink are wired as parameters but nothing in src/
      passes them; trace append has no live driver.

Only synthetic payloads; never real faces; camera-free.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import (
    DecisionEvent,
    FrameDiagnostics,
    FrameObservation,
    FramePacket,
    ResearchProfile,
    SessionStatus,
)
from facecore.live.controller import LiveController
from facecore.live.session import SessionEngine
from facecore.research.cli import cmd_live
from facecore.research.experiment import ExperimentManifest
from facecore.research.records import ConsentRecord
from facecore.research.recorder import ResearchRecorder
from tests.conftest import assert_no_plaintext_leak


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _profile(
    *,
    timeout_ms: int = 5000,
    sample_interval_ms: int = 200,
    max_frames: int = 26,
    required_support: int = 3,
    min_support_interval_ms: int = 200,
) -> ResearchProfile:
    return ResearchProfile(
        schema_version="v1",
        profile_version="prof-e3-001",
        timeout_ms=timeout_ms,
        sample_interval_ms=sample_interval_ms,
        max_frames=max_frames,
        queue_limit=1,
        required_support=required_support,
        min_support_interval_ms=min_support_interval_ms,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-e3-1",
        quality_policy_version="qual-e3-1",
        continuity_max_center_delta_ratio=0.50,
    )


def _packet(seq: int, captured_ns: int) -> FramePacket:
    rgb = np.ascontiguousarray(
        np.full((16, 16, 3), 120 + (seq % 40), dtype=np.uint8)
    )
    return FramePacket(sequence=seq, captured_ns=captured_ns, rgb=rgb)


def _matched_obs(
    seq: int,
    captured_ns: int,
    identity: str = "person-01",
    score: float = 0.85,
    runner_up: float = 0.20,
) -> FrameObservation:
    return FrameObservation(
        sequence=seq,
        captured_ns=captured_ns,
        processed_ns=captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(4.0, 4.0, 8.0, 8.0),
        identity_scores={identity: score, "person-02": runner_up},
        quality_rank=0.5,
        model_generation="gen-e3",
        gallery_digest="gal-e3",
    )


def _profile_dict(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": "t8-cli-v1",
        "timeout_ms": 5000,
        "sample_interval_ms": 200,
        "max_frames": 26,
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


class TestFixedWindowCollectorSeparation:
    """B inference terminal locks; collector keeps sampling to deadline."""

    def test_b_matched_at_0_4s_still_collects_to_5s_boundary(self) -> None:
        profile = _profile(required_support=1, max_frames=26)
        # Build a source covering 0..5s at 200ms spacing (26 frames).
        frames = [
            _packet(seq, (seq - 1) * 200_000_000) for seq in range(1, 28)
        ]

        def scorer(packet: FramePacket) -> FrameObservation:
            return _matched_obs(packet.sequence, packet.captured_ns)

        engine = SessionEngine(profile, "gal-e3", "gen-e3")
        controller = LiveController(
            engine,
            FakeCapture(frames=frames),
            scorer,
            fixed_seconds=True,
        )
        controller.start_session("sess-fixed-001", 0)
        terminal = controller.run_until_terminal(max_steps=100)
        assert terminal is not None
        assert terminal.status == SessionStatus.matched
        # B locked early (required_support=1 → seq 1).
        assert terminal.support_sequences == (1,)
        # Collector evidence: sampled to the 5s deadline boundary.
        assert controller.frames_sampled == 26
        assert controller.collection_complete is True
        assert controller.collection_stop_reason == "deadline_reached"
        # B result was never rewritten by post-lock frames.
        assert controller.inference_terminal is terminal
        assert controller.inference_terminal.support_sequences == (1,)

    def test_post_lock_frames_available_to_arm_a_not_b(self) -> None:
        profile = _profile(required_support=1)
        frames = [
            _packet(seq, (seq - 1) * 200_000_000) for seq in range(1, 27)
        ]

        def scorer(packet: FramePacket) -> FrameObservation:
            # Early frames match person-01; late frames match person-02.
            # B must keep person-01 after locking; A sees the late person-02.
            if packet.sequence <= 2:
                return _matched_obs(
                    packet.sequence, packet.captured_ns, "person-01"
                )
            return _matched_obs(
                packet.sequence, packet.captured_ns, "person-02"
            )

        engine = SessionEngine(profile, "gal-e3", "gen-e3")
        controller = LiveController(
            engine,
            FakeCapture(frames=frames),
            scorer,
            fixed_seconds=True,
        )
        controller.start_session("sess-fixed-002", 0)
        terminal = controller.run_until_terminal(max_steps=100)
        assert terminal is not None
        assert terminal.matched_identity == "person-01"
        # Post-lock observations went to the collector, not back into B.
        late_identities = [
            obs.identity_scores for obs in controller.post_lock_observations
        ]
        assert late_identities, "expected post-lock collector observations"
        top_late = max(late_identities[-1].items(), key=lambda kv: kv[1])[0]
        assert top_late == "person-02"

    def test_full_25_frames_does_not_fabricate_5s_evidence(self) -> None:
        """Negative profile case: 5000/200/25 is now fail-closed (#84).

        The 25-frame triple is exactly the rejected class under the
        fixed-window cross-field invariant (needs 26), so construction
        itself must raise with the required minimum — the collector
        never gets to run. (Pre-invariant, this test pinned the
        max_frames_reached-at-4.8s execution path; non-fixed execution-layer
        cap coverage remains in `test_execution_max_frames_cap_enforced`.)
        """
        with pytest.raises(ValueError, match="26"):
            _profile(timeout_ms=5000, max_frames=25)

    def test_cancel_close_multi_face_stop_collector_incomplete(self) -> None:
        profile = _profile(required_support=3)
        frames = [
            _packet(seq, (seq - 1) * 200_000_000) for seq in range(1, 27)
        ]
        engine = SessionEngine(profile, "gal-e3", "gen-e3")
        controller = LiveController(
            engine,
            FakeCapture(frames=frames),
            lambda p: _matched_obs(p.sequence, p.captured_ns),
            fixed_seconds=True,
        )
        controller.start_session("sess-fixed-004", 0)
        controller.cancel_collection(400_000_000)
        assert controller.collection_complete is False
        assert controller.collection_stop_reason == "cancelled"

    def test_safety_monitoring_after_lock_still_detects_multi_face(
        self,
    ) -> None:
        profile = _profile(required_support=1)
        frames = [
            _packet(seq, (seq - 1) * 200_000_000) for seq in range(1, 27)
        ]

        def scorer(packet: FramePacket) -> FrameObservation:
            if packet.sequence >= 5:
                return FrameObservation(
                    sequence=packet.sequence,
                    captured_ns=packet.captured_ns,
                    processed_ns=packet.captured_ns + 1_000_000,
                    quality_pass=False,
                    quality_reasons=("input_multiple_faces",),
                    face_count=2,
                    face_box=None,
                    identity_scores={},
                    quality_rank=0.0,
                    model_generation="gen-e3",
                    gallery_digest="gal-e3",
                )
            return _matched_obs(packet.sequence, packet.captured_ns)

        engine = SessionEngine(profile, "gal-e3", "gen-e3")
        controller = LiveController(
            engine,
            FakeCapture(frames=frames),
            scorer,
            fixed_seconds=True,
        )
        controller.start_session("sess-fixed-005", 0)
        terminal = controller.run_until_terminal(max_steps=100)
        assert terminal is not None
        # B locked on person-01 before the multi-face frames arrived.
        assert terminal.matched_identity == "person-01"
        # Collector safety monitoring flagged the later multi-face input.
        assert "input_multiple_faces" in controller.collection_safety_flags


class TestCliAttemptPreplacement:
    """Attempt ledger is durable before camera open / model setup."""

    def test_camera_open_failure_still_leaves_one_attempt(
        self, tmp_path: Path
    ) -> None:
        profile_path = _profile_dict(tmp_path)
        store = tmp_path / "store"
        key_dir = tmp_path / "keys"

        def _failing_capture(device: str) -> Any:
            class _FailOpen:
                def open(self, device_id: str) -> None:
                    raise RuntimeError("injected camera open failure")

                def read(self) -> None:
                    return None

                def close(self) -> None:
                    pass

                @property
                def is_closed(self) -> bool:
                    return True

            return _FailOpen()

        rc = cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=key_dir,
            device="cam-9",
            session_id="sess-e3-openfail",
            record_consent=True,
            image_consent=True,
            models=tmp_path / "models",
            corpus=tmp_path / "corpus.json",
            capture_factory=_failing_capture,
        )
        assert rc == 2
        # The accepted Start left exactly one durable attempt despite the
        # camera failing before any scoring. The attempt id is namespaced
        # (att- prefix) so its rk_ DEK never collides with session staging.
        rec = ResearchRecorder(
            store_root=store, key_dir=key_dir, clock=datetime.now
        )
        attempts = rec.list_attempts(experiment_id="exp-cli-e3")
        assert [a.attempt_id for a in attempts] == ["att-sess-e3-openfail"]
        assert attempts[0].operational_status == "open_error"

    def test_fixed_fake_cli_closes_incomplete_collection_without_crash(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """E7-B r2 F1: seven fake frames leave an explicit incomplete window."""
        profile_path = _profile_dict(tmp_path)
        store = tmp_path / "store"
        key_dir = tmp_path / "research_keys"
        rc = cmd_live(
            profile_path=profile_path,
            store=store,
            key_dir=key_dir,
            device="fake",
            session_id="sess-e7-fixed-fake",
            record_consent=True,
            image_consent=True,
            fixed_seconds=True,
        )

        assert rc == 0
        lines = capsys.readouterr().out.splitlines()
        payload = json.loads(lines[-1])
        assert payload["window"] == "fixed-window-incomplete"
        recorder = ResearchRecorder(
            store_root=store,
            key_dir=key_dir,
            clock=lambda: datetime.now(timezone.utc),
        )
        record = recorder.read_record("sess-e7-fixed-fake")
        assert record.collection_window is not None
        assert record.collection_window.collection_complete is False


class TestTraceSinkLiveWiring:
    """Production path drives diagnostic/event sinks and trace append."""

    def test_controller_drives_sinks_and_trace_append(
        self, tmp_path: Path
    ) -> None:
        now = _utc("2026-09-16T10:00:00Z")
        store = tmp_path / "store"
        key_dir = tmp_path / "keys"
        rec = ResearchRecorder(
            store_root=store, key_dir=key_dir, clock=lambda: now
        )
        manifest = ExperimentManifest.from_dict(
            {
                "identity": {"experiment_id": "exp-e3-live"},
                "software": {},
                "gallery": {},
                "policy": {},
                "capture": {},
                "privacy": {},
                "study": {},
                "analysis": {},
            }
        )
        from facecore.research.experiment import AttemptRecord

        attempt = AttemptRecord(
            experiment_id="exp-e3-live",
            attempt_id="att-e3-live-001",
            participant_id="part-synth-e3",
            visit_id="visit-001",
            condition_id="cond-001",
            attempt_index=1,
            retry_of=None,
            consent_ref="consent-e3",
            requested_at_utc="2026-09-16T10:00:00Z",
            accepted_at_utc="2026-09-16T10:00:01Z",
            started_at_utc="2026-09-16T10:00:02Z",
            ended_at_utc=None,
            operational_status="accepted",
            error_code=None,
            bundle_ref=None,
        )
        consent = ConsentRecord(
            session_id="sess-e3-live",
            participant_id="part-synth-e3",
            record_consent=True,
            image_consent=True,
            consented_at_utc="2026-09-16T10:00:00Z",
            record_expires_at_utc="2026-10-16T10:00:00Z",
            image_expires_at_utc="2026-09-23T10:00:00Z",
        )
        rec.begin_attempt(manifest, attempt, consent)

        profile = _profile(required_support=1)
        frames = [_packet(seq, (seq - 1) * 200_000_000) for seq in range(1, 6)]
        diags: list[FrameDiagnostics] = []
        events: list[DecisionEvent] = []

        from facecore.live.frame_pipeline import (
            ResearchGallery,
            ScoringContext,
            score_frame,
        )
        from facecore.contracts.policy import PolicyProfile
        from facecore.pipeline.detect import DetectedFace

        face = DetectedFace(
            box=(20.0, 20.0, 120.0, 120.0),
            landmarks=(
                (50.0, 60.0),
                (110.0, 60.0),
                (80.0, 90.0),
                (60.0, 115.0),
                (100.0, 115.0),
            ),
            confidence=0.99,
        )
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [face]
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = (
            np.array([1.0, 0.0], dtype=np.float32),
            "sface_2021dec",
        )
        gallery = ResearchGallery(
            embeddings={"person-01": np.array([1.0, 0.0], dtype=np.float32)},
            model_version="sface_2021dec",
            generation="gen-e3",
            digest="gal-e3",
        )
        ctx = ScoringContext(
            gallery=gallery,
            model_version="sface_2021dec",
            policy=PolicyProfile.frozen_v1().with_thresholds(
                match_threshold=0.45,
                review_threshold=0.30,
                margin_threshold=0.10,
            ),
            detector=mock_detector,
            embedder=mock_embedder,
        )

        def scorer(packet: FramePacket) -> FrameObservation:
            return score_frame(
                packet, ctx, diagnostic_sink=diags.append
            )

        engine = SessionEngine(
            profile, "gal-e3", "gen-e3", event_sink=events.append
        )
        controller = LiveController(
            engine,
            FakeCapture(frames=frames),
            scorer,
            trace_recorder=rec,
            trace_attempt_id="att-e3-live-001",
        )
        controller.start_session("sess-e3-live", 0)
        terminal = controller.run_until_terminal(max_steps=50)
        assert terminal is not None
        # Production path drove both sinks and persisted trace frames.
        assert len(diags) >= 1
        assert len(events) >= 1
        trace = rec.read_trace("att-e3-live-001")
        assert len(trace.entries) == len(diags)
        # Byte-level at-rest check over the parent root (store + keys).
        assert_no_plaintext_leak(
            tmp_path,
            ["part-synth-e3", "person-01", "consent-e3"],
        )
