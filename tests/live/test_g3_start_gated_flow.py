"""G3 R1 PR-A RED: per-round data + clock + stepping (Start-gated).

Source of truth: docs/specs/2026-09-24-g3-local-test-app.md R1 §2-3
(Start-gated: clock from this round's post-open monotonic clock) and §3
(per-round staging/crop/trace bound to the round's attempt/session),
plus docs/plans/2026-09-24-g3-start-gated-repair-plan.md D1 manifest
(frozen changes 2/3/4) and PR-A acceptance.

All tests use synthetic frames, FakeCapture, an offscreen Qt
application, and a tmp ResearchRecorder. No camera, real faces,
gallery, embeddings, or photos.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from facecore.live.capture import FakeCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.live.qt_window import QtResearchWindow
from tests.live.test_qt_window import (
    _AdvancingClock,
    _attempt,
    _consent,
    _manifest,
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
        frame = np.full((16, 16, 3), 150, dtype=np.uint8)
        frames.append(
            FramePacket(sequence=seq, captured_ns=seq * 200_000_000, rgb=frame)
        )
    return frames


def _matching_scorer(packet: FramePacket) -> FrameObservation:
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=True,
        quality_reasons=(),
        face_count=1,
        face_box=(0.0, 0.0, 2.0, 2.0),
        identity_scores={"person-synth-01": 0.9, "person-synth-02": 0.1},
        quality_rank=0.9,
        model_generation="gen-g3a-test",
        gallery_digest="gallery-g3a-test",
    )


def _low_quality_scorer(packet: FramePacket) -> FrameObservation:
    """Face present but unusable: the round must not end before 5 s."""
    return FrameObservation(
        sequence=packet.sequence,
        captured_ns=packet.captured_ns,
        processed_ns=packet.captured_ns + 1_000_000,
        quality_pass=False,
        quality_reasons=("no_face_detected",),
        face_count=0,
        face_box=None,
        identity_scores={},
        quality_rank=0.0,
        model_generation="gen-g3a-test",
        gallery_digest="gallery-g3a-test",
    )


class _RoundFactory:
    """Recorder-bound round sessions mirroring the production factory."""

    def __init__(self, tmp_path: Any, source: FakeCapture) -> None:
        self.recorder = _recorder(tmp_path)
        self.manifest = _manifest()
        self.source = source
        self.counter = 0
        self.attempt_ids: list[str] = []
        self.retired_initial: list[str] = []

    def retire_initial(self, session_id: str) -> None:
        """Dispose the never-started initial session (expected product
        semantics, pending lead ruling): the continuous loop must not
        keep two active sessions or append_frame refuses everything."""
        self.recorder.abort(session_id, reason="never_started")
        self.retired_initial.append(session_id)

    def __call__(self) -> tuple[DesktopSession, Any, str]:
        self.counter += 1
        session_id = f"g3a-round-{self.counter}"
        attempt = _attempt(f"attempt-g3a-{self.counter}")
        consent = _consent(session_id)
        self.recorder.begin_attempt(self.manifest, attempt, consent)
        self.recorder.begin(session_id, consent)
        self.attempt_ids.append(attempt.attempt_id)
        # Mirror the production factory (PR-A change 4): the round owns
        # its staging (round session) and square-crop mapping (round
        # attempt). A round without this path stages nothing.
        from facecore.live.qt_window import crop_packet as _crop_packet

        def _round_transform(packet: FramePacket) -> FramePacket:
            cropped_packet, mapping = _crop_packet(packet)
            self.recorder.record_crop_mapping(attempt.attempt_id, mapping.to_dict())
            return cropped_packet

        def _round_sink(packet: FramePacket) -> None:
            self.recorder.append_frame(packet)

        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-g3a-test", "gen-g3a-test"),
            source=self.source,
            scorer=_matching_scorer,
            session_id=session_id,
            frame_sink=_round_sink,
            frame_transform=_round_transform,
            release_source_on_terminal=False,
            label_recorder=self.recorder,
            label_attempt_id=attempt.attempt_id,
        )
        return desktop, consent, attempt.attempt_id


class TestG3StartGatedFlow:
    def test_round_factory_wires_own_frame_path(
        self, qt_app: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        """PR-A change 4 (product code): rounds own sink + transform.

        Intercepts DesktopSession construction during a synthetic
        continuous cmd_live run (offscreen Qt, FakeCapture backend —
        no camera): every round session (id ``*-r<N>``) must be built
        with its own frame_sink and frame_transform. Before the fix
        the factory passed neither (rounds staged nothing).
        """
        import json

        import facecore.research.cli as cli_module
        from facecore.live.desktop import DesktopSession as RealDesktop

        profile = {
            "schema_version": "v1",
            "profile_version": "g3a-cli-v1",
            "timeout_ms": 5000,
            "sample_interval_ms": 200,
            "max_frames": 26,
            "queue_limit": 1,
            "required_support": 1,
            "min_support_interval_ms": 1,
            "match_threshold": 0.10,
            "review_threshold": 0.05,
            "margin_threshold": 0.01,
            "detector_version": "yunet-test",
            "quality_policy_version": "q-test-v1",
            "continuity_max_center_delta_ratio": 0.50,
        }
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile))
        frames = _face_frames(120)
        capture = FakeCapture(frames=frames)
        seen: list[dict[str, Any]] = []
        orig_init = RealDesktop.__init__

        def _spy_init(self: Any, *args: Any, **kwargs: Any) -> None:
            seen.append(dict(kwargs))
            orig_init(self, *args, **kwargs)

        monkeypatch.setattr(RealDesktop, "__init__", _spy_init)
        rc = cli_module.cmd_live(
            profile_path=profile_path,
            store=tmp_path / "store",
            key_dir=tmp_path / "keys",
            device="fake",
            session_id="sess-g3a-factory",
            record_consent=True,
            image_consent=True,
            ui="qt",
            qt_offscreen=True,
            capture_factory=lambda _dev: capture,
            continuous=True,
        )
        assert rc in (0, 4)
        rounds = [
            kwargs
            for kwargs in seen
            if isinstance(kwargs.get("session_id"), str)
            and "-r" in str(kwargs.get("session_id"))
        ]
        assert rounds, "continuous run must build at least one round session"
        for kwargs in rounds:
            assert kwargs.get("frame_sink") is not None, (
                f"round {kwargs.get('session_id')} must own a frame_sink"
            )
            assert kwargs.get("frame_transform") is not None, (
                f"round {kwargs.get('session_id')} must own a frame_transform"
            )

    def test_round_stages_own_frames_and_mapping(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """Round frame-path behavior: sink stages, transform maps.

        Uses a factory mirroring the production round wiring (own sink
        bound to the round session, own square-crop bound to the round
        attempt): after one standby-triggered round reaches terminal,
        the round's session must hold staged frames and the mapping
        must persist under the round's attempt id.
        """
        source = FakeCapture(_face_frames())
        factory = _RoundFactory(tmp_path, source)
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
        window.enter_standby()
        factory.retire_initial(first_desktop.session_id)
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        round_session_id = window.desktop.session_id
        round_attempt_id = window.attempt_id
        state = factory.recorder._active.get(round_session_id)
        assert state is not None, "round session must be active for staging"
        assert state.frame_count > 0, "round must stage its own frames"
        mapping = factory.recorder.read_crop_mapping(round_attempt_id)
        assert mapping["frame_w"] == 16
        assert mapping["frame_h"] == 16
        window.close()

    def test_round_clock_anchors_at_round_open(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """PR-A change 2: the 5 s window starts at this round's open.

        The window records each round's start (taken after the round's
        source open) as ``round_start_ns``. The second round's anchor
        must be later than the first round's: rounds must not share one
        pre-anchored clock value (the pre-R1 bug anchored every round
        at a single ``start_ns`` taken before the factory existed).
        """
        source = FakeCapture(_face_frames(120))
        factory = _RoundFactory(tmp_path, source)
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
        window.enter_standby()
        factory.retire_initial(first_desktop.session_id)
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        first_anchor = window.round_start_ns
        assert first_anchor is not None
        first_round_session = window.desktop.session_id
        window.press_correct()
        assert window.mode == "standby"
        # Simulate the commit-on-label release (no csv target here, so
        # the queue path keeps the session active; the product commit
        # deletes it from _active via recorder.commit).
        factory.recorder.abort(first_round_session, reason="test_released")
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        second_anchor = window.round_start_ns
        assert second_anchor is not None
        assert second_anchor > first_anchor, (
            "round 2 clock must re-anchor at its own open, "
            f"got {second_anchor} <= {first_anchor}"
        )
        window.close()

    def test_low_quality_round_does_not_finish_before_deadline(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """PR-A change 3: bounded steps must not finish before 5 s.

        With only unusable frames, a 50-step drive must leave the round
        running (no terminal), so the UI can keep ticking until the real
        deadline. A step-exhausted early finish (invalid_input with
        zero_usable_frames_collected) fails this test.
        """
        source = FakeCapture(_face_frames())
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-g3a-test", "gen-g3a-test"),
            source=source,
            scorer=_low_quality_scorer,
            session_id="g3a-slow-round",
            release_source_on_terminal=False,
        )
        desktop.on_start(_consent("g3a-slow-round"), now_ns=0, device_id="fake")
        terminal = desktop.run_until_terminal(max_steps=50, finish_on_exhaust=False)
        assert terminal is None, (
            "50 bounded steps must not finish a round whose 5 s window has not elapsed"
        )
        assert desktop.state == "running"
        desktop.close()


class _GatedCountingSource(FakeCapture):
    """FakeCapture counting open/read calls (camera-open evidence)."""

    def __init__(self, frames: list[FramePacket]) -> None:
        super().__init__(frames=frames)
        self.open_calls = 0
        self.read_calls = 0
        self.close_calls = 0

    def open(self, device_id: str) -> None:
        self.open_calls += 1
        super().open(device_id)

    def read(self) -> FramePacket | None:
        self.read_calls += 1
        return super().read()

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _gated_factory(tmp_path: Any, source: FakeCapture) -> Any:
    """Round factory mirroring the production continuous factory."""
    recorder = _recorder(tmp_path)
    manifest = _manifest()
    counter = 0

    def make() -> tuple[DesktopSession, Any, str]:
        nonlocal counter
        counter += 1
        session_id = f"g3b-round-{counter}"
        attempt = _attempt(f"attempt-g3b-{counter}")
        consent = _consent(session_id)
        recorder.begin_attempt(manifest, attempt, consent)
        recorder.begin(session_id, consent)
        desktop = DesktopSession(
            engine=SessionEngine(_profile(), "gallery-g3a-test", "gen-g3a-test"),
            source=source,
            scorer=_matching_scorer,
            session_id=session_id,
            release_source_on_terminal=False,
            label_recorder=recorder,
            label_attempt_id=attempt.attempt_id,
        )
        return desktop, consent, attempt.attempt_id

    make.recorder = recorder  # type: ignore[attr-defined]
    return make


def _gated_window(
    qt_app: Any, tmp_path: Any, source: FakeCapture, **kwargs: Any
) -> tuple[QtResearchWindow, Any]:
    factory = _gated_factory(tmp_path, source)
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
        camera_options=[(0, "Inner Cam"), (1, "相機 1")],
        **kwargs,
    )
    window.show()
    return window, factory


class TestStartGatedLifecycle:
    """PR-B RED: Start-gated Ready→Running→Result→Ready (R1 spec §2).

    Each case asserts the R1 contract against the pre-PR-B standby
    behavior (pick opens, face auto-starts, result keeps the camera
    and the photo, keys return to auto-preview standby).
    """

    def test_pick_opens_nothing(self, qt_app: Any, tmp_path: Any) -> None:
        """R1 §2-1/2: selecting a camera opens no stream, reads no frame."""
        source = _GatedCountingSource(_face_frames())
        window, _factory = _gated_window(qt_app, tmp_path, source)
        window.enter_ready()
        assert window.mode == "ready"
        window.camera_combo.setCurrentIndex(1)
        assert window.selected_camera_index() == 0
        assert source.open_calls == 0, "pick must not open the camera"
        assert source.read_calls == 0, "pick must not read any frame"
        assert source.is_closed is True
        window.close()

    def test_start_opens_selected_only_and_previews(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """R1 §2-3: Start opens the picked camera; preview advances."""
        source = _GatedCountingSource(_face_frames(120))
        window, _factory = _gated_window(qt_app, tmp_path, source)
        window.enter_ready()
        window.camera_combo.setCurrentIndex(2)
        assert window.selected_camera_index() == 1
        opens_before = source.open_calls
        window.start_clicked()
        assert window.mode == "running"
        assert source.open_calls == opens_before + 1
        first = window.preview_image
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        assert source.read_calls > 0
        window.close()

    def test_result_releases_and_clears_photo(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """R1 §2-4: terminal releases first, then clears the photo."""
        source = _GatedCountingSource(_face_frames(120))
        window, _factory = _gated_window(qt_app, tmp_path, source)
        window.enter_ready()
        window.camera_combo.setCurrentIndex(1)
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        assert source.is_closed is True, "result must release the camera"
        assert window.preview_image is None, "result must clear the photo"
        assert window.result_text.startswith("person-synth-01")
        window.close()

    def test_labeled_round_returns_to_ready_with_pick_kept(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """R1 §2-5: after a key the pick stays but the lens stays shut."""
        source = _GatedCountingSource(_face_frames(240))
        window, factory = _gated_window(qt_app, tmp_path, source)
        window.enter_ready()
        window.camera_combo.setCurrentIndex(1)
        window.start_clicked()
        window.process_until_terminal(max_steps=200)
        assert window.mode == "result"
        factory.recorder.abort(window.desktop.session_id, reason="test_released")
        window.press_correct()
        assert window.mode == "ready"
        assert window.selected_camera_index() == 0, "pick must be retained"
        assert source.is_closed is True, "lens must stay shut after labeling"
        # The next round needs another Start: no auto preview, no auto run.
        reads_after_label = source.read_calls
        window.process_until_terminal(max_steps=20)
        assert window.mode == "ready"
        assert source.read_calls == reads_after_label
        window.start_clicked()
        assert window.mode == "running"
        window.close()

    def test_three_failure_kinds_are_diagnosable(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """R1 §2-4: no-frame / no-face / quality-reject text differs."""
        source = _GatedCountingSource(_face_frames())
        window, _factory = _gated_window(qt_app, tmp_path, source)
        no_frame = window.format_result_for_test("no_frame")
        no_face = window.format_result_for_test("no_face")
        rejected = window.format_result_for_test("quality_rejected")
        assert no_frame != no_face != rejected
        for text in (no_frame, no_face, rejected):
            assert "zero_usable_frames_collected" not in text
        assert "不在" not in no_frame and "不在" not in no_face
        window.close()

    def test_cancel_closes_lens_and_returns_to_ready(
        self, qt_app: Any, tmp_path: Any
    ) -> None:
        """R1 §2-6: Cancel stops the round, closes the lens, clears."""
        source = _GatedCountingSource(_face_frames(120))
        window, _factory = _gated_window(qt_app, tmp_path, source)
        window.enter_ready()
        window.camera_combo.setCurrentIndex(1)
        window.start_clicked()
        assert window.mode == "running"
        window.cancel_clicked()
        assert window.mode == "ready"
        assert source.is_closed is True
        assert window.preview_image is None
        window.close()
