"""Phase 2B end-to-end enablement acceptance (E8-A, synthetic only).

Chain under test (plan §12 E8 acceptance, verbatim):
real CLI / Qt Start → fake camera → real score adapter (test doubles) →
real engine → AEAD store → label → dual arms → analysis → freeze →
future synthetic visit → holdout → delete / restart-unreadable.

No real camera is ever opened (this file never touches
``scripts/live_preflight.py`` or ``live_teardown.py --device``), no real
faces exist in the repo, and every paragraph below runs camera-free.

Two injection points cover the same §9.1 contract at different depths:

- ``tests/research/test_paired_analysis.py:164``
  ``test_six_row_reconciliation_exact`` fixes the arithmetic with directly
  constructed ``AttemptRecord`` / ``ArmOutcome`` / ``EvaluationLabel``.
  That test is kept unchanged.
- THIS file drives the front half of the chain for real: ``cmd_live``
  over a fake camera and test-double detector/embedder, through the real
  ``score_frame`` adapter and the real ``SessionEngine``, so that the six
  rows of outcomes are produced by the production wiring — never injected
  pre-built. A wiring fault anywhere in the front half moves the endpoint
  numbers, which is exactly the value this file adds.

§9.1 frozen rows (truth → A-terminal / B-terminal):

- s1 enrolled p1 → matched p1 / timeout  (T06: A correct, B unsuccessful)
- s2 enrolled p1 → matched p2 / matched p2 (T01: wrong enrolled)
- s3 enrolled p1 → invalid_input / invalid_input (all frames rejected)
- s4 enrolled p1 → no inference result (camera open error after Start)
- s5 unenrolled → review / timeout
- s6 unenrolled → matched p2 / matched p2 (T01: unenrolled false accept)

Expected endpoints: attempted=6, truth-known enrolled=4, unknown=2,
paired-complete=5; A correct=1/4, B correct=0/4; both arms wrong
enrolled=1/4, unknown FA=1/2.

Per-arm shaping (same gallery person-01/person-02 throughout; only the
live embedding vector and the detector script change):

- s1: only two non-adjacent frames carry a face (strong p1). Support can
  never reach ``required_support=3``, so the time-consistency arm times
  out while the best-quality arm matches p1.
- s2/s6: every frame strongly matches p2 → both arms match p2.
- s3: the detector goes blind after the gallery build → every frame is
  quality-rejected, but the full fixed window is still collected, so the
  attempt stays in the paired set with invalid_input on both arms and is
  NOT silently dropped (T08 fires as well).
- s4: the capture source refuses ``open()`` after the attempt ledger
  already accepted the Start → operational open_error, no bundle, no
  trace, ``no_result`` on both arms, still counted in the denominator.
- s5: every frame sits in the review band (top score in
  [review_threshold, match_threshold)) → A review, B timeout.

The window provenance in every complete row comes from the real
collector (``deadline_reached`` / ``complete`` with the closing frame
strictly past the deadline — see
``tests/research/test_paired_window_boundary.py`` for why that property
is load-bearing), never from a hand-built ``CollectionWindow``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from facecore.live.capture import CaptureSource
from facecore.live.contracts import FramePacket
from facecore.research.analysis import analyze_batch
from facecore.research.cli import _load_profile, cmd_analyze, cmd_live, cmd_replay
from facecore.research.experiment import EvaluationLabel
from facecore.research.recorder import ResearchRecorder
from facecore.research.replay import evaluate_arms
from facecore.research.split import authorize_holdout, freeze_candidate

EXPERIMENT_ID = "exp-phase2b-e2e"

PROFILE_VERSION = "phase2b-e2e-v1"

# Orthogonal unit gallery vectors; cosine scores are pure arithmetic.
_GALLERY_P1 = (1.0, 0.0, 0.0)
_GALLERY_P2 = (0.0, 1.0, 0.0)
_STRONG_P1 = (0.97, 0.20, 0.10)  # top ~0.97, margin ~0.77
_STRONG_P2 = (0.20, 0.97, 0.10)  # top ~0.97, margin ~0.77
_REVIEW_BAND = (0.40, 0.35, 0.80)  # top ~0.42 in [0.30, 0.45)


def _clock() -> datetime:
    return datetime.now(timezone.utc)


def _profile_path(tmp_path: Path) -> Path:
    profile = {
        "schema_version": "v1",
        "profile_version": PROFILE_VERSION,
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

    # Checkerboard texture: real sharpness energy so the quality gate
    # accepts these frames (a flat field reads as blurry, sharpness 0.0).
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


def _face() -> Any:
    from facecore.pipeline.detect import DetectedFace

    landmarks = (
        (50.0, 60.0),
        (110.0, 60.0),
        (80.0, 90.0),
        (60.0, 115.0),
        (100.0, 115.0),
    )
    return DetectedFace(
        box=(20.0, 20.0, 120.0, 120.0), landmarks=landmarks, confidence=0.99
    )


def _scripted_detector(
    face_live_seqs: set[int] | None = None,
    *,
    blind_after_gallery: bool = False,
) -> MagicMock:
    """Test-double detector with a programmed live behaviour.

    The gallery build always consumes the first two ``detect`` calls, so
    live frame ``n`` is call ``n + 2``. ``face_live_seqs`` limits visible
    faces to those live sequences (``None`` = every frame has a face);
    ``blind_after_gallery`` returns no faces at all once the gallery is
    built (the s3 all-frames-rejected shape).
    """
    face = _face()
    detector = MagicMock()
    calls = 0

    def _detect(image: Any) -> list[Any]:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return [face]
        if blind_after_gallery:
            return []
        if face_live_seqs is not None and (calls - 2) not in face_live_seqs:
            return []
        return [face]

    detector.detect.side_effect = _detect
    return detector


class _ScriptedEmbedder:
    """Gallery build consumes the first two embeds; live frames are scripted.

    ``live_vec`` is either one vector for every live frame or a callable
    mapping the 1-based live-frame index to a vector.
    """

    model_version = "sface_2021dec"

    def __init__(
        self, live_vec: tuple[float, float, float] | Callable[[int], tuple[float, float, float]]
    ) -> None:
        self._calls = 0
        self._live_vec = live_vec

    def embed(self, crop: Any) -> tuple[np.ndarray, str]:
        self._calls += 1
        if self._calls == 1:
            vec = _GALLERY_P1
        elif self._calls == 2:
            vec = _GALLERY_P2
        elif callable(self._live_vec):
            vec = self._live_vec(self._calls - 2)
        else:
            vec = self._live_vec
        arr = np.array(vec, dtype=np.float32)
        return arr / float(np.linalg.norm(arr)), "sface_2021dec"


class _SteppedCamera(CaptureSource):
    """Fake camera pacing real wall-clock time at 200ms per frame.

    ``read()`` blocks until the frame's capture instant, so stamps are
    wall-true and ``score_frame``'s ``processed_ns`` monotonicity holds on
    every path — including the quality-rejection path that stamps
    ``time.monotonic_ns()`` directly (a fast-forward stepped camera puts
    ``captured_ns`` in the future and trips that check with
    ``scorer_failure: ValueError``).

    The anchor is taken at ``open()``, i.e. microseconds AFTER the CLI's
    session start, the way a real device trails session start. The closing
    frame therefore lands marginally beyond the deadline — the real
    collector's signature (see ``test_paired_window_boundary.py``). Each E8
    attempt gets its own camera instance: the anchor must be per-session,
    never shared.

    Cost: ~5s wall time per full-window session. That is the price of
    wall-true stamps; fast-forwarding would fake the very provenance the
    paired assertions depend on.
    """

    def __init__(self, count: int = 30, *, paced: bool = True) -> None:
        self._lock = threading.Lock()
        self._count = count
        self._paced = paced
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
            target_ns = self._anchor + self._seq * 200_000_000
            paced = self._paced
        if paced:
            now_ns = time.monotonic_ns()
            if target_ns > now_ns:
                time.sleep((target_ns - now_ns) / 1_000_000_000)
        rgb = _live_frame_rgb(self._seq)
        return FramePacket(
            sequence=self._seq,
            captured_ns=target_ns,
            rgb=rgb,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._opened = False

    def is_closed(self) -> bool:
        with self._lock:
            return self._closed


class _RefusingCamera(_SteppedCamera):
    """The s4 shape: the device refuses ``open()`` after Start was accepted."""

    def open(self, device_id: str) -> None:
        raise RuntimeError("synthetic camera open refused (s4 probe)")


def _live_frame_rgb(seq: int) -> np.ndarray:
    rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
    rgb[::2, ::2] = 160
    rgb[1::2, 1::2] = 80
    rgb[0, 0, 0] = seq % 256
    return rgb


class _Chain:
    """One shared synthetic world for the six §9.1 rows.

    A single store / key dir / profile / corpus / models directory hosts
    all six sessions, so the analysis denominators are computed over the
    real multi-attempt ledger — not over six isolated fixtures.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.profile_path = _profile_path(tmp_path)
        self.profile = _load_profile(self.profile_path)
        self.corpus = _corpus_manifest(tmp_path)
        self.models = tmp_path / "models"
        self.models.mkdir(exist_ok=True)
        self.store = tmp_path / "store"
        self.key_dir = tmp_path / "research_keys"

    def run_live(
        self,
        session: str,
        *,
        live_vec: tuple[float, float, float] | Callable[[int], tuple[float, float, float]] = _STRONG_P2,
        face_live_seqs: set[int] | None = None,
        blind_after_gallery: bool = False,
        camera: CaptureSource | None = None,
    ) -> int:
        return cmd_live(
            profile_path=self.profile_path,
            store=self.store,
            key_dir=self.key_dir,
            device="0",
            session_id=session,
            record_consent=True,
            image_consent=True,
            fixed_seconds=True,
            models=self.models,
            corpus=self.corpus,
            capture_factory=lambda _device: camera or _SteppedCamera(),
            detector_factory=lambda _models: _scripted_detector(
                face_live_seqs, blind_after_gallery=blind_after_gallery
            ),
            embedder_factory=lambda _models: _ScriptedEmbedder(live_vec),
            experiment_id=EXPERIMENT_ID,
            attempt_id=f"att-{session}",
        )

    def recorder(self) -> ResearchRecorder:
        return ResearchRecorder(
            store_root=self.store, key_dir=self.key_dir, clock=_clock
        )

    def label(
        self,
        recorder: ResearchRecorder,
        session: str,
        kind: str,
        identity_id: str | None = None,
    ) -> None:
        recorder.write_label(
            EvaluationLabel(
                attempt_id=f"att-{session}",
                revision=1,
                kind=kind,  # type: ignore[arg-type]
                identity_id=identity_id,
                actor_ref="phase2b-e2e-evaluator",
                labeled_at=_clock().isoformat(),
            )
        )

    def arms(self, recorder: ResearchRecorder, session: str) -> tuple[Any, Any]:
        attempt_id = f"att-{session}"
        window = recorder.read_record(session).collection_window
        trace = recorder.read_trace(attempt_id)
        return evaluate_arms(trace, self.profile, window=window)


def _run_six_rows(chain: _Chain) -> dict[str, int]:
    """Drive s1–s6 through the real chain; return each live exit code."""
    rc: dict[str, int] = {}
    # s1: two non-adjacent faces (strong p1); support never reaches 3.
    rc["s1"] = chain.run_live(
        "sess-e2e-s1", live_vec=_STRONG_P1, face_live_seqs={5, 18}
    )
    # s2: every frame strongly p2.
    rc["s2"] = chain.run_live("sess-e2e-s2", live_vec=_STRONG_P2)
    # s3: detector blind after the gallery build; full window collected.
    rc["s3"] = chain.run_live(
        "sess-e2e-s3", live_vec=_STRONG_P1, blind_after_gallery=True
    )
    # s4: camera open error after the Start was accepted into the ledger.
    rc["s4"] = chain.run_live(
        "sess-e2e-s4", camera=_RefusingCamera()
    )
    # s5: every frame in the review band.
    rc["s5"] = chain.run_live("sess-e2e-s5", live_vec=_REVIEW_BAND)
    # s6: every frame strongly p2 (unenrolled truth → false accept).
    rc["s6"] = chain.run_live("sess-e2e-s6", live_vec=_STRONG_P2)
    return rc


def _label_six_rows(chain: _Chain, recorder: ResearchRecorder) -> None:
    chain.label(recorder, "sess-e2e-s1", "enrolled", "person-01")
    chain.label(recorder, "sess-e2e-s2", "enrolled", "person-01")
    chain.label(recorder, "sess-e2e-s3", "enrolled", "person-01")
    chain.label(recorder, "sess-e2e-s4", "enrolled", "person-01")
    chain.label(recorder, "sess-e2e-s5", "unenrolled")
    chain.label(recorder, "sess-e2e-s6", "unenrolled")


def _collect_outcomes(chain: _Chain, recorder: ResearchRecorder) -> list[Any]:
    outcomes: list[Any] = []
    for session in (
        "sess-e2e-s1",
        "sess-e2e-s2",
        "sess-e2e-s3",
        "sess-e2e-s4",
        "sess-e2e-s5",
        "sess-e2e-s6",
    ):
        attempt_id = f"att-{session}"
        trace_dir = recorder._trace_dir(attempt_id)
        if trace_dir.is_dir() and any(trace_dir.glob("frame_*.enc")):
            arm_a, arm_b = chain.arms(recorder, session)
            outcomes.extend([arm_a, arm_b])
    return outcomes


class TestPhase2BSixRowEndToEnd:
    """§9.1, produced by the production wiring — never injected pre-built."""

    def test_six_rows_reconcile_exact(self, tmp_path: Path) -> None:
        chain = _Chain(tmp_path)
        rc = _run_six_rows(chain)

        # Live outcomes: five full fixed windows; s4 fails at open.
        for session in ("s1", "s2", "s3", "s5", "s6"):
            assert rc[session] == 0, f"live {session} failed with rc={rc[session]}"
        assert rc["s4"] == 2, "s4 must refuse the camera open fail-closed"

        recorder = chain.recorder()
        _label_six_rows(chain, recorder)
        outcomes = _collect_outcomes(chain, recorder)
        attempts = recorder.list_attempts(EXPERIMENT_ID)
        labels = [
            recorder.read_label(a.attempt_id)
            for a in attempts
            if recorder._label_dir(a.attempt_id).is_dir()
        ]

        # Provenance guard: every complete row's closing frame is strictly
        # past the deadline — the real collector's signature. If this ever
        # stops holding, the paired assertions below lose their meaning.
        for session in (
            "sess-e2e-s1",
            "sess-e2e-s2",
            "sess-e2e-s3",
            "sess-e2e-s5",
            "sess-e2e-s6",
        ):
            window = recorder.read_record(session).collection_window
            trace = recorder.read_trace(f"att-{session}")
            assert window.collection_complete is True
            assert window.collection_stop_reason == "deadline_reached"
            last_ns = max(e.captured_ns for e in trace.entries)
            assert last_ns > window.collection_deadline_ns
            assert len(trace.entries) == window.frames_sampled

        report = analyze_batch(attempts, outcomes, labels, profile=chain.profile)

        # §9.1 frozen denominators.
        assert report.attempted == 6
        assert report.truth_known_enrolled == 4
        assert report.unknown == 2
        assert report.paired_complete == 5
        assert report.operation_errors == 1

        # Per-arm endpoints, produced by the real engine — not constructed.
        by_arm = {o.attempt_id: o for o in outcomes if o.arm_id == "A"}
        by_arm_b = {o.attempt_id: o for o in outcomes if o.arm_id == "B"}
        assert (by_arm["att-sess-e2e-s1"].terminal,
                by_arm["att-sess-e2e-s1"].matched_identity) == ("matched", "person-01")
        assert by_arm_b["att-sess-e2e-s1"].terminal == "timeout"
        assert (by_arm["att-sess-e2e-s2"].terminal,
                by_arm["att-sess-e2e-s2"].matched_identity) == ("matched", "person-02")
        assert (by_arm_b["att-sess-e2e-s2"].terminal,
                by_arm_b["att-sess-e2e-s2"].matched_identity) == ("matched", "person-02")
        assert by_arm["att-sess-e2e-s3"].terminal == "invalid_input"
        assert by_arm_b["att-sess-e2e-s3"].terminal == "invalid_input"
        assert by_arm["att-sess-e2e-s5"].terminal == "review"
        assert by_arm_b["att-sess-e2e-s5"].terminal == "timeout"
        assert (by_arm["att-sess-e2e-s6"].terminal,
                by_arm["att-sess-e2e-s6"].matched_identity) == ("matched", "person-02")
        # s4 has no inference result on either arm.
        assert "att-sess-e2e-s4" not in by_arm
        assert "att-sess-e2e-s4" not in by_arm_b

        assert report.arm_a.correct == 1
        assert report.arm_a.enrolled_correct_rate == pytest.approx(0.25)  # 1/4
        assert report.arm_b.correct == 0
        assert report.arm_b.enrolled_correct_rate == pytest.approx(0.0)  # 0/4
        assert report.arm_a.wrong_enrolled == 1
        assert report.arm_b.wrong_enrolled == 1
        assert report.arm_a.unknown_false_accept == 1
        assert report.arm_b.unknown_false_accept == 1

        # s3 must not vanish for having no usable frame; s4 stays in the
        # denominator as an operation error.
        assert report.arm_a.invalid_input == 1
        assert report.arm_a.no_result == 1
        assert report.operation_errors == 1

        # Trigger wiring: s1 → T06, s2/s6 → T01. B never borrows A's s1
        # diagnosis to inflate its own success.
        assert "att-sess-e2e-s1" in report.triggers.get("T06", ())
        assert "att-sess-e2e-s2" in report.triggers.get("T01", ())
        assert "att-sess-e2e-s6" in report.triggers.get("T01", ())
        assert "att-sess-e2e-s1" not in report.triggers.get("T01", ())

        # Same participant across s1–s4 stays four attempts, one
        # participant, one visit group — replay never mints new humans.
        assert report.participants_count == 1
        assert report.visits_count == 1

    def test_refused_replay_run_is_attempt_scoped(self, tmp_path: Path) -> None:
        """§3.2 (commander ruling 1): refusal isolation is attempt-level.

        Appending one refused replay run for s2 keeps ``attempted`` at 6,
        moves the whole attempt out of the paired set (5 → 4, consistent
        with the spec §5 paired-replay eligible definition: no missing /
        tampered members), fires T03, and leaves the original live
        outcomes untouched.
        """
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        _label_six_rows(chain, recorder)
        outcomes = _collect_outcomes(chain, recorder)
        attempts = recorder.list_attempts(EXPERIMENT_ID)
        labels = [
            recorder.read_label(a.attempt_id)
            for a in attempts
            if recorder._label_dir(a.attempt_id).is_dir()
        ]
        live_s2 = [o for o in outcomes if o.attempt_id == "att-sess-e2e-s2"]

        refused_a = replace(
            live_s2[0], run_id="run-att-sess-e2e-s2-refused", refusal="staging_incomplete"
        )
        refused_b = replace(
            live_s2[1], run_id="run-att-sess-e2e-s2-refused", refusal="staging_incomplete"
        )

        report = analyze_batch(
            attempts, outcomes + [refused_a, refused_b], labels,
            profile=chain.profile,
        )
        assert report.attempted == 6
        assert report.paired_complete == 4
        assert "att-sess-e2e-s2" in report.triggers.get("T03", ())
        # The original live results are not deleted by the extra run.
        assert all(o in outcomes for o in live_s2)

    def test_replay_does_not_mint_new_human_sessions(self, tmp_path: Path) -> None:
        """plan :271 — replaying s2 never adds attempts to the ledger."""
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        before = len(recorder.list_attempts(EXPERIMENT_ID))
        assert before == 6
        for _ in range(3):
            rc = cmd_replay(
                store=chain.store,
                key_dir=chain.key_dir,
                session_id="sess-e2e-s2",
                profile_path=chain.profile_path,
                models=chain.models,
                corpus=chain.corpus,
                detector_factory=lambda _models: _scripted_detector(),
                embedder_factory=lambda _models: _ScriptedEmbedder(_STRONG_P2),
            )
            assert rc == 0
        assert len(recorder.list_attempts(EXPERIMENT_ID)) == 6


class TestPhase2BChainRemainder:
    """Late / reset / missing-blob / failure / contamination / expiry /
    early-match-cancel — each detector gets a synthetic positive control."""

    def test_b_timeout_is_the_late_path(self, tmp_path: Path) -> None:
        """``late``: observations past the effective deadline end the B
        replay via ``deadline_exceeded`` → timeout (s1's B arm)."""
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        _label_six_rows(chain, recorder)
        arm_a, arm_b = chain.arms(recorder, "sess-e2e-s1")
        assert arm_b.terminal == "timeout"
        assert arm_b.decision_codes[0] == "b_timeout"
        # ...while the same window's best frame still matched on A.
        assert arm_a.terminal == "matched"
        assert arm_a.matched_identity == "person-01"

    def test_support_reset_blocks_b_lock(self, tmp_path: Path) -> None:
        """``reset``: faceless frames clear the support window, so B can
        never accumulate the three consecutive supports it needs."""
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        trace = recorder.read_trace("att-sess-e2e-s1")
        faceless = [e for e in trace.entries if e.face_count == 0]
        assert len(faceless) == 23, "only seq 5 and 18 carry a face"
        _label_six_rows(chain, recorder)
        _arm_a, arm_b = chain.arms(recorder, "sess-e2e-s1")
        assert arm_b.terminal == "timeout"
        assert arm_b.support_sequences == ()

    def test_missing_blob_refuses_and_unpairs(self, tmp_path: Path) -> None:
        """``missing blob``: a trace entry whose staged blob is gone
        refuses both arms and drops the attempt from the paired set."""
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        _label_six_rows(chain, recorder)
        window = recorder.read_record("sess-e2e-s2").collection_window
        trace = recorder.read_trace("att-sess-e2e-s2")
        entries = list(trace.entries)
        entries[7] = replace(entries[7], stage_missing_reason="blob_lost")
        gapped = replace(trace, entries=tuple(entries))
        arm_a, arm_b = evaluate_arms(gapped, chain.profile, window=window)
        assert arm_a.refusal == "staging_incomplete"
        assert arm_b.refusal == "staging_incomplete"
        assert arm_a.terminal != "matched" and arm_b.terminal != "matched"

        outcomes = _collect_outcomes(chain, recorder)
        outcomes = [o for o in outcomes if o.attempt_id != "att-sess-e2e-s2"]
        outcomes.extend([arm_a, arm_b])
        attempts = recorder.list_attempts(EXPERIMENT_ID)
        labels = [
            recorder.read_label(a.attempt_id)
            for a in attempts
            if recorder._label_dir(a.attempt_id).is_dir()
        ]
        report = analyze_batch(attempts, outcomes, labels, profile=chain.profile)
        assert report.attempted == 6
        assert report.paired_complete == 4

    def test_failed_attempt_stays_in_denominator(self, tmp_path: Path) -> None:
        """``failed attempt``: s4's open error is an operation error AND a
        denominator member — never silently dropped."""
        chain = _Chain(tmp_path)
        rc = _run_six_rows(chain)
        assert rc["s4"] == 2
        recorder = chain.recorder()
        _label_six_rows(chain, recorder)
        outcomes = _collect_outcomes(chain, recorder)
        attempts = recorder.list_attempts(EXPERIMENT_ID)
        labels = [
            recorder.read_label(a.attempt_id)
            for a in attempts
            if recorder._label_dir(a.attempt_id).is_dir()
        ]
        report = analyze_batch(attempts, outcomes, labels, profile=chain.profile)
        assert report.attempted == 6
        assert report.operation_errors == 1
        assert report.arm_a.no_result == 1
        assert report.arm_b.no_result == 1

    def test_holdout_contamination_is_refused(self, tmp_path: Path) -> None:
        """``holdout contamination``: development analysis refuses a batch
        containing a sealed-holdout attempt; holdout analysis refuses
        before the authorized release."""
        from facecore.research.cli import _default_experiment_manifest
        from facecore.research.experiment import AttemptRecord
        from facecore.research.records import ConsentRecord

        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        _label_six_rows(chain, recorder)

        manifest = _default_experiment_manifest(EXPERIMENT_ID)
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest=chain.profile.profile_digest(),
            analysis_digest="a" * 64,
            planned_visit_ids=("visit-future-001",),
        )
        recorder.record_freeze(freeze)

        now = _clock().isoformat()
        holdout_attempt = AttemptRecord(
            experiment_id=EXPERIMENT_ID,
            attempt_id="att-holdout-future-1",
            participant_id="cli-operator",
            visit_id="visit-future-001",
            condition_id="cond-cli-live",
            attempt_index=1,
            retry_of=None,
            consent_ref="sess-holdout-future",
            requested_at_utc=now,
            accepted_at_utc=now,
            started_at_utc=None,
            ended_at_utc=None,
            operational_status="accepted",
            error_code=None,
            bundle_ref=None,
            split="holdout",
        )
        consent = ConsentRecord(
            session_id="sess-holdout-future",
            participant_id="cli-operator",
            record_consent=True,
            image_consent=True,
            consented_at_utc=now,
            record_expires_at_utc=(_clock() + timedelta(days=30)).isoformat(),
            image_expires_at_utc=(_clock() + timedelta(days=7)).isoformat(),
        )
        recorder.begin_attempt(manifest, holdout_attempt, consent)

        attempts = recorder.list_attempts(EXPERIMENT_ID)
        labels = [
            recorder.read_label(a.attempt_id)
            for a in attempts
            if recorder._label_dir(a.attempt_id).is_dir()
        ]
        # Development mode refuses the contaminated batch.
        with pytest.raises(Exception):
            analyze_batch(
                attempts, _collect_outcomes(chain, recorder), labels,
                profile=chain.profile, mode="development", freeze=freeze,
            )
        # Holdout mode refuses before the authorized release.
        with pytest.raises(Exception):
            analyze_batch(
                [holdout_attempt], [], [], profile=chain.profile,
                mode="holdout", freeze=freeze, release=None,
            )
        # The authorized release unseals holdout analysis.
        release = authorize_holdout(freeze, operator_decision_id="d-e2e-test")
        recorder.record_release(release, experiment_id=EXPERIMENT_ID)
        holdout_report = analyze_batch(
            [holdout_attempt], [], [], profile=chain.profile,
            mode="holdout", freeze=freeze, release=release,
        )
        assert holdout_report.attempted == 1

    def test_record_and_image_expiry(self, tmp_path: Path) -> None:
        """``record/image expiry``: image blobs purge at 7d, the record at
        30d — both through the real ``purge_expired`` path."""
        chain = _Chain(tmp_path)
        rc = chain.run_live("sess-e2e-exp", live_vec=_STRONG_P2)
        assert rc == 0
        recorder = chain.recorder()
        manifest_before = json.loads(
            (chain.store / "sess-e2e-exp" / "manifest.json").read_text()
        )
        assert manifest_before["frame_count"] > 0

        mid = _clock() + timedelta(days=8)
        purged_images = recorder.purge_expired(mid)
        assert "sess-e2e-exp" in purged_images
        manifest_mid = json.loads(
            (chain.store / "sess-e2e-exp" / "manifest.json").read_text()
        )
        assert manifest_mid["frame_count"] == 0
        # The record itself survives image expiry.
        recorder.read_record("sess-e2e-exp")

        late = _clock() + timedelta(days=31)
        purged_records = recorder.purge_expired(late)
        assert "sess-e2e-exp" in purged_records
        with pytest.raises(KeyError):
            recorder.read_record("sess-e2e-exp")

    def test_early_match_keeps_collecting_to_deadline(self, tmp_path: Path) -> None:
        """``early matched then keep collecting``: B locks within three
        frames on s2, yet the collector still samples the full 25-frame
        window to the original deadline for arm A."""
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        _label_six_rows(chain, recorder)
        _arm_a, arm_b = chain.arms(recorder, "sess-e2e-s2")
        assert arm_b.terminal == "matched"
        assert arm_b.matched_identity == "person-02"
        assert len(arm_b.support_sequences) == 3
        window = recorder.read_record("sess-e2e-s2").collection_window
        assert window.frames_sampled == 25
        assert window.collection_complete is True

    def test_qt_start_collects_with_crop_mapping(self, tmp_path: Path) -> None:
        """Qt Start (offscreen): the guided window collects through the
        square-capture path and persists the Appendix A crop mapping."""
        pytest.importorskip("PySide6.QtWidgets")
        chain = _Chain(tmp_path)
        rc = cmd_live(
            profile_path=chain.profile_path,
            store=chain.store,
            key_dir=chain.key_dir,
            device="fake",
            session_id="sess-e2e-qt",
            record_consent=True,
            image_consent=True,
            ui="qt",
            qt_offscreen=True,
            capture_factory=lambda _device: _SteppedCamera(count=4, paced=False),
            experiment_id=EXPERIMENT_ID,
            attempt_id="att-sess-e2e-qt",
        )
        assert rc == 0
        recorder = chain.recorder()
        mapping = recorder.read_crop_mapping("att-sess-e2e-qt")
        assert mapping["size"] == 16
        attempts = [
            a
            for a in recorder.list_attempts(EXPERIMENT_ID)
            if a.attempt_id == "att-sess-e2e-qt"
        ]
        assert len(attempts) == 1

    def test_cli_analyze_reports_six_rows(self, tmp_path: Path, capsys: Any) -> None:
        """The production ``analyze`` entrypoint renders the same endpoints
        over the real ledger, traces, windows, and label sidecars."""
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        _label_six_rows(chain, chain.recorder())
        rc = cmd_analyze(
            store=chain.store,
            key_dir=chain.key_dir,
            experiment_id=EXPERIMENT_ID,
            mode="development",
            profile_path=chain.profile_path,
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Attempted: 6" in out
        assert "Paired-complete: 5" in out

    def test_withdraw_and_delete_are_restart_unreadable(
        self, tmp_path: Path
    ) -> None:
        """``delete / restart-unreadable``: consent withdrawal destroys the
        attempt DEK plus sidecars; bundle delete removes the session; a
        fresh recorder over the same store reads neither."""
        chain = _Chain(tmp_path)
        _run_six_rows(chain)
        recorder = chain.recorder()
        _label_six_rows(chain, recorder)

        recorder.withdraw_attempt("att-sess-e2e-s5")
        with pytest.raises(KeyError):
            recorder.read_trace("att-sess-e2e-s5")
        with pytest.raises(KeyError):
            recorder.read_label("att-sess-e2e-s5")

        rc = cmd_live(
            profile_path=chain.profile_path,
            store=chain.store,
            key_dir=chain.key_dir,
            device="fake",
            session_id="sess-e2e-del",
            record_consent=True,
            image_consent=True,
        )
        assert rc == 0
        assert chain.recorder().delete("sess-e2e-del") is True
        assert not (chain.store / "sess-e2e-del").exists()

        fresh = ResearchRecorder(
            store_root=chain.store, key_dir=chain.key_dir, clock=_clock
        )
        with pytest.raises(KeyError):
            fresh.read_trace("att-sess-e2e-s5")
        with pytest.raises(KeyError):
            fresh.read_record("sess-e2e-del")
        intact = fresh.read_record("sess-e2e-s1")
        assert intact.result.status.value == "matched"
