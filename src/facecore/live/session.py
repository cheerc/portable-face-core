"""Bounded interactive live session engine and dual strategies (Task T3).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T3;
    - Task: t-20260914111116050440-76424-34;
    - Governing decision: d-20260914110757304910-5.

Hard boundaries:
    - Pure sequence of time-ordered FrameObservation inputs.
    - Zero camera I/O, zero file paths.
    - Strictly uncoupled from ground truth labels or participant names.
    - Monotonic clock injection; deadline is strictly enforced (start + 5.0s).
    - Continuity: default bound is 0.50; None forces fallback.
    - Exactly one immutable terminal result per start.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
import math

from facecore.live.contracts import (
    DecisionEvent,
    FrameObservation,
    ResearchProfile,
    SessionResult,
    SessionStatus,
    TimingMarks,
)

DEFAULT_CONTINUITY_MAX_CENTER_DELTA_RATIO = 0.50


def _center_and_dim(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    dim = max(w, h)
    return cx, cy, dim


class SessionEngine:
    """State machine governing live session multi-frame evidence accumulation."""

    def __init__(
        self,
        profile: ResearchProfile,
        gallery_digest: str,
        model_generation: str,
        *,
        event_sink: Callable[[DecisionEvent], None] | None = None,
    ) -> None:
        self.profile = profile
        self.gallery_digest = gallery_digest
        self.model_generation = model_generation
        self._event_sink = event_sink

        self._session_id: str | None = None
        self._start_ns: int | None = None
        self._deadline_ns: int | None = None
        # D1: the deadline above is the frozen contract (start + timeout)
        # used by replay and the fixed-window collector. `start()` arms it
        # immediately, as before. `anchor_recognition()` re-arms the SAME
        # deadline at the first valid frame, so camera-open time never
        # eats the recognition window. Callers that never re-anchor keep
        # the pre-D1 behaviour byte-for-byte.
        self._recognition_anchored: bool = False
        self._first_frame_ns: int | None = None
        self._timing_marks: TimingMarks | None = None

        # Evidence accumulation state
        self._current_candidate: str | None = None
        self._support_sequences: list[int] = []
        self._last_support_ns: int | None = None
        self._last_box: tuple[float, float, float, float] | None = None
        # D1: which rejection kinds this round actually saw, so the
        # terminal can distinguish "camera never produced a usable face"
        # from "the operator never looked at the lens".
        self._rejection_reasons: set[str] = set()

        # D7-A W3: per-round event tallies, kept regardless of whether an
        # `event_sink` is wired. `_emit_event` returns early when the sink
        # is None, so on the live path — which passes no sink — every event
        # was previously discarded at the source and the demo log could not
        # tell 「support 窗被清空」 from 「support 窗從未累積」 (D4 §11 item
        # 18b). Counting here makes the tally independent of the sink, so a
        # caller that only wants the counts needs no wiring.
        #
        # Initialised in `__init__` as well as `start()` on purpose: a
        # caller may read the tally before a round is ever started, and an
        # attribute that only exists after `start()` would turn that read
        # into an AttributeError rather than a truthful zero.
        self._event_counts: Counter[str] = Counter()

        # Diagnostic counters
        self._frames_sampled: int = 0
        self._frames_usable: int = 0
        self._frames_rejected: int = 0
        self._frames_dropped: int = 0
        self._observations: list[FrameObservation] = []

        # Sequence and time tracking
        self._last_sequence: int = 0
        self._last_observation_ns: int = 0

        # Terminal state
        self._terminal_result: SessionResult | None = None

    def _emit_event(
        self,
        *,
        sequence: int,
        event_type: str,
        accepted: bool,
        reset_reason: str | None,
        support_before: int,
        support_after: int,
        candidate_before: str | None,
        candidate_after: str | None,
        terminal_status: str | None,
        terminal_identity: str | None,
        now_ns: int,
    ) -> None:
        # D7-A W3: tally before the sink guard, which is an early return.
        # `event_counts()` is therefore a sink-independent read: a caller
        # that only wants the tally does not have to become an
        # `event_sink` and start retaining every DecisionEvent.
        self._event_counts[event_type] += 1
        if self._event_sink is None:
            return
        remaining_ms = (
            max(0.0, (self._deadline_ns - now_ns) / 1_000_000.0)
            if self._deadline_ns is not None
            else 0.0
        )
        event = DecisionEvent(
            sequence=sequence,
            event_type=event_type,
            accepted=accepted,
            reset_reason=reset_reason,
            support_before=support_before,
            support_after=support_after,
            candidate_before=candidate_before,
            candidate_after=candidate_after,
            terminal_status=terminal_status,
            terminal_identity=terminal_identity,
            deadline_remaining_ms=remaining_ms,
        )
        self._event_sink(event)

    @property
    def is_terminal(self) -> bool:
        return self._terminal_result is not None

    def start(self, session_id: str, now_ns: int) -> None:
        """Start a new bounded session with monotonically injected clock."""
        if not session_id:
            raise ValueError("session_id must not be empty")
        if now_ns < 0:
            raise ValueError(f"now_ns must be >= 0, got {now_ns}")

        self._session_id = session_id
        self._start_ns = now_ns
        self._deadline_ns = now_ns + int(self.profile.timeout_ms * 1_000_000)

        self._current_candidate = None
        self._support_sequences = []
        self._last_support_ns = None
        self._last_box = None

        self._frames_sampled = 0
        self._frames_usable = 0
        self._frames_rejected = 0
        self._frames_dropped = 0
        self._observations = []
        self._rejection_reasons = set()

        self._last_sequence = 0
        self._last_observation_ns = now_ns
        self._terminal_result = None
        self._recognition_anchored = False
        self._first_frame_ns = None
        self._timing_marks = None
        # D7-A W3: per-round event tally, cleared here with the rest of the
        # per-round state. A reused engine must report this round's events,
        # never the previous round's. No re-annotation: the attribute is
        # declared in `__init__` and mypy (strict) rejects a second one.
        self._event_counts = Counter()

    @property
    def deadline_ns(self) -> int | None:
        """The armed recognition deadline (session-clock domain)."""
        return self._deadline_ns

    @property
    def recognition_anchored(self) -> bool:
        """True once the window has been re-armed at a first valid frame."""
        return self._recognition_anchored

    def event_count(self, event_type: str) -> int:
        """How many times `event_type` fired in this round (D7-A W3).

        Read-only and sink-independent: a round that emitted nothing
        returns 0, which is a real measurement, not a missing value. The
        caller distinguishes that from "field absent" by the column always
        being present in the demo CSV.
        """
        return self._event_counts[event_type]

    def event_counts(self) -> Mapping[str, int]:
        """A copy of this round's per-event-type tallies (D7-A W3)."""
        return dict(self._event_counts)

    def note_timing(
        self,
        *,
        open_begin_ns: int | None = None,
        open_end_ns: int | None = None,
    ) -> None:
        """Record the D1 open segment; survives until terminal."""
        previous = self._timing_marks
        self._timing_marks = TimingMarks(
            open_begin_ns=open_begin_ns,
            open_end_ns=open_end_ns,
            first_frame_ns=previous.first_frame_ns if previous else None,
            recognition_start_ns=(
                previous.recognition_start_ns if previous else None
            ),
            terminal_ns=previous.terminal_ns if previous else None,
        )

    def anchor_recognition(self, first_frame_ns: int) -> None:
        """Re-arm the recognition window at the first valid frame (D1).

        Idempotent: the FIRST valid frame owns the anchor, so a later
        frame can never slide the window forward. Without this call the
        engine behaves exactly as before (deadline armed at start()).
        """
        if self._recognition_anchored:
            return
        if self._start_ns is None or self._deadline_ns is None:
            raise RuntimeError("SessionEngine must be started before anchoring")
        if first_frame_ns < 0:
            raise ValueError(
                f"first_frame_ns must be >= 0, got {first_frame_ns}"
            )
        # The window runs forward from the first valid frame, never
        # backward: a first frame captured before start() (possible with
        # a replay clock) must not pull the deadline earlier than start.
        anchor_ns = max(first_frame_ns, self._start_ns)
        self._deadline_ns = anchor_ns + int(self.profile.timeout_ms * 1_000_000)
        self._recognition_anchored = True
        self._first_frame_ns = first_frame_ns
        previous = self._timing_marks
        self._timing_marks = TimingMarks(
            open_begin_ns=previous.open_begin_ns if previous else None,
            open_end_ns=previous.open_end_ns if previous else None,
            first_frame_ns=first_frame_ns,
            recognition_start_ns=anchor_ns,
            terminal_ns=previous.terminal_ns if previous else None,
        )

    def _clear_support_window(self) -> None:
        self._current_candidate = None
        self._support_sequences.clear()
        self._last_support_ns = None
        self._last_box = None

    def observe(self, obs: FrameObservation) -> SessionResult | None:
        """Process a single incoming frame observation."""
        if (
            self._start_ns is None
            or self._deadline_ns is None
            or self._session_id is None
        ):
            raise RuntimeError("SessionEngine must be started before observe()")

        # If already terminal, return the cached terminal result (idempotency)
        if self._terminal_result is not None:
            return self._terminal_result

        self._frames_sampled += 1

        # Check for sequence or time anomalies (Error termination)
        if obs.sequence <= self._last_sequence:
            return self._terminate_with_error(
                obs.captured_ns,
                (f"duplicate_sequence: {obs.sequence} <= {self._last_sequence}",),
            )
        self._last_sequence = obs.sequence

        if obs.captured_ns < self._last_observation_ns:
            return self._terminate_with_error(
                obs.captured_ns,
                (f"time_backwards: {obs.captured_ns} < {self._last_observation_ns}",),
            )
        self._last_observation_ns = obs.captured_ns

        # Check generation and gallery digest consistency
        if (
            obs.model_generation != self.model_generation
            or obs.gallery_digest != self.gallery_digest
        ):
            return self._terminate_with_error(
                obs.captured_ns,
                ("model_generation_or_gallery_digest_changed_mid_session",),
            )

        # Check deadline: if observation captured past deadline, terminate with timeout
        if obs.captured_ns > self._deadline_ns:
            self._emit_event(
                sequence=obs.sequence,
                event_type="late_processing",
                accepted=False,
                reset_reason="deadline_exceeded",
                support_before=len(self._support_sequences),
                support_after=0,
                candidate_before=self._current_candidate,
                candidate_after=None,
                terminal_status=SessionStatus.timeout.value,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return self.finish(obs.captured_ns, reason="deadline_exceeded")

        # Multi-face -> terminal invalid_input (restart required)
        if obs.face_count > 1:
            self._frames_rejected += 1
            self._clear_support_window()
            self._emit_event(
                sequence=obs.sequence,
                event_type="rejected",
                accepted=False,
                reset_reason="input_multiple_faces",
                support_before=len(self._support_sequences),
                support_after=0,
                candidate_before=self._current_candidate,
                candidate_after=None,
                terminal_status=SessionStatus.invalid_input.value,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return self._terminate_terminal(
                status=SessionStatus.invalid_input,
                identity=None,
                reason_codes=("input_multiple_faces", "session_restart_required"),
                now_ns=obs.captured_ns,
            )

        # Zero faces or quality rejected -> clear support window, continue sampling
        if not obs.quality_pass or obs.face_count == 0 or obs.face_box is None:
            self._frames_rejected += 1
            supp_before = len(self._support_sequences)
            self._clear_support_window()
            q_reason = (
                "no_face_detected"
                if obs.face_count == 0
                else f"quality_rejected: {','.join(obs.quality_reasons)}"
            )
            # D1: remember WHY frames were dropped so a round that ends
            # with nothing usable can say "no face was ever seen" rather
            # than the single lumped zero_usable code, and can still tell
            # it apart from "frames arrived but all were rejected".
            self._rejection_reasons.add(
                "no_face_detected" if obs.face_count == 0 else "quality_rejected"
            )
            self._emit_event(
                sequence=obs.sequence,
                event_type="rejected",
                accepted=False,
                reset_reason=q_reason,
                support_before=supp_before,
                support_after=0,
                candidate_before=self._current_candidate,
                candidate_after=None,
                terminal_status=None,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return None

        # Passed detection & quality
        self._frames_usable += 1
        self._observations.append(obs)

        # Continuity displacement check:
        continuity_limit = self.profile.continuity_max_center_delta_ratio
        if continuity_limit is not None and self._last_box is not None:
            cx1, cy1, dim1 = _center_and_dim(self._last_box)
            cx2, cy2, _ = _center_and_dim(obs.face_box)
            delta = math.hypot(cx2 - cx1, cy2 - cy1)
            ratio = delta / max(dim1, 1.0)
            if ratio > continuity_limit:
                # Discontinuous spatial jump -> invalid_input (restart required)
                supp_before = len(self._support_sequences)
                self._clear_support_window()
                self._emit_event(
                    sequence=obs.sequence,
                    event_type="rejected",
                    accepted=False,
                    reset_reason="continuity_jump_detected",
                    support_before=supp_before,
                    support_after=0,
                    candidate_before=self._current_candidate,
                    candidate_after=None,
                    terminal_status=SessionStatus.invalid_input.value,
                    terminal_identity=None,
                    now_ns=obs.captured_ns,
                )
                return self._terminate_terminal(
                    status=SessionStatus.invalid_input,
                    identity=None,
                    reason_codes=(
                        "continuity_jump_detected",
                        f"center_delta_ratio_{ratio:.3f}",
                    ),
                    now_ns=obs.captured_ns,
                )

        self._last_box = obs.face_box

        # Evaluate top match and runner-up margin
        if not obs.identity_scores:
            supp_before = len(self._support_sequences)
            self._clear_support_window()
            self._emit_event(
                sequence=obs.sequence,
                event_type="rejected",
                accepted=False,
                reset_reason="empty_identity_scores",
                support_before=supp_before,
                support_after=0,
                candidate_before=self._current_candidate,
                candidate_after=None,
                terminal_status=None,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return None

        sorted_candidates = sorted(
            obs.identity_scores.items(), key=lambda it: it[1], reverse=True
        )
        top_identity, top_score = sorted_candidates[0]
        runner_up_score = (
            sorted_candidates[1][1] if len(sorted_candidates) > 1 else None
        )

        if runner_up_score is None:
            # None margin cannot qualify for matched
            supp_before = len(self._support_sequences)
            self._clear_support_window()
            self._emit_event(
                sequence=obs.sequence,
                event_type="none_runner_up",
                accepted=False,
                reset_reason="none_runner_up",
                support_before=supp_before,
                support_after=0,
                candidate_before=self._current_candidate,
                candidate_after=None,
                terminal_status=None,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return None

        margin = top_score - runner_up_score

        # Check if frame meets match and margin thresholds
        if (
            top_score < self.profile.match_threshold
            or margin < self.profile.margin_threshold
        ):
            # Did not qualify -> clear support window
            supp_before = len(self._support_sequences)
            self._clear_support_window()
            s_reason = (
                "score_below_threshold"
                if top_score < self.profile.match_threshold
                else "margin_below_threshold"
            )
            self._emit_event(
                sequence=obs.sequence,
                event_type="score_reset",
                accepted=False,
                reset_reason=s_reason,
                support_before=supp_before,
                support_after=0,
                candidate_before=self._current_candidate,
                candidate_after=None,
                terminal_status=None,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return None

        # If continuity limit is None (T1 contract), auto-match is disabled!
        if not self.profile.can_auto_match():
            supp_before = len(self._support_sequences)
            self._clear_support_window()
            self._emit_event(
                sequence=obs.sequence,
                event_type="reset",
                accepted=False,
                reset_reason="auto_match_disabled",
                support_before=supp_before,
                support_after=0,
                candidate_before=self._current_candidate,
                candidate_after=None,
                terminal_status=None,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return None

        # Check minimum interval from previous support frame (>= 200ms)
        min_interval_ns = int(self.profile.min_support_interval_ms * 1_000_000)
        if (
            self._last_support_ns is not None
            and (obs.captured_ns - self._last_support_ns) < min_interval_ns
        ):
            # Too fast, skip accumulating
            self._emit_event(
                sequence=obs.sequence,
                event_type="interval_skip",
                accepted=False,
                reset_reason=None,
                support_before=len(self._support_sequences),
                support_after=len(self._support_sequences),
                candidate_before=self._current_candidate,
                candidate_after=self._current_candidate,
                terminal_status=None,
                terminal_identity=None,
                now_ns=obs.captured_ns,
            )
            return None

        # Cross-person accumulation check
        if self._current_candidate != top_identity:
            # Switched to a different qualified person -> reset and start at 1
            prev_cand = self._current_candidate
            prev_supp = len(self._support_sequences)
            self._current_candidate = top_identity
            self._support_sequences = [obs.sequence]
            self._last_support_ns = obs.captured_ns
            is_term = len(self._support_sequences) >= self.profile.required_support
            term_status = SessionStatus.matched.value if is_term else None
            term_id = self._current_candidate if is_term else None
            self._emit_event(
                sequence=obs.sequence,
                event_type="identity_change",
                accepted=True,
                reset_reason=None,
                support_before=prev_supp,
                support_after=1,
                candidate_before=prev_cand,
                candidate_after=top_identity,
                terminal_status=term_status,
                terminal_identity=term_id,
                now_ns=obs.captured_ns,
            )
            if is_term:
                return self._terminate_terminal(
                    status=SessionStatus.matched,
                    identity=self._current_candidate,
                    reason_codes=(f"supported_{len(self._support_sequences)}_frames",),
                    now_ns=obs.captured_ns,
                )
        else:
            # Same identity -> accumulate support
            prev_supp = len(self._support_sequences)
            self._support_sequences.append(obs.sequence)
            self._last_support_ns = obs.captured_ns
            is_term = len(self._support_sequences) >= self.profile.required_support
            term_status = SessionStatus.matched.value if is_term else None
            term_id = self._current_candidate if is_term else None
            self._emit_event(
                sequence=obs.sequence,
                event_type="continuity",
                accepted=True,
                reset_reason=None,
                support_before=prev_supp,
                support_after=len(self._support_sequences),
                candidate_before=self._current_candidate,
                candidate_after=self._current_candidate,
                terminal_status=term_status,
                terminal_identity=term_id,
                now_ns=obs.captured_ns,
            )
            if is_term:
                return self._terminate_terminal(
                    status=SessionStatus.matched,
                    identity=self._current_candidate,
                    reason_codes=(f"supported_{len(self._support_sequences)}_frames",),
                    now_ns=obs.captured_ns,
                )

        return None


    def finish(self, now_ns: int, reason: str = "timeout") -> SessionResult:
        """Explicitly conclude session at deadline, cancellation, or manual stop."""
        if self._start_ns is None or self._session_id is None:
            raise RuntimeError("SessionEngine must be started before finish()")

        if self._terminal_result is not None:
            return self._terminal_result

        if reason == "cancelled":
            return self._terminate_terminal(
                status=SessionStatus.cancelled,
                identity=None,
                reason_codes=("session_cancelled_by_operator",),
                now_ns=now_ns,
            )

        # D1 §D1-3: a round that ends with no usable evidence must say WHY.
        # The pre-D1 single code (zero_usable_frames_collected) could not
        # tell a dead camera from a person who never faced it from a
        # round that ran out of time before the first frame landed. The
        # field rounds that motivated D1 all landed here.
        if not self._observations:
            if self._frames_sampled == 0:
                # Not one frame ever reached the engine: the camera
                # never delivered. `reason` names which bounded wait ended
                # it (first_frame_timeout for a dry source, or the usual
                # deadline reasons), so a dead camera is never reported as
                # "the operator was not recognised".
                codes: tuple[str, ...] = ("no_frames_captured", reason)
            elif "no_face_detected" in self._rejection_reasons and (
                "quality_rejected" in self._rejection_reasons
            ):
                codes = ("all_frames_rejected_mixed_causes", reason)
            elif "no_face_detected" in self._rejection_reasons:
                codes = ("all_frames_rejected_no_face", reason)
            elif "quality_rejected" in self._rejection_reasons:
                codes = ("all_frames_rejected_quality", reason)
            else:
                codes = ("zero_usable_frames_collected", reason)
            return self._terminate_terminal(
                status=SessionStatus.invalid_input,
                identity=None,
                reason_codes=codes,
                now_ns=now_ns,
            )

        # D1: frames qualified but the multi-frame rule never closed.
        # "A person was visible but the evidence was too thin" is a
        # different operator-facing outcome than "nobody resembling the
        # gallery was here" — and only the first must be reported as
        # insufficient. The gate is the review band: a face whose best
        # score never reached `review_threshold` is a stranger, and
        # 「找不到此註冊人員」 is the honest answer for a stranger.
        # Lumping them together (the pre-rework D1 behaviour) showed
        # "saw a face, evidence too thin" for people who simply were not
        # in the gallery — the mirror image of the defect review
        # ce0a95a7 raised, and it broke the R1 not-found contract.
        #
        # The band is not a match gate: reaching `review` is NOT
        # matched, and a strong single frame is still insufficient under
        # the 3-frame rule. It only decides which sentence the operator
        # reads.
        if self._frames_usable > 0 and (
            len(self._support_sequences) < self.profile.required_support
        ):
            _, baseline_status, _ = compute_baseline_best_quality(
                self._observations, self.profile
            )
            if baseline_status in (
                SessionStatus.review,
                SessionStatus.matched,
            ):
                return self._terminate_terminal(
                    status=SessionStatus.timeout,
                    identity=None,
                    reason_codes=(
                        "insufficient_evidence",
                        f"support_{len(self._support_sequences)}"
                        f"_of_{self.profile.required_support}",
                        f"best_baseline_{baseline_status.value}",
                    ),
                    now_ns=now_ns,
                )

        # Baseline best-frame evaluation
        best_obs, baseline_status, _ = compute_baseline_best_quality(
            self._observations, self.profile
        )

        # If session timed out without satisfying time-consistency, report timeout
        # with diagnostic best-frame band
        diag_code = f"best_baseline_{baseline_status.value}"
        return self._terminate_terminal(
            status=SessionStatus.timeout,
            identity=None,
            reason_codes=("deadline_exceeded", diag_code),
            now_ns=now_ns,
        )

    def terminate_with_error(
        self, now_ns: int, *, reason_codes: tuple[str, ...]
    ) -> SessionResult:
        """D4-F1: conclude as a visible error terminal.

        Public counterpart of `_terminate_with_error`: the controller's
        inference worker owns the failure, but the engine owns the
        terminal, and the window only ever reads a terminal. Without this
        an unexpected worker failure left the round "running" forever
        and the window kept painting a stale result.

        Idempotent: an already concluded session returns its existing
        terminal, so a failure arriving after a real result can never
        overwrite it. `_terminate_terminal` itself has no such guard —
        `finish()` carries its own — so the check lives here, mirroring
        `finish()` rather than assuming the private path provides it.
        """
        if self._terminal_result is not None:
            return self._terminal_result
        return self._terminate_with_error(now_ns, reason_codes)

    def _terminate_with_error(
        self, now_ns: int, reason_codes: tuple[str, ...]
    ) -> SessionResult:
        return self._terminate_terminal(
            status=SessionStatus.error,
            identity=None,
            reason_codes=reason_codes,
            now_ns=now_ns,
        )

    def _terminate_terminal(
        self,
        *,
        status: SessionStatus,
        identity: str | None,
        reason_codes: tuple[str, ...],
        now_ns: int,
    ) -> SessionResult:
        assert self._start_ns is not None
        assert self._session_id is not None

        elapsed_ms = (now_ns - self._start_ns) / 1_000_000.0
        if elapsed_ms < 0.0:
            elapsed_ms = 0.0

        support_seqs = (
            tuple(self._support_sequences) if status == SessionStatus.matched else ()
        )
        matched_ident = identity if status == SessionStatus.matched else None

        # D1: seal the terminal timestamp onto the segmented marks so the
        # record shows where the round's wall clock actually went.
        marks = self._timing_marks
        if marks is not None:
            marks = replace(
                marks, terminal_ns=now_ns
            )

        result = SessionResult(
            session_id=self._session_id,
            schema_version="v1",
            status=status,
            matched_identity=matched_ident,
            reason_codes=reason_codes,
            elapsed_ms=round(elapsed_ms, 2),
            frames_sampled=self._frames_sampled,
            frames_usable=self._frames_usable,
            frames_rejected=self._frames_rejected,
            frames_dropped=self._frames_dropped,
            support_sequences=support_seqs,
            profile_digest=self.profile.profile_digest(),
            model_generation=self.model_generation,
            gallery_digest=self.gallery_digest,
            timing_marks=marks,
        )
        self._terminal_result = result
        return result


def compute_baseline_best_quality(
    observations: list[FrameObservation],
    profile: ResearchProfile,
) -> tuple[FrameObservation | None, SessionStatus, str | None]:
    """Dual strategy baseline: selects highest quality_rank among qualified frames."""
    usable_frames = [
        o
        for o in observations
        if o.quality_pass and o.face_count == 1 and o.face_box is not None
    ]

    if not usable_frames:
        return None, SessionStatus.invalid_input, None

    # Sort descending by quality_rank; earlier sequence breaks tie
    usable_frames.sort(key=lambda o: (o.quality_rank, -o.sequence), reverse=True)
    best_frame = usable_frames[0]

    if not best_frame.identity_scores:
        return best_frame, SessionStatus.unknown, None

    sorted_scores = sorted(
        best_frame.identity_scores.items(), key=lambda it: it[1], reverse=True
    )
    top_ident, top_score = sorted_scores[0]
    runner_up = sorted_scores[1][1] if len(sorted_scores) > 1 else None

    if runner_up is None:
        # None margin cannot be matched
        return best_frame, SessionStatus.unknown, None

    margin = top_score - runner_up

    if top_score >= profile.match_threshold and margin >= profile.margin_threshold:
        return best_frame, SessionStatus.matched, top_ident
    elif top_score >= profile.review_threshold:
        return best_frame, SessionStatus.review, None
    else:
        return best_frame, SessionStatus.unknown, None
