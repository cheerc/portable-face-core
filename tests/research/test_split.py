"""Phase 2B Task E6 tests: prospective holdout, candidate freeze, and read-time guards.

Source of truth:
    - docs/specs/2026-09-16-phase2b-mac-recognition-research.md §8;
    - docs/plans/2026-09-16-phase2b-mac-recognition-execution-plan.md
      §11.2, §12 E6;
    - ADR 0010 (Phase 2B evidence isolation).

Invariants & acceptance:
    - candidate freeze must precede collection of future visits;
    - same visit cannot cross splits (must not appear in both development and holdout);
    - already-used development data cannot be labeled or reused as holdout;
    - candidate/hash swaps after freeze are rejected fail-closed;
    - unreleased holdout data must not be subject to content analysis;
    - legacy read/replay entrypoints (replay_session:180, read_trace:985,
      read_record:406) must not allow sealed holdout bundles to bypass the
      research guard;
    - single-evaluation rule: holdout dataset is evaluated once for confirmation;
      re-evaluating new strategies or candidates on the same holdout is rejected;
    - identical re-run of the same frozen candidate on released holdout is permitted
      for reproducibility and auditing;
    - contamination is audited and recorded; original data is preserved;
    - pure functions have no hidden disk/network side effects; custody is
      managed by recorder;
    - CLI --mode holdout requires valid freeze and authorized release proofs.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from facecore.live.contracts import (
    FrameDiagnostics,
    ResearchProfile,
    SessionResult,
    SessionStatus,
)
from facecore.research.analysis import analyze_batch
from facecore.research.cli import cmd_analyze
from facecore.research.diagnostics import FrameTraceEntry
from facecore.research.experiment import (
    AttemptRecord,
    EvaluationLabel,
    ExperimentManifest,
)
from facecore.research.recorder import ConsentRecord, ResearchRecorder
from facecore.research.replay import ArmOutcome, ReplayRefusal, replay_session
from facecore.research.split import (
    CandidateFreeze,
    HoldoutRelease,
    HoldoutSealedError,
    SplitContaminationError,
    authorize_holdout,
    classify_split,
    freeze_candidate,
    validate_holdout_release,
)


def _now_utc():
    from datetime import datetime, timezone

    return datetime(2026, 9, 16, 8, 0, 0, tzinfo=timezone.utc)


TEST_PROFILE_DIGEST = (
    "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
)


def _manifest(
    experiment_id: str = "exp-e6",
    *,
    profile_digest: str = TEST_PROFILE_DIGEST,
) -> ExperimentManifest:
    return ExperimentManifest.from_dict(
        {
            "identity": {
                "experiment_id": experiment_id,
                "schema_version": "v2",
                "owner": "lead-test",
                "custodian": "custodian-test",
            },
            "software": {"code_sha": "c" * 40, "generation": "gen-e6"},
            "gallery": {"gallery_digest": "gal-e6"},
            "policy": {
                "profile_version": "prof-e6",
                "profile_digest": profile_digest,
            },
            "capture": {"device": "fake"},
            "privacy": {"record_ttl_days": 30},
            "study": {"participants": ["p1", "p2"]},
            "analysis": {"arms": ["A", "B"]},
        }
    )


def _attempt(
    attempt_id: str,
    *,
    experiment_id: str = "exp-e6",
    participant_id: str = "p1",
    visit_id: str = "v1",
    requested_at_utc: str = "2026-09-16T08:00:00Z",
    split: str = "development",
    bundle_ref: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        experiment_id=experiment_id,
        attempt_id=attempt_id,
        participant_id=participant_id,
        visit_id=visit_id,
        condition_id="cond-01",
        attempt_index=1,
        retry_of=None,
        consent_ref="cs-01",
        requested_at_utc=requested_at_utc,
        accepted_at_utc="2026-09-16T08:00:01Z",
        started_at_utc="2026-09-16T08:00:02Z",
        ended_at_utc="2026-09-16T08:00:07Z",
        operational_status="completed",
        error_code=None,
        bundle_ref=bundle_ref,
        split=split,
    )


def _consent(session_id: str, participant_id: str = "p1") -> ConsentRecord:
    return ConsentRecord(
        session_id=session_id,
        participant_id=participant_id,
        record_consent=True,
        image_consent=True,
        consented_at_utc="2026-09-16T08:00:00Z",
        record_expires_at_utc="2026-10-16T08:00:00Z",
        image_expires_at_utc="2026-09-23T08:00:00Z",
    )


def _profile(profile_digest: str = TEST_PROFILE_DIGEST) -> ResearchProfile:
    # A lightweight profile whose profile_digest matches
    return ResearchProfile(
        schema_version="v1",
        profile_version="prof-e6",
        timeout_ms=5000,
        sample_interval_ms=200,
        max_frames=26,
        queue_limit=1,
        required_support=3,
        min_support_interval_ms=200,
        match_threshold=0.45,
        review_threshold=0.30,
        margin_threshold=0.10,
        detector_version="det-e6",
        quality_policy_version="quality-e6",
    )


class TestCandidateFreezeAndAuthorizeHoldout:
    """Pure function contracts for freeze_candidate and authorize_holdout."""

    def test_freeze_candidate_pure_function(self) -> None:
        manifest = _manifest(
            "exp-e6",
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
        )
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future_1", "v_future_2"),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        assert isinstance(freeze, CandidateFreeze)
        assert freeze.freeze_id.startswith("frz_")
        assert freeze.manifest_digest == manifest.digest()
        assert freeze.code_sha == "c" * 40
        assert (
            freeze.profile_digest
            == "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
        )
        assert freeze.analysis_digest == "ana-001"
        assert freeze.planned_visit_ids == ("v_future_1", "v_future_2")
        assert freeze.frozen_at_utc == "2026-09-16T09:00:00Z"

        # Digest must be 64-char sha256 hex
        d = freeze.digest()
        assert len(d) == 64
        int(d, 16)  # must be valid hex

        # Serialization round-trip
        data = freeze.to_dict()
        restored = CandidateFreeze.from_dict(data)
        assert restored == freeze
        assert restored.digest() == freeze.digest()

    def test_freeze_candidate_rejects_empty_or_invalid_inputs(self) -> None:
        manifest = _manifest(
            "exp-e6",
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
        )
        with pytest.raises(ValueError, match="code_sha"):
            freeze_candidate(
                manifest,
                code_sha="",
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                analysis_digest="ana-001",
                planned_visit_ids=("v1",),
            )

        with pytest.raises(ValueError, match="profile_digest"):
            freeze_candidate(
                manifest,
                code_sha="c" * 40,
                profile_digest="",
                analysis_digest="ana-001",
                planned_visit_ids=("v1",),
            )

        with pytest.raises(ValueError, match="analysis_digest"):
            freeze_candidate(
                manifest,
                code_sha="c" * 40,
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                analysis_digest="",
                planned_visit_ids=("v1",),
            )

        with pytest.raises(ValueError, match="planned_visit_ids"):
            freeze_candidate(
                manifest,
                code_sha="c" * 40,
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                analysis_digest="ana-001",
                planned_visit_ids=(),
            )

        with pytest.raises(ValueError, match="duplicate"):
            freeze_candidate(
                manifest,
                code_sha="c" * 40,
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                analysis_digest="ana-001",
                planned_visit_ids=("v1", "v1"),
            )

        # Manifest policy profile_digest mismatch
        with pytest.raises(ValueError, match="profile digest mismatch"):
            freeze_candidate(
                manifest,
                code_sha="c" * 40,
                profile_digest="prof-DIFFERENT",
                analysis_digest="ana-001",
                planned_visit_ids=("v1",),
            )

    def test_authorize_holdout_pure_function(self) -> None:
        manifest = _manifest(
            "exp-e6",
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
        )
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future_1",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        release = authorize_holdout(
            freeze,
            operator_decision_id="d-20260916085552511689-1",
            authorized_at_utc="2026-09-16T12:00:00Z",
        )
        assert isinstance(release, HoldoutRelease)
        assert release.release_id.startswith("rel_")
        assert release.freeze_id == freeze.freeze_id
        assert release.freeze_digest == freeze.digest()
        assert release.operator_decision_id == "d-20260916085552511689-1"
        assert release.authorized_at_utc == "2026-09-16T12:00:00Z"

        # Validation helper
        errors = validate_holdout_release(freeze, release)
        assert errors == []

        # Serialization round-trip
        data = release.to_dict()
        restored = HoldoutRelease.from_dict(data)
        assert restored == release
        assert restored.digest() == release.digest()

    def test_authorize_holdout_rejects_empty_decision_id(self) -> None:
        manifest = _manifest(
            "exp-e6",
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
        )
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future_1",),
        )
        with pytest.raises(ValueError, match="operator_decision_id"):
            authorize_holdout(freeze, operator_decision_id="")


class TestSplitInvariants:
    """Core split invariant enforcement (ADR 0010, Spec §8)."""

    def test_classify_split(self) -> None:
        manifest = _manifest(
            "exp-e6",
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
        )
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_holdout_1", "v_holdout_2"),
        )
        assert classify_split("v_holdout_1", freeze) == "holdout"
        assert classify_split("v_holdout_2", freeze) == "holdout"
        assert classify_split("v_dev_1", freeze) == "development"
        assert classify_split("v_any", None) == "development"

    def test_same_visit_cannot_cross_split(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        # Record attempt 1 under development with visit v1
        att_dev = _attempt(
            "s1",
            visit_id="v1",
            split="development",
            requested_at_utc="2026-09-16T08:00:00Z",
        )
        recorder.begin_attempt(manifest, att_dev, _consent("s1"))

        # Freeze planned for future visit v_future
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        # Attempt with visit v1 labeled as holdout -> cross-split violation!
        att_cross = _attempt(
            "s2",
            visit_id="v1",
            split="holdout",
            requested_at_utc="2026-09-16T10:00:00Z",
        )
        with pytest.raises(SplitContaminationError, match="cross-split"):
            recorder.begin_attempt(manifest, att_cross, _consent("s2"))

        # Check contamination record logged
        contam = recorder.list_contamination("exp-e6")
        assert len(contam) >= 1
        assert any("cross_split" in c.reason for c in contam)

    def test_already_used_development_visit_cannot_be_holdout(
        self, tmp_path: Path
    ) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        # Record development visit v_past
        att_dev = _attempt("s1", visit_id="v_past", split="development")
        recorder.begin_attempt(manifest, att_dev, _consent("s1"))

        # Now try to freeze candidate naming v_past as a planned holdout visit
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_past",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        with pytest.raises(
            SplitContaminationError, match="already has recorded attempts"
        ):
            recorder.record_freeze(freeze)

        # Contamination recorded
        contam = recorder.list_contamination("exp-e6")
        assert len(contam) >= 1
        assert any(
            "retroactive_holdout" in c.reason or "already_used" in c.reason
            for c in contam
        )

    def test_freeze_must_precede_future_visit_collection(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        # Freeze established at 09:00:00Z
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        # Attempt requested at 08:30:00Z (before freeze!) claiming to be holdout
        att_early = _attempt(
            "s_early",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T08:30:00Z",
        )
        with pytest.raises(SplitContaminationError, match="precede"):
            recorder.begin_attempt(manifest, att_early, _consent("s_early"))

        # Contamination recorded
        contam = recorder.list_contamination("exp-e6")
        assert len(contam) >= 1

    def test_timestamp_suffix_cannot_bypass_freeze_order(
        self, tmp_path: Path
    ) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest=TEST_PROFILE_DIGEST,
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T01:00:00Z",
        )
        recorder.record_freeze(freeze)

        # 00:59Z is before 01:00Z, despite the +08:00 lexical prefix.
        before = _attempt(
            "s_suffix_before",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T08:59:00+08:00",
        )
        with pytest.raises(SplitContaminationError, match="precede"):
            recorder.begin_attempt(manifest, before, _consent("s_suffix_before"))

        # A genuinely later instant remains admissible.
        after = _attempt(
            "s_suffix_after",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:01:00+08:00",
        )
        recorder.begin_attempt(manifest, after, _consent("s_suffix_after"))
        assert (
            recorder.list_attempts(experiment_id="exp-e6")[0].attempt_id
            == "s_suffix_after"
        )

    def test_candidate_or_hash_swap_rejected_after_freeze(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        # Attempt to record a DIFFERENT freeze for same experiment
        freeze_swapped = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-002",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:30:00Z",
        )
        with pytest.raises(SplitContaminationError, match="already exists"):
            recorder.record_freeze(freeze_swapped)

        # Contamination recorded
        contam = recorder.list_contamination("exp-e6")
        assert len(contam) >= 1
        assert any("overwrite" in c.reason or "swap" in c.reason for c in contam)


class TestRecorderCustodyAndReadTimeGuards:
    """Read-time guards refuse content inspection prior to authorized release."""

    def test_unreleased_holdout_read_record_refused(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        # Freeze planned visit v_future
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        # Commit an active session for bundle
        recorder.begin("b_holdout", _consent("b_holdout"))
        recorder.commit(
            SessionResult(
                session_id="b_holdout",
                schema_version="v1",
                status=SessionStatus.matched,
                matched_identity="p1",
                reason_codes=("matched",),
                elapsed_ms=1000.0,
                frames_sampled=1,
                frames_usable=1,
                frames_rejected=0,
                frames_dropped=0,
                support_sequences=(1,),
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                model_generation="gen-e6",
                gallery_digest="gal-e6",
            )
        )

        # Record attempt linked to bundle
        att = _attempt(
            "s_holdout",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
            bundle_ref="b_holdout",
        )
        recorder.begin_attempt(manifest, att, _consent("s_holdout"))

        # Reading record of unreleased holdout MUST raise HoldoutSealedError
        with pytest.raises(HoldoutSealedError, match="sealed holdout"):
            recorder.read_record("b_holdout")

        # Reading frame must also raise HoldoutSealedError
        with pytest.raises(HoldoutSealedError, match="sealed holdout"):
            recorder.read_frame("b_holdout", 0)

        # Contamination recorded
        contam = recorder.list_contamination("exp-e6")
        assert len(contam) >= 1
        assert any("premature" in c.reason for c in contam)

        # Original data is NOT deleted!
        assert (store_dir / "b_holdout" / "manifest.json").is_file()

    def test_unreleased_holdout_read_trace_and_label_refused(
        self, tmp_path: Path
    ) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        att = _attempt(
            "s_holdout",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
        )
        recorder.begin_attempt(manifest, att, _consent("s_holdout"))

        # Append a trace entry and write a label
        diag = FrameDiagnostics(
            sequence=1,
            original_shape=(16, 16, 3),
            normalized_shape=(16, 16, 3),
            orientation=0,
            mirrored=False,
            face_count=1,
            detector_confidence=0.99,
            face_box=(4.0, 4.0, 8.0, 8.0),
            landmarks=None,
            quality_status="accepted",
        )
        entry = FrameTraceEntry(
            sequence=1,
            captured_ns=0,
            processed_ns=10_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(4.0, 4.0, 8.0, 8.0),
            identity_score_pairs=(("p1", 0.9),),
            quality_rank=0.9,
            model_generation="gen-e6",
            gallery_digest="gal-e6",
            diagnostics=diag,
            decision_event=None,
            staged_index=1,
        )
        recorder.append_trace("s_holdout", entry)

        lbl = EvaluationLabel(
            attempt_id="s_holdout",
            revision=1,
            kind="enrolled",
            identity_id="p1",
            actor_ref="evaluator",
            labeled_at="2026-09-16T09:10:00Z",
        )
        recorder.write_label(lbl)

        # Reading trace or label of sealed holdout MUST be refused
        with pytest.raises(HoldoutSealedError, match="sealed holdout"):
            recorder.read_trace("s_holdout")

        with pytest.raises(HoldoutSealedError, match="sealed holdout"):
            recorder.read_label("s_holdout")

        with pytest.raises(HoldoutSealedError, match="sealed holdout"):
            recorder.read_label_history("s_holdout")

    def test_legacy_replay_session_refuses_sealed_bundle(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        recorder.begin("b_sealed", _consent("b_sealed"))
        recorder.commit(
            SessionResult(
                session_id="b_sealed",
                schema_version="v1",
                status=SessionStatus.matched,
                matched_identity="p1",
                reason_codes=("matched",),
                elapsed_ms=1000.0,
                frames_sampled=0,
                frames_usable=0,
                frames_rejected=0,
                frames_dropped=0,
                support_sequences=(),
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                model_generation="gen-e6",
                gallery_digest="gal-e6",
            )
        )
        att = _attempt(
            "s_sealed",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
            bundle_ref="b_sealed",
        )
        recorder.begin_attempt(manifest, att, _consent("s_sealed"))

        # Legacy replay_session must raise ReplayRefusal(kind="sealed")
        with pytest.raises(ReplayRefusal) as exc_info:
            replay_session(
                "b_sealed",
                store_root=store_dir,
                key_dir=key_dir,
                clock=_now_utc,
                profile=_profile(
                    "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
                ),
                scorer=lambda f: None,  # type: ignore[return-value]
                model_generation="gen-e6",
                gallery_digest="gal-e6",
            )
        assert exc_info.value.kind == "sealed"

    def test_unreleased_holdout_permits_noncontent_integrity_and_count_checks(
        self, tmp_path: Path
    ) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        att = _attempt(
            "s_holdout",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
        )
        recorder.begin_attempt(manifest, att, _consent("s_holdout"))

        # Listing attempts returns metadata without accessing content
        attempts = recorder.list_attempts("exp-e6")
        assert len(attempts) == 1
        assert attempts[0].attempt_id == "s_holdout"
        assert attempts[0].split == "holdout"


class TestHoldoutReleaseAndSingleEvaluation:
    """Release unlocks analysis; single-evaluation rule strictly enforced."""

    def test_authorized_release_unlocks_content_analysis(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6")

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        att = _attempt(
            "s_holdout",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
        )
        recorder.begin_attempt(manifest, att, _consent("s_holdout"))

        # Authorize release
        release = authorize_holdout(
            freeze, operator_decision_id="d-20260916085552511689-1"
        )
        recorder.record_release(release)

        # Now reads succeed without HoldoutSealedError
        attempts = recorder.list_attempts("exp-e6")
        assert len(attempts) == 1

        # Content analysis in holdout mode succeeds
        outcomes = [
            ArmOutcome(
                attempt_id="s_holdout",
                run_id="run-1",
                arm_id="A",
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                selected_sequences=(1,),
                support_sequences=(1,),
                terminal=SessionStatus.matched.value,
                matched_identity="p1",
                collection_extent="full",
                decision_time_ns=1_000_000_000,
                decision_codes=("A_matched",),
                frames_read=1,
                frames_scored=1,
                frames_consumed=1,
                frames_staged=1,
            ),
            ArmOutcome(
                attempt_id="s_holdout",
                run_id="run-1",
                arm_id="B",
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                selected_sequences=(1,),
                support_sequences=(1,),
                terminal=SessionStatus.matched.value,
                matched_identity="p1",
                collection_extent="full",
                decision_time_ns=1_000_000_000,
                decision_codes=("B_matched",),
                frames_read=1,
                frames_scored=1,
                frames_consumed=1,
                frames_staged=1,
            ),
        ]
        labels = [
            EvaluationLabel(
                "s_holdout", 1, "enrolled", "p1", "evaluator", "2026-09-16T10:00:00Z"
            )
        ]

        batch = analyze_batch(
            attempts,
            outcomes,
            labels,
            mode="holdout",
            freeze=freeze,
            release=release,
            profile=_profile(
                "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
            ),
        )
        assert batch.attempted == 1
        assert batch.truth_known_enrolled == 1
        assert batch.arm_a.correct == 1
        assert batch.arm_b.correct == 1

    def test_holdout_evaluation_can_be_reproduced_identically(
        self, tmp_path: Path
    ) -> None:
        manifest = _manifest("exp-e6")
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        release = authorize_holdout(
            freeze, operator_decision_id="d-20260916085552511689-1"
        )
        att = _attempt("s1", visit_id="v_future", split="holdout")
        outcomes = [
            ArmOutcome(
                attempt_id="s1",
                run_id="run-1",
                arm_id="A",
                profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
                selected_sequences=(),
                support_sequences=(),
                terminal=SessionStatus.timeout.value,
                matched_identity=None,
                collection_extent="full",
                decision_time_ns=5_000_000_000,
                decision_codes=(),
                frames_read=1,
                frames_scored=1,
                frames_consumed=1,
                frames_staged=1,
            )
        ]
        labels = [
            EvaluationLabel(
                "s1", 1, "enrolled", "p1", "evaluator", "2026-09-16T10:00:00Z"
            )
        ]

        # First run
        b1 = analyze_batch(
            [att],
            outcomes,
            labels,
            mode="holdout",
            freeze=freeze,
            release=release,
            profile=_profile(
                "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
            ),
        )
        # Second run (exact reproduction / audit)
        b2 = analyze_batch(
            [att],
            outcomes,
            labels,
            mode="holdout",
            freeze=freeze,
            release=release,
            profile=_profile(
                "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
            ),
        )
        assert b1.to_dict() == b2.to_dict()

    def test_holdout_reuse_with_new_candidate_or_strategy_rejected(
        self, tmp_path: Path
    ) -> None:
        manifest = _manifest("exp-e6")
        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest="c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        release = authorize_holdout(
            freeze, operator_decision_id="d-20260916085552511689-1"
        )
        att = _attempt("s1", visit_id="v_future", split="holdout")

        # Running with a DIFFERENT profile on the same holdout must be rejected
        new_profile = ResearchProfile(
            schema_version="v1",
            profile_version="prof-e6-new",
            timeout_ms=5000,
            sample_interval_ms=200,
            max_frames=26,
            queue_limit=1,
            required_support=3,
            min_support_interval_ms=200,
            match_threshold=0.50,  # changed threshold
            review_threshold=0.30,
            margin_threshold=0.10,
            detector_version="det-e6",
            quality_policy_version="quality-e6",
        )
        with pytest.raises(SplitContaminationError, match="profile digest"):
            analyze_batch(
                [att],
                [],
                [],
                mode="holdout",
                freeze=freeze,
                release=release,
                profile=new_profile,
            )


class TestCLIAnalyzeHoldoutMode:
    """CLI analyze integration testing for holdout and development modes."""

    def test_cli_analyze_holdout_unreleased_fails_closed(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        profile_file = tmp_path / "profile.json"
        prof = _profile(
            "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
        )
        profile_file.write_text(json.dumps(prof.to_dict()))

        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6", profile_digest=prof.profile_digest())

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest=prof.profile_digest(),
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        att = _attempt(
            "s1",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
        )
        recorder.begin_attempt(manifest, att, _consent("s1"))

        # CLI analyze in holdout mode WITHOUT release MUST fail closed (exit 4)
        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e6",
            mode="holdout",
            profile_path=profile_file,
        )
        assert rc == 4

    def test_cli_analyze_holdout_hash_mismatch_fails_closed(
        self, tmp_path: Path
    ) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        profile_file = tmp_path / "profile.json"
        prof = _profile(
            "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
        )
        profile_file.write_text(json.dumps(prof.to_dict()))

        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6", profile_digest=prof.profile_digest())

        freeze = CandidateFreeze(
            freeze_id="frz_tampered",
            experiment_id="exp-e6",
            manifest_digest=manifest.digest(),
            code_sha="c" * 40,
            profile_digest="prof-DIFFERENT-DIGEST",
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        # Force write freeze with mismatched digest to simulate tampering
        (store_dir / "_splits" / "exp-e6").mkdir(parents=True)
        recorder._atomic_write_json(
            store_dir / "_splits" / "exp-e6" / "freeze.json", freeze.to_dict()
        )
        release = authorize_holdout(
            freeze, operator_decision_id="d-20260916085552511689-1"
        )
        recorder._atomic_write_json(
            store_dir / "_splits" / "exp-e6" / "release.json", release.to_dict()
        )

        att = _attempt(
            "s1",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
        )
        recorder.begin_attempt(manifest, att, _consent("s1"))

        # Hash mismatch must fail closed (exit 4)
        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e6",
            mode="holdout",
            profile_path=profile_file,
        )
        assert rc == 4

        # Contamination recorded
        contam = recorder.list_contamination("exp-e6")
        assert len(contam) >= 1

    def test_cli_analyze_holdout_released_succeeds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        profile_file = tmp_path / "profile.json"
        prof = _profile(
            "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
        )
        profile_file.write_text(json.dumps(prof.to_dict()))

        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6", profile_digest=prof.profile_digest())

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest=prof.profile_digest(),
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)
        release = authorize_holdout(
            freeze, operator_decision_id="d-20260916085552511689-1"
        )
        recorder.record_release(release)

        att = _attempt(
            "s1",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
        )
        recorder.begin_attempt(manifest, att, _consent("s1"))

        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e6",
            mode="holdout",
            profile_path=profile_file,
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "=== Batch Analysis: exp-e6 ===" in captured.out
        assert "Attempted: 1" in captured.out

    def test_cli_analyze_development_mode_excludes_holdout_attempts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store_dir = tmp_path / "store"
        key_dir = tmp_path / "keys"
        profile_file = tmp_path / "profile.json"
        prof = _profile(
            "c77088e765f8f90de132b71e4ccb014a3b08fe6f8fbee8024905575e559a2265"
        )
        profile_file.write_text(json.dumps(prof.to_dict()))

        recorder = ResearchRecorder(store_dir, key_dir, clock=_now_utc)
        manifest = _manifest("exp-e6", profile_digest=prof.profile_digest())

        freeze = freeze_candidate(
            manifest,
            code_sha="c" * 40,
            profile_digest=prof.profile_digest(),
            analysis_digest="ana-001",
            planned_visit_ids=("v_future",),
            frozen_at_utc="2026-09-16T09:00:00Z",
        )
        recorder.record_freeze(freeze)

        # Development attempt
        att_dev = _attempt(
            "s_dev",
            visit_id="v_dev",
            split="development",
            requested_at_utc="2026-09-16T08:00:00Z",
        )
        recorder.begin_attempt(manifest, att_dev, _consent("s_dev"))

        # Holdout attempt
        att_hold = _attempt(
            "s_hold",
            visit_id="v_future",
            split="holdout",
            requested_at_utc="2026-09-16T09:05:00Z",
        )
        recorder.begin_attempt(manifest, att_hold, _consent("s_hold"))

        # Running analyze in development mode must count only s_dev, not s_hold
        rc = cmd_analyze(
            store=store_dir,
            key_dir=key_dir,
            experiment_id="exp-e6",
            mode="development",
            profile_path=profile_file,
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "Attempted: 1" in captured.out
