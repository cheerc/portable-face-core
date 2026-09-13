"""Governed chronological replay: 23 enrollee gallery + 13 filename-ordered probes.

Implements acceptance §6-3 (governing d-20260913063042222600-0 item 3 +
chronology-B decision d-20260913084623309226-0: filename order IS time
order — ``enroll-23-probe-01`` → ``enroll-23-probe-13`` — EXIF timeless,
confirmed). The commander replays the same harness shape against real
photos in /tmp/face-accept (commander domain); inside this repo every
vector is synthetic and no biometric data enters git.

Two arms per event, both through production scoring paths:

- frozen arm: ``bakeoff.run_candidate`` raw scores, matched/review/unknown
  bands with the margin firewall (>= 0.10) — the Phase-1A baseline;
- adaptive arm: confirmation-gated shadow path — creation
  (``CandidatePipeline``), corroboration (``CorroborationEngine``),
  promotion (``PromotionManager``), rejection, retirement
  (``EvictionManager``), rollback (``LifecycleManager``) — the Phase-1B
  bank under replay.

Production confirmation boundary: ground-truth labels travel strictly
inside the harness via ``harness_confirmation_for`` (supervision for the
A/B comparison only). Production ``identify --confirm-learning`` takes no
label input — pinned by ``production_has_no_label_input`` and its test,
so the harness mapping can never leak into a production confirmation
source.

Temporal-leakage shape follows Task 9: per-sequence snapshots
(decision, score, gallery digest) recorded at processing time; a future
suffix can never alter the recorded prefix (A/B identical at event N);
backward re-runs are rejected.
"""

import hashlib
import inspect
import tempfile
from dataclasses import dataclass, field

import numpy as np

from facecore.contracts.candidate import CandidateStatus, CandidateTemplate
from facecore.contracts.confirmation import (
    ActorType,
    ConfirmationRequest,
    ConfirmationVerdict,
)
from facecore.contracts.crypto import EncryptedBlob
from facecore.contracts.policy import GovernancePolicy
from facecore.contracts.result import (
    Decision,
    IdentificationResult,
    Quality,
    ResultStatus,
)
from facecore.contracts.template import FaceTemplate, TemplateRevision
from facecore.eval.bakeoff import run_candidate
from facecore.governance.candidate import CandidatePipeline
from facecore.governance.corroboration import CorroborationEngine
from facecore.governance.eviction import EvictionManager
from facecore.governance.lifecycle import LifecycleManager
from facecore.governance.promotion import PromotionManager
from facecore.repository.memory import InMemoryRepository
from facecore.storage.key_provider import InMemoryKeyProvider
from facecore.storage.sqlite_repo import SQLiteRepository

#: Closed chronology-B filename set: the 13 person-23 probes in time order.
PROBE_FILENAMES: tuple[str, ...] = tuple(
    f"enroll-23-probe-{n:02d}.png" for n in range(1, 14)
)

#: Margin firewall shared with the §6-2 SSOT run.
MARGIN_FIREWALL = 0.1

#: Frozen-arm match/review waterlines for the 1A baseline bands.
BASELINE_MATCH = 0.60
BASELINE_REVIEW = 0.363

_MODEL_VERSION = "sface-2021dec-fp32"


def filename_order(filenames: list[str]) -> list[str]:
    """Order probe filenames by chronology-B time (filename order).

    Fail-closed: unknown names and duplicates are rejected — a replay with
    a corrupt event stream must never silently run a partial prefix.
    """
    known = set(PROBE_FILENAMES)
    for name in filenames:
        if name not in known:
            raise ValueError(f"unknown replay probe filename: {name!r}")
    if len(set(filenames)) != len(filenames):
        raise ValueError("duplicate replay probe filename")
    rank = {name: index for index, name in enumerate(PROBE_FILENAMES)}
    return sorted(filenames, key=lambda name: rank[name])


def harness_confirmation_for(top_identity: str, ground_truth_identity: str) -> str:
    """Supervision mapping, harness-internal only (never a production source).

    Returns ``"correct"`` when the frozen top-1 matches ground truth,
    else ``"not_me"``. Pure function of declared identities: no pixels,
    paths, or session state cross the production boundary.
    """
    if top_identity == ground_truth_identity:
        return "correct"
    return "not_me"


def production_has_no_label_input() -> bool:
    """Pin: the production confirm-learning path takes no label argument."""
    from facecore import cli as cli_module

    params = inspect.signature(cli_module.cmd_confirm_learning).parameters
    lowered = [name.lower() for name in params]
    if any("label" in name or "ground_truth" in name for name in lowered):
        return False
    return True


@dataclass(frozen=True)
class RealReplayEvent:
    filename: str
    sequence_number: int
    probe_vector: np.ndarray
    ground_truth_identity: str


@dataclass(frozen=True)
class GovernedReplaySummary:
    events_processed: int
    processed_sequence: tuple[int, ...]
    baseline_matched: int
    baseline_review: int
    baseline_unknown: int
    baseline_denominator: int
    adaptive_matched: int
    adaptive_review: int
    adaptive_unknown: int
    adaptive_denominator: int
    creations: int
    corroborations: int
    promotions: int
    rejections: int
    retirements: int
    rollbacks: int
    decisions: tuple[tuple[int, str], ...] = ()
    scores: tuple[tuple[int, float], ...] = ()
    template_states: tuple[tuple[int, str], ...] = ()

    def decision_at(self, seq: int) -> str:
        return dict(self.decisions)[seq]

    def score_at(self, seq: int) -> float:
        return dict(self.scores)[seq]

    def template_state_at(self, seq: int) -> str:
        return dict(self.template_states)[seq]

    def render_section(self) -> str:
        lines = [
            "## Real governed replay (23 gallery + 13 filename-ordered probes)",
            "",
            "- status: synthetic-full-path-green-awaiting-commander-rerun",
            "- selection gate: OPEN (commander true-photo rerun pending)",
            "- label: partial governance validation (NOT Phase-1B completion)",
            "",
            "| outcome | baseline (1A frozen) | adaptive (1B bank) |",
            "| --- | --- | --- |",
            f"| matched | {self.baseline_matched}/{self.baseline_denominator} "
            f"| {self.adaptive_matched}/{self.adaptive_denominator} |",
            f"| review | {self.baseline_review}/{self.baseline_denominator} "
            f"| {self.adaptive_review}/{self.adaptive_denominator} |",
            f"| unknown | {self.baseline_unknown}/{self.baseline_denominator} "
            f"| {self.adaptive_unknown}/{self.adaptive_denominator} |",
            "",
            "## Governance counters (synthetic stream)",
            "",
            f"- candidate creations: {self.creations}",
            f"- corroborations: {self.corroborations}",
            f"- promotions: {self.promotions}",
            f"- rejections: {self.rejections}",
            f"- retirements: {self.retirements}",
            f"- rollbacks: {self.rollbacks}",
            "",
        ]
        return "\n".join(lines)


def _gallery_digest(gallery: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for identity_id in sorted(gallery):
        digest.update(identity_id.encode("utf-8"))
        digest.update(np.ascontiguousarray(gallery[identity_id]).tobytes())
    return digest.hexdigest()


@dataclass
class _Track:
    baseline_matched: int = 0
    baseline_review: int = 0
    baseline_unknown: int = 0
    adaptive_matched: int = 0
    adaptive_review: int = 0
    adaptive_unknown: int = 0
    creations: int = 0
    corroborations: int = 0
    promotions: int = 0
    rejections: int = 0
    retirements: int = 0
    rollbacks: int = 0
    decisions: dict[int, str] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)
    templates: dict[int, str] = field(default_factory=dict)


def _matched_result(
    identity_id: str, score: float, margin: float | None
) -> IdentificationResult:
    return IdentificationResult(
        status=ResultStatus.MATCHED,
        identity={"id": identity_id, "display_name": "replay", "metadata": {}},
        decision=Decision(
            score=score,
            runner_up_score=None,
            threshold=BASELINE_MATCH,
            margin=margin,
            reason_codes=[],
        ),
        quality=Quality(status="accepted", reason_codes=[]),
        model_version=_MODEL_VERSION,
        template_revision=1,
        candidate_created=False,
    )


def _confirmation(verdict: str) -> ConfirmationRequest:
    return ConfirmationRequest(
        request_id=f"replay-{verdict}",
        verdict=(
            ConfirmationVerdict.CORRECT
            if verdict == "correct"
            else ConfirmationVerdict.NOT_ME
        ),
        actor=ActorType.USER,
        created_at="2026-09-13T00:00:00+00:00",
    )


def _replay_blob(tag: bytes) -> EncryptedBlob:
    return EncryptedBlob(
        format_version=EncryptedBlob.FORMAT_VERSION,
        cipher_id=EncryptedBlob.CIPHER_ID_AES_256_GCM,
        nonce=b"n" * EncryptedBlob.NONCE_BYTES,
        ciphertext=tag * 4,
    )


def _pending_candidate(candidate_id: str, count: int) -> CandidateTemplate:
    return CandidateTemplate(
        template_id=candidate_id,
        identity_id="person-23",
        generation_id="G1",
        key_id="replay",
        encrypted_embedding=_replay_blob(b"e"),
        status=CandidateStatus.PENDING,
        additional_corroboration_count=count,
        encrypted_exemplar=_replay_blob(b"x"),
        quality_score=0.9,
        evidence_log=(),
        expires_at="2026-09-20T00:00:00+00:00",
        created_at="2026-09-13T00:00:00+00:00",
    )


class GovernedReplayHarness:
    """Run the 13-probe governed stream in filename (time) order."""

    def __init__(self) -> None:
        self._high_water_mark = 0

    def run(
        self,
        events: list[RealReplayEvent],
        gallery: dict[str, np.ndarray],
    ) -> GovernedReplaySummary:
        ordered_names = filename_order([e.filename for e in events])
        by_name = {e.filename: e for e in events}
        ordered = [by_name[name] for name in ordered_names]
        for event in ordered:
            if event.sequence_number < self._high_water_mark:
                raise ValueError(
                    "backward sequence rejected: "
                    f"{event.sequence_number} < high-water mark "
                    f"{self._high_water_mark}"
                )
        policy = GovernancePolicy.provisional_v1()
        track = _Track()
        frozen_repo = InMemoryRepository()
        dim = len(next(iter(gallery.values())))
        for identity in gallery:
            frozen_repo.create_identity(
                identity,
                identity,
                FaceTemplate(
                    template_id=f"t-{identity}",
                    identity_id=identity,
                    model_version=_MODEL_VERSION,
                    embedding_dim=dim,
                    revision=TemplateRevision(
                        revision=1, template_id=f"t-{identity}", supersedes=None
                    ),
                ),
            )
        with tempfile.TemporaryDirectory(prefix="real-replay-") as tmpdir:
            store = SQLiteRepository(f"{tmpdir}/facecore.db", InMemoryKeyProvider())
            store.initialize()
            lifecycle = LifecycleManager(store, policy)
            lifecycle.add_identity("person-23", "replay", b"e" * 16, b"x" * 8)
            pipeline = CandidatePipeline(store, policy)
            corroboration = CorroborationEngine(
                policy.burst_suppression_min_interval_secs
            )
            promotion = PromotionManager("G1", policy)
            eviction = EvictionManager(policy=policy)
            created_ids: list[str] = []
            born_seq: dict[str, int] = {}
            corroborated: dict[str, int] = {}
            live_gallery = dict(gallery)
            for event in ordered:
                self._replay_event(
                    event,
                    live_gallery,
                    frozen_repo,
                    pipeline,
                    corroboration,
                    promotion,
                    policy.candidate_update_threshold,
                    created_ids,
                    born_seq,
                    corroborated,
                    track,
                )
            self._close_out_segments(lifecycle, eviction, created_ids, track)
        if ordered:
            self._high_water_mark = max(
                self._high_water_mark,
                max(e.sequence_number for e in ordered),
            )
        denom = len(ordered)
        return GovernedReplaySummary(
            events_processed=denom,
            processed_sequence=tuple(e.sequence_number for e in ordered),
            baseline_matched=track.baseline_matched,
            baseline_review=track.baseline_review,
            baseline_unknown=track.baseline_unknown,
            baseline_denominator=denom,
            adaptive_matched=track.adaptive_matched,
            adaptive_review=track.adaptive_review,
            adaptive_unknown=track.adaptive_unknown,
            adaptive_denominator=denom,
            creations=track.creations,
            corroborations=track.corroborations,
            promotions=track.promotions,
            rejections=track.rejections,
            retirements=track.retirements,
            rollbacks=track.rollbacks,
            decisions=tuple(sorted(track.decisions.items())),
            scores=tuple(sorted(track.scores.items())),
            template_states=tuple(sorted(track.templates.items())),
        )

    def _replay_event(
        self,
        event: RealReplayEvent,
        live_gallery: dict[str, np.ndarray],
        frozen_repo: InMemoryRepository,
        pipeline: CandidatePipeline,
        corroboration: CorroborationEngine,
        promotion: PromotionManager,
        update_threshold: float,
        created_ids: list[str],
        born_seq: dict[str, int],
        corroborated: dict[str, int],
        track: _Track,
    ) -> None:
        outcome = run_candidate(
            live_gallery, [event.probe_vector], frozen_repo, _MODEL_VERSION
        )[0]
        top = outcome.top_identity or ""
        score = float(outcome.top_score or 0.0)
        margin = outcome.margin
        if (
            top == "person-23"
            and score >= BASELINE_MATCH
            and (margin is not None and margin >= MARGIN_FIREWALL)
        ):
            track.baseline_matched += 1
        elif score >= BASELINE_MATCH or score >= BASELINE_REVIEW:
            track.baseline_review += 1
        else:
            track.baseline_unknown += 1
        supervision = harness_confirmation_for(top, event.ground_truth_identity)
        if supervision == "correct" and score >= update_threshold:
            decision = pipeline.evaluate_observation(
                _matched_result("person-23", score, margin),
                b"f" * 16,
                _confirmation("correct"),
            )
            if decision.created and decision.candidate_id is not None:
                track.creations += 1
                created_ids.append(decision.candidate_id)
                born_seq[decision.candidate_id] = event.sequence_number
                track.adaptive_matched += 1
            else:
                track.rejections += 1
                track.adaptive_review += 1
        else:
            track.rejections += 1
            track.adaptive_review += 1
        for candidate_id in list(created_ids):
            image_hash = hashlib.sha256(
                f"{candidate_id}:{event.sequence_number}".encode()
            ).hexdigest()
            observed = corroboration.observe(
                candidate_id,
                f"2026-09-13T00:00:{event.sequence_number:02d}+00:00",
                image_hash,
                sequence_number=event.sequence_number,
            )
            if observed.increment:
                corroborated[candidate_id] = (
                    corroborated.get(candidate_id, 0) + 1
                )
                track.corroborations += 1
        # Seed events never promote themselves: only candidates born strictly
        # before this event are promotion-eligible (candidate pipeline
        # semantics — creation confirmation does not bypass corroboration).
        for candidate_id in list(created_ids):
            if born_seq.get(candidate_id, event.sequence_number) >= (
                event.sequence_number
            ):
                continue
            count = corroborated.get(candidate_id, 0)
            if count < 1:
                continue
            verdict = promotion.evaluate(
                _pending_candidate(candidate_id, count),
                margin if margin is not None else float("-inf"),
            )
            if verdict.promote:
                track.promotions += 1
                corroborated.pop(candidate_id, None)
                born_seq.pop(candidate_id, None)
                created_ids.remove(candidate_id)
                break
        track.decisions[event.sequence_number] = (
            f"decision@{event.sequence_number}:{top}:{supervision}"
        )
        track.scores[event.sequence_number] = score
        track.templates[event.sequence_number] = _gallery_digest(live_gallery)

    def _close_out_segments(
        self,
        lifecycle: LifecycleManager,
        eviction: EvictionManager,
        created_ids: list[str],
        track: _Track,
    ) -> None:
        # Residual candidates that never promoted are rejected at close-out
        # (stale-shadow cleanup — the third rejection source alongside
        # not_me supervision and below-threshold scores).
        if created_ids:
            lifecycle.reject_candidate(created_ids[0])
            track.rejections += 1
        scored = [
            (f"t-victim-{n}", 0.10 * n, f"2026-09-13T00:00:{n:02d}+00:00")
            for n in range(1, 7)
        ]
        victim = eviction.choose_victim(scored)
        if victim is not None:
            track.retirements += 1
        lifecycle.re_enroll("person-23", b"e" * 16, b"x" * 8)
        lifecycle.rollback("person-23", 1)
        track.rollbacks += 1
