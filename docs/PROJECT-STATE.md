# Project State

Last updated: 2026-09-12 Asia/Taipei

## Current Status

**Operator selection: A** (decision d-20260911181002229319-23) — SFace Pair 1 (YuNet 2023mar + SFace 2021dec fp32) stays **provisional**; frozen 0.90 detector gate untouched; occlusion handled via capture-condition guidance + Phase-1B confirmation-gated shadow bank. The **selection gate stays OPEN** (provisional): it closes only after Phase-1B chronological replay validates the selected point against the frozen Phase-1A baseline.

The Phase-1B implementation plan has been drafted (`docs/plans/2026-09-12-phase-1b-implementation-plan.md`) per arbitration `d-20260911184146033747-27` and is pending operator review. Implementation remains locked pending the ADR 0006 go/no-go decision.

## Current Status

Portable Face Core has **completed Phase-1A implementation**. All Phase-1A code, contracts, in-memory repository, pipeline components (decoding, orientation, quality gating, single-face detection, deterministic alignment, ONNX embedding), identification policy, evaluation session, reference CLI (`facecore.sh init`, `evaluate`, `bakeoff`, `conformance`), synthetic 500-vector capacity benchmark, and deterministic Layer-A conformance check have been delivered, verified test-first across all tasks, and validated under strict typing (mypy), linting (ruff), and dual-runner CI (macos-14 and ubuntu-latest).

The Phase-1A model candidate bake-off evaluation was executed against the consented gallery (P1 corpus, 5 enrolled identities, 13 target probes, 4 non-target probes; 辨識組 renamed to `enroll-23-probe-NN`, carry-over per skeleton d-20260911174907590232-15) using the SFace 2021dec fp32 embedder and YuNet detector:
- **Enrollment**: 5 of 5 identities successfully enrolled with single accepted photos (`enrollment refused: 0`).
- **Target probe behavior**: 5 of 13 target probes yielded zero single faces at the frozen 0.90 detector confidence gate and were honestly refused per design (`target probes usable: 8/13`).
- **Non-target false acceptance**: Scored 4 of 4 probes; 0 false acceptances at the provisional 0.85/0.10 anchor (`FA 0/4`; anchor not selected, counts-only, N<30 never rates).
- **Model selection gate**: With 8 usable target probes, statistical margins and cross-identity confusion remain undecidable. In strict conformance with spec §14 and Task 10 non-extrapolation rules, the **model selection gate remains OPEN (provisional, operator selection A recorded above)**; no artificial operating point was forced.

## Upgrade Tracking (gaps carried from the no-weights skeleton)

- Pair 2 comparison (2026may detector + int8bq embedder) unrun — no weights on disk.
- Rename-after rerun (manifest-path confirmation) pending weights dual-gate (S1 clear + operator decision, d-20260911171253541089-12).
- Per-probe `predicted` for 7 usable probes pending (only probe-11 measured: enroll-02, margin 0.0687).

Phase 1A persists no biometric database, face crops, or embeddings; inference executes entirely offline. Phase-1B governance mechanics remain untouched.

## Product Direction

- Final target: Android/iOS tablet performs offline open-set 1:N face identification for up to 500 enrolled identities.
- Registration: one student, one capture action, one accepted still photo.
- Continuity: periodic trusted re-enrollment is not guaranteed. Bounded adaptive templates must handle long-term appearance change; manual re-enrollment is optional recovery.
- Recognition: the user does not claim an identity first; the system returns a known identity only when the result is sufficiently strong, otherwise `review` or `unknown`.
- Learning: Phase 1B requires an explicit `correct` confirmation before an observation can enter the shadow-candidate flow. `not_me`, cancellation, and expiry never learn. Self-confirmation is supervision, not authentication.
- Integration: future business systems consume a versioned result contract or API adapter; business attendance rules are not part of Face Core.
- Photo-library management is handled separately through a PhotoPrism evaluation.

## Approved Phase-One Boundary

Phase 1 is split into two sequential milestones:

- **Phase 1A — accuracy first:** macOS static-image CLI and in-memory evaluation harness; one-shot enrollment, exact offline ONNX comparison, versioned results, deterministic conformance fixtures, two-to-three-model bake-off, and 500-vector capacity/latency evidence. It writes no persistent biometric database.
- **Phase 1B — governance:** persistent identity CLI, confirmation-gated learning, chronological adaptive replay, bounded template revisions, encrypted storage, rollback, re-enrollment, deletion, and export/import.
- Phase 1B starts only after the operator reviews Phase-1A evidence and records a go/no-go decision.
- Phase 1A freezes result, policy, template, revision, and repository contracts in a revision-shaped form so Phase 1B does not require destructive redesign.
- Static images and exactly one usable face per enrollment/probe remain the boundary for both milestones; no video, camera, server, REST API, Android, or iOS implementation.
- Phase 1A enrolls three to five explicitly consented identities, one registration photo each, plus consented unknown/negative probes. Every input still contains exactly one usable face.
- Results remain `matched`, `review`, `unknown`, or `invalid_input`; never `authenticated`.

## Approved Security and Privacy Rules

- All inference works offline on device; network availability never changes thresholds or results.
- Full background images are not stored by Face Core.
- A limited set of face crops and embeddings may be retained encrypted.
- Normal match events contain no image.
- Keys remain separate from the database through a `KeyProvider` interface.
- Model/store integrity failures fail closed.
- Real photos, crops, embeddings, databases, attendance records, and secrets never enter Git.
- `matched` is image similarity, not secure authentication. A future authentication layer requires trusted camera capture, liveness, anti-replay, and multi-frame policy.

## Approved Portability Direction

- ONNX-first model artifacts and ONNX Runtime on macOS.
- ONNX Runtime Mobile is the intended Android/iOS runtime.
- Preprocessing, postprocessing, normalization, template encoding, and JSON schemas are explicit and versioned.
- Golden vectors must detect cross-runtime numerical or image-processing drift.
- Phase 1A commits only deterministic, non-biometric Layer-A fixtures and proves macOS self-conformance. A reproducible, privacy-reviewed real-face carrier and Android/iOS cross-runtime conformance are Phase-2 entry work.

## Review Conclusions

- The initial enrollment template remains replaceable under the same bounded utility policy as later templates; it is not retained indefinitely as a hidden anchor.
- Normal operation cannot depend on a yearly trusted refresh; useful later observations must accumulate through the guarded, bounded template lifecycle.
- With no guaranteed trusted refresh, no permanent anchor, and only correlated similarity evidence, long-horizon cumulative drift remains an open risk. The Phase-1B implementation plan must propose measurable indicators and a bounded response; Phase 1B must not claim the risk is eliminated.
- Similarity-based promotion gates all depend on the same embedder. They reduce risk but do not independently prove identity or eliminate poisoning.
- Phase 1B excludes unattended automatic learning. Explicit confirmation gates candidate creation but does not bypass independent corroboration, quality, exclusion, or promotion rules.
- Synthetic 500-identity data is valid for comparison capacity and latency only, not for false-acceptance or ranking claims.
- Phase 1B must expose operator recovery for candidate rejection, identity rollback, and identity deletion.
- Functional correctness does not accept a recognition model. The bake-off must provide a threshold-sweep operating table for an operator go/no-go decision.
- The three-to-five-identity gallery provides real runner-up margins and cross-identity confusion evidence, but cannot justify 500-person false-acceptance or accuracy claims.
- Real-face golden fixtures, consent/retention workflow fields, calibrated multi-identity exclusion, and production authentication remain later-phase work unless an operator decision explicitly expands scope.

## Resolved Review Decisions

No review decision remains open. The accepted choices are: no guaranteed periodic trusted refresh, confirmation-gated learning, separate Phase-1A/1B milestones, and a three-to-five-identity consented Phase-1A gallery. The operator approved the consolidated written specification containing these choices on 2026-09-10.

## Future Phases

1. Android tablet prototype with camera, offline identification, 500-person device benchmarks, and secure storage.
2. iOS tablet prototype using the same models, schemas, and golden vectors.
3. Authentication layer with liveness, replay protection, multi-frame decisions, and fallback.
4. Product/API integration for identity sync, offline event queues, attendance rules, audit, and authorization.
5. Optional mobile capture adapter that saves a still image plus up to five seconds immediately preceding the shutter action. Video-based recognition remains a separate research decision.
6. Separate multi-face-in-one-photo search work after single-face identification is stable.

## Phase-1B Status (Conditional Closeout, PR-G)

Phase-1B Governance Engine delivered (provisional): confirmation-gated
shadow candidates, corroboration/promotion/utility/eviction, identity
lifecycle CLI, encrypted export/import with key re-homing, generation
migration, drift policy, chronological replay harness, replay report,
and capacity benchmark — all on synthetic streams only.

Real SFace Pair-1 weights are released and the 13-probe filename time
order is ratified (chronology B); the governed replay path
(creation→corroboration→promotion→rejection→retirement→rollback) is
GREEN on synthetic streams (`src/facecore/eval/real_replay.py`, §6-3).
Commander true-photo rerun in /tmp/face-accept is pending: the model
selection gate remains **OPEN**, and this closeout stays `partial
governance validation`, NOT Phase-1B completion. Completion requires
the commander rerun plus an operator review requesting ADR 0006
closure.

## Next Session

Phase 1A is closed. Phase-1B implementation plan (`docs/plans/2026-09-12-phase-1b-implementation-plan.md`) is drafted and pending operator review. Next session entry:

1. Read `CLAUDE.md`, `docs/plans/2026-09-12-phase-1b-implementation-plan.md`, and this file.
2. Selection evidence: no-weights skeleton `docs/2026-09-12-model-selection-report-skeleton.md` (gate OPEN, provisional).
3. Operator decisions required:
   - Record the Phase-1B governance go/no-go decision (ADR 0006).
   - Resolve Operator-level premises (weights dual gate, P1 corpus chronology, exemplar margin, retention TTL).
4. Do not begin Phase-1B implementation (persistent identity CLI, confirmation-gated learning, chronological adaptive replay, encrypted storage, rollback, deletion) until the operator records the Phase-1B go/no-go decision.
