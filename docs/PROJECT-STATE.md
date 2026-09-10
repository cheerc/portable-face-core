# Project State

Last updated: 2026-09-10 Asia/Taipei

## Current Status

Portable Face Core is in **written design pending final operator approval**. Six design sections and all review decisions were approved interactively, followed by independent completeness and adversarial review. The consolidated specification is at `docs/specs/2026-09-09-portable-face-core-design.md`.

There is no implementation, installed dependency, downloaded model, enrolled identity, private photo, embedding database, API, mobile app, or attendance product in this repository.

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

No review decision remains open. The accepted choices are: no guaranteed periodic trusted refresh, confirmation-gated learning, separate Phase-1A/1B milestones, and a three-to-five-identity consented Phase-1A gallery.

## Future Phases

1. Android tablet prototype with camera, offline identification, 500-person device benchmarks, and secure storage.
2. iOS tablet prototype using the same models, schemas, and golden vectors.
3. Authentication layer with liveness, replay protection, multi-frame decisions, and fallback.
4. Product/API integration for identity sync, offline event queues, attendance rules, audit, and authorization.
5. Optional mobile capture adapter that saves a still image plus up to five seconds immediately preceding the shutter action. Video-based recognition remains a separate research decision.
6. Separate multi-face-in-one-photo search work after single-face identification is stable.

## Next Session

1. Read `AGENTS.md` and this file.
2. Repeat an exact-head spec review after the final gallery decision.
3. Ask the operator for final approval of the consolidated written spec.
4. After explicit written-spec approval, invoke the planning workflow and create a detailed Phase-1A implementation plan.
5. Do not implement during the planning step.
