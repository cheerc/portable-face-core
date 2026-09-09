# Project State

Last updated: 2026-09-09 Asia/Taipei

## Current Status

Portable Face Core is in **written design pending four operator decisions and final approval**. Six design sections were approved interactively, followed by an independent completeness and adversarial review. The consolidated specification is at `docs/specs/2026-09-09-portable-face-core-design.md`.

There is no implementation, installed dependency, downloaded model, enrolled identity, private photo, embedding database, API, mobile app, or attendance product in this repository.

## Product Direction

- Final target: Android/iOS tablet performs offline open-set 1:N face identification for up to 500 enrolled identities.
- Registration: one student, one capture action, one accepted still photo.
- Recognition: the user does not claim an identity first; the system returns a known identity only when the result is sufficiently strong, otherwise `review` or `unknown`.
- Integration: future business systems consume a versioned result contract or API adapter; business attendance rules are not part of Face Core.
- Photo-library management is handled separately through a PhotoPrism evaluation.

## Approved Phase-One Boundary

- macOS CLI reference prototype with one shell entrypoint.
- Static images only; no video, camera, server, REST API, Android, or iOS implementation.
- One real enrolled identity for the first accuracy demonstration; other consented faces are unknown/negative probes.
- Every enrollment or probe image must contain exactly one usable face.
- One-shot enrollment creates the initial template from one photo.
- Offline ONNX inference and exact search; synthetic embeddings exercise the 500-identity capacity path.
- Human-readable output plus a versioned JSON result.
- Results are `matched`, `review`, `unknown`, or `invalid_input`; never `authenticated`.
- High-confidence events enter a guarded shadow-candidate flow and may add templates after independent corroboration.
- Active templates form a bounded, diverse, versioned bank. No template is permanent, and no template is overwritten in place.
- Two to three license-compliant ONNX candidates are benchmarked before selecting one model stack.

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
- Phase 1 commits only deterministic, non-biometric Layer-A fixtures and proves macOS self-conformance. A reproducible, privacy-reviewed real-face carrier and Android/iOS cross-runtime conformance are Phase-2 entry work.

## Review Conclusions

- The initial enrollment template remains replaceable under the same bounded utility policy as later templates; it is not retained indefinitely as a hidden anchor.
- Similarity-based promotion gates all depend on the same embedder. They reduce risk but do not independently prove identity or eliminate poisoning.
- Synthetic 500-identity data is valid for comparison capacity and latency only, not for false-acceptance or ranking claims.
- Phase 1 must expose operator recovery for candidate rejection, identity rollback, and identity deletion.
- Functional correctness does not accept a recognition model. The bake-off must provide a threshold-sweep operating table for an operator go/no-go decision.
- Real-face golden fixtures, consent/retention workflow fields, calibrated multi-identity exclusion, and production authentication remain later-phase work unless an operator decision explicitly expands scope.

## Pending Operator Decisions

1. Split Phase 1 into accuracy-focused 1A and governance-focused 1B, or keep a combined milestone.
2. Whether a trusted registration refresh occurs periodically, such as once per school year.
3. Whether interactive Phase-1 identification may auto-promote templates or promotion remains evaluation-only until supported by evidence.
4. Whether Phase 1 adds a few consented enrolled identities for real runner-up evidence or remains deliberately single-identity.

## Future Phases

1. Android tablet prototype with camera, offline identification, 500-person device benchmarks, and secure storage.
2. iOS tablet prototype using the same models, schemas, and golden vectors.
3. Authentication layer with liveness, replay protection, multi-frame decisions, and fallback.
4. Product/API integration for identity sync, offline event queues, attendance rules, audit, and authorization.
5. Optional mobile capture adapter that saves a still image plus up to five seconds immediately preceding the shutter action. Video-based recognition remains a separate research decision.
6. Separate multi-face-in-one-photo search work after single-face identification is stable.

## Next Session

1. Read `AGENTS.md` and this file.
2. Obtain answers to the four pending operator decisions recorded above.
3. Apply those decisions and repeat an exact-head spec review.
4. Ask the operator for final approval of the consolidated written spec.
5. After explicit written-spec approval, invoke the planning workflow and create a detailed Phase-one implementation plan.
6. Do not implement during the planning step.
