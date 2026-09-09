# Project State

Last updated: 2026-09-09 Asia/Taipei

## Current Status

Portable Face Core is in **written design pending final operator review**. Six design sections were approved interactively. The consolidated specification is at `docs/specs/2026-09-09-portable-face-core-design.md`.

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

## Future Phases

1. Android tablet prototype with camera, offline identification, 500-person device benchmarks, and secure storage.
2. iOS tablet prototype using the same models, schemas, and golden vectors.
3. Authentication layer with liveness, replay protection, multi-frame decisions, and fallback.
4. Product/API integration for identity sync, offline event queues, attendance rules, audit, and authorization.
5. Optional mobile capture adapter that saves a still image plus up to five seconds immediately preceding the shutter action. Video-based recognition remains a separate research decision.
6. Separate multi-face-in-one-photo search work after single-face identification is stable.

## Next Session

1. Read `AGENTS.md` and this file.
2. Ask the operator for final review of the consolidated written spec.
3. Apply requested corrections and repeat spec self-review.
4. After explicit written-spec approval, invoke the planning workflow and create a detailed Phase-one implementation plan.
5. Do not implement during the planning step.
