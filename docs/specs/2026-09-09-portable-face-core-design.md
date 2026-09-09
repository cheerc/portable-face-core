# Portable Face Core Design

- Status: section-approved; consolidated document pending final operator review
- Date: 2026-09-09
- Scope: macOS reference implementation for a future offline Android/iOS open-set face-identification core

## 1. Purpose

Portable Face Core provides local face detection, alignment, embedding, and open-set 1:N identification for applications such as tablet check-in. The final product target is an Android or iOS tablet with no more than about 500 enrolled identities. It must identify locally without network access and return associated identity data only when confidence and quality are sufficient.

Phase 1 is a macOS CLI reference prototype using static images. It proves one-shot enrollment, single-face identification, structured results, guarded adaptive updates, model selection, storage behavior, and failure handling. It does not claim secure authentication because it has no trusted camera, liveness, or replay protection.

## 2. Product Requirements

- Registration is one capture action and one accepted still photo per person.
- The user does not claim an identity first; the system performs open-set 1:N identification.
- False acceptance is more serious than false rejection. Uncertain inputs return `review` or `unknown`, not a guessed name.
- A successful result can include opaque identity ID, display name, and application metadata.
- Repeated, trustworthy observations add useful templates over time instead of overwriting the previous face.
- Children aging, hairstyle, eyewear, hats, pose, lighting, and background variation are explicit evaluation conditions.
- Masked or seriously occluded faces may be rejected; mask recognition is not a Phase-one success requirement.
- All inference must work on-device and offline.
- Final capacity is no more than about 500 identities.
- Code, libraries, and distributed model artifacts must permit commercial use and redistribution.
- Real biometric data stays out of Git and is not sent to cloud face APIs or telemetry.

## 3. Phase-One Scope

Phase 1 includes:

- one shell entrypoint wrapping a macOS CLI;
- static-image input only;
- one real enrolled identity for the initial demonstration;
- consented non-target faces as unknown/negative probes;
- exactly one usable face in every enrollment and probe image;
- one-shot enrollment from one image;
- ONNX Runtime inference;
- exact comparison, a versioned JSON result, and a human-readable summary;
- shadow candidates, guarded promotion, bounded active templates, retirement, and rollback;
- two to three compliant model candidates evaluated under the same protocol;
- synthetic embeddings for the 500-identity capacity path;
- encrypted exemplar/template storage through a replaceable repository interface.

Phase 1 excludes:

- video ingestion or frame extraction;
- camera capture, Android, and iOS applications;
- multiple faces in a single request image;
- REST/server deployment and business-system integration;
- liveness, replay protection, and an `authenticated` decision;
- attendance rules, authorization, payroll, access control, or surveillance;
- PhotoPrism, photo-library management, Google Photos, or permanent classification copies;
- foundation-model training.

## 4. Terminology and Decision States

- **Enrollment:** create the initial identity template from one accepted registration photo.
- **Probe:** the face presented for identification after enrollment.
- **Embedding/template:** a normalized vector representing similarity-relevant face features; it is sensitive biometric data.
- **Open-set identification:** find a known identity or decide that the probe is unknown.
- **Shadow candidate:** a high-confidence observation not yet trusted as an active identity template.
- **Active template:** a template participating in identity scoring.
- **Retired template:** a non-matching template retained temporarily for rollback.

Phase-one states are:

- `matched`: strong similarity result under the current model and policy;
- `review`: plausible but insufficient for automatic acceptance;
- `unknown`: no trustworthy known identity;
- `invalid_input`: the request cannot be evaluated, such as zero faces, multiple faces, unreadable input, or inadequate quality.

`matched` never means `authenticated` in Phase 1.

## 5. Architecture

```text
CLI / future mobile adapter
  → input orientation and decoding
  → quality gate and exactly-one-face detector
  → landmark alignment and normalized crop
  → ONNX embedding inference
  → exact active-template comparison
  → identity aggregation and open-set policy
  → matched / review / unknown / invalid_input
  → event record and optional shadow candidate
```

Five logical components have clear boundaries:

1. **Adapter** reads files in Phase 1 and later camera frames on mobile. It is not part of the recognition policy.
2. **Face Pipeline** validates quality, detects exactly one face, aligns it, and produces a versioned normalized embedding.
3. **Identity Repository** stores identities, model generations, active/retired templates, candidates, encrypted exemplars, revisions, and match events.
4. **Identification Policy** computes identity-level scores and decision states.
5. **Adaptive Update Policy** creates, corroborates, promotes, retires, and rolls back templates without in-place overwrite.

The core accepts decoded image data plus caller identifiers. Filesystem traversal, UI, HTTP, camera capture, and attendance events remain outside the core.

## 6. ONNX-First Portability Contract

ONNX is the primary model artifact format. Phase 1 uses Python with ONNX Runtime; Android and iOS are intended to use ONNX Runtime Mobile.

The portable contract versions all of the following:

- image orientation and color order;
- face box and landmark coordinates;
- resize, crop, alignment, padding, and interpolation;
- tensor layout, type, scale, mean, and standard deviation;
- embedding normalization and numeric encoding;
- identity aggregation and score direction;
- model and preprocessing manifest;
- template serialization and result JSON.

Golden input/output fixtures must detect drift between macOS and later Android/iOS implementations. Platform accelerators such as NNAPI or Core ML are optional optimizations; CPU-correctness is the baseline.

## 7. One-Shot Enrollment

The real registration flow is:

```text
identity data + one photo
  → decode and orient
  → require exactly one face
  → require minimum quality
  → align and embed
  → encrypt exemplar and template
  → create identity revision 1
```

No person is asked to submit many headshots or perform a long scan. A single accepted photo is sufficient to create the identity, but the result is labeled a one-shot initial state rather than proof of production-grade robustness.

Zero-face, multi-face, unreadable, low-quality, or incompatible samples fail with reason codes and do not create a partial identity.

## 8. Identification Policy

For each probe, the system compares the normalized embedding with all eligible active templates. At the intended capacity, the default is exact comparison rather than an approximate vector index.

The policy considers:

- the best and robust aggregate scores for each identity;
- support across more than one active template when available;
- the absolute top identity score;
- the top-1/top-2 margin when multiple identities exist;
- detector confidence and face quality;
- model, preprocessing, template, and policy versions.

The policy must be calibrated at the identity level because adding templates increases the opportunity for coincidental high scores. Ranking first is never sufficient by itself. With one enrolled identity, the runner-up is absent, so non-target probes are essential to calibrate the unknown boundary.

## 9. Adaptive Template Bank

Recognition success never overwrites an existing template. A stricter update threshold than the normal `matched` threshold gates shadow-candidate creation.

A candidate must be:

- high quality and above the strict update threshold;
- consistent with the identity, not merely one weak template;
- non-duplicative and useful for appearance, age, accessory, pose, or lighting coverage;
- corroborated by independent later events before promotion;
- rejected from automatic promotion when severely occluded or otherwise outside the validated policy.

For an identity that still has only its one-shot enrollment template, the first later event may create a shadow candidate but cannot promote itself. At least one additional, temporally independent event must strongly match both the enrollment identity and the candidate cluster before the first promotion. Repeated processing of the same file, burst, or event never counts as independent corroboration.

Active templates are cumulative but bounded. No template, including the initial enrollment template, is permanent. When the bank reaches capacity, all templates are rescored by utility:

```text
utility =
  quality
  + independent-event support
  + recency
  + appearance and pose coverage
  - redundancy
  - mismatch or outlier risk
```

The lowest-utility template retires only after the new candidate passes promotion. Distance from a centroid alone cannot drive eviction because a useful profile or eyewear sample may be intentionally different. Every promotion and retirement creates a new atomic revision. Retired templates do not participate in matching, remain available for a limited rollback policy, and are deleted after retention expiry.

Chronological evaluation must prevent future probes from leaking into earlier identity state.

## 10. Data Model and Storage

- `Identity`: opaque ID, display name, optional application metadata, lifecycle state.
- `ModelManifest`: exact artifact hashes, licenses, provenance status, input/output contract, embedding dimensions.
- `TemplateGeneration`: identity, model/preprocessing generation, compatibility state.
- `FaceTemplate`: active or retired normalized embedding, quality, support, utility inputs, source reference.
- `CandidateTemplate`: encrypted candidate embedding/crop, evidence, expiry, decision state.
- `EncryptedExemplar`: bounded face crop needed for future re-embedding; never a full background image.
- `PolicyProfile`: quality, match, review, unknown, update, aggregation, retention, and capacity settings.
- `MatchEvent`: request, result, scores, versions, timestamp, and candidate side effect; no image.
- `TemplateRevision`: atomic membership and policy history used for audit and rollback.

Phase 1 may use SQLite, but core services depend on repository and `KeyProvider` interfaces. Encryption keys remain outside the database. Later mobile adapters can use platform-secure storage without changing core semantics.

Names are application metadata, not identity keys. Export/import uses a versioned encrypted format and rejects incompatible model generations.

## 11. CLI and Result Contract

The Phase-one shell entrypoint exposes explicit subcommands equivalent to:

```text
./facecore.sh init
./facecore.sh identity add --id person-001 --name "Test Person" --image enroll.jpg
./facecore.sh identify --image probe.jpg
./facecore.sh identity show --id person-001
./facecore.sh candidates list
./facecore.sh status
```

Commands provide a human-readable summary and stable JSON. A successful match resembles:

```json
{
  "schema_version": 1,
  "status": "matched",
  "identity": {
    "id": "person-001",
    "display_name": "Test Person",
    "metadata": {}
  },
  "score": 0.82,
  "runner_up_score": null,
  "decision": {
    "threshold": 0.76,
    "margin": null
  },
  "quality": {
    "status": "accepted",
    "reason_codes": []
  },
  "model_version": "model-id",
  "template_revision": 3,
  "candidate_created": true
}
```

The numeric score and threshold above are illustrative schema values, not selected model defaults. The model bake-off and calibration evidence determine deployable values.

`review`, `unknown`, and `invalid_input` never fill a guessed display name. Normal recognition outcomes are not process failures; unreadable input, model integrity, store integrity, invalid configuration, or internal failures use non-zero exit codes.

No REST API is delivered in Phase 1. Future API or mobile layers wrap the same domain/result contract.

## 12. Privacy, Security, and Failure Handling

- Full probe/background images are not retained by Face Core.
- A bounded set of approved and candidate face crops and embeddings is encrypted at rest.
- Normal logs exclude raw images, face crops, embeddings, full local paths, and personal data.
- Identity deletion immediately removes active, candidate, retired, exemplar, and identity-link biometric data; a non-biometric audit tombstone may remain only if an application policy requires it.
- Network availability cannot weaken thresholds or select an unsafe fallback.
- Model checksum, model/preprocessing incompatibility, corrupt storage, unavailable keys, or incomplete migrations fail closed.
- Template promotions, retirements, and event side effects are atomic and idempotent.
- Interrupted work must not expose half-created identities or half-applied revisions.
- Model changes create a new template generation and require re-embedding retained exemplars. Old and new generations never compare as if compatible.

Phase 1 is vulnerable to a printed or displayed photo because it has no liveness or replay protection. Its output is deliberately `matched`, not `authenticated`.

## 13. Model Bake-Off

Two to three candidates must use the same one-shot enrollment photo, probe order, policy interface, and output measurements. A candidate first passes hard gates:

- exact code and weight licenses allow commercial use and redistribution;
- artifact source, version, checksum, and known training-data provenance are recorded;
- inference runs under ONNX Runtime on macOS;
- ONNX Runtime Mobile checks record operator, shape, fallback, and accelerator findings;
- preprocessing and output contracts are reproducible and versioned.

Selection priorities are:

1. lowest observed non-target false acceptance;
2. highest direct `matched` rate for ordinary unoccluded target probes;
3. useful `review/unknown` behavior rather than unsafe guessing;
4. separately reported performance for eyewear, hats, profiles, complex backgrounds, age change, masks, blur, and occlusion;
5. macOS latency, memory, and artifact size;
6. Android/iOS feasibility and later real-device performance.

If fewer than two candidates satisfy licensing and provenance gates, the project reports the shortage rather than weakening those gates.

## 14. Evaluation and Acceptance

Evaluation data has three isolated roles:

- one registration photo used for enrollment;
- later target probes that never participate in initial enrollment;
- consented non-target probes used to test the unknown boundary.

The production registration UX still uses one photo. Additional target probes are development evidence that simulate later encounters, not required user submissions.

Evaluation runs twice:

1. **Frozen one-shot:** only the initial registration template is active.
2. **Chronological adaptive replay:** later events are processed in time order with shadow candidates enabled.

The report compares target match/review/unknown counts, non-target false acceptance, candidate creation/promotion, template churn, and drift. It always reports exact denominators; zero observed failures in a small corpus cannot be described as a zero real-world error rate.

Phase-one functional acceptance requires:

- one accepted photo creates a complete identity;
- invalid enrollment leaves no partial identity;
- target and non-target probes produce deterministic versioned results;
- uncertain probes never become successful business events;
- adaptive updates append revisions instead of overwriting templates;
- initial templates can retire under the same utility policy as later templates;
- rejected or ambiguous events never enter active templates;
- database/model/key corruption fails closed;
- real biometric files remain untracked;
- synthetic 500-identity comparison is measured and is not the dominant end-to-end latency;
- the winning model and policy are selected with documented evidence, limitations, and no production-authentication claim.

Test categories include schema/reason codes, zero/one/multiple-face behavior, preprocessing and embedding normalization, model-generation incompatibility, score/margin boundaries, candidate corroboration, utility eviction, atomic revision rollback, encrypted export/import, deletion, no-network inference, log redaction, and failure injection.

## 15. Future Roadmap

1. **Android tablet prototype:** camera capture, offline 1:N identification, secure storage, and real 500-person device benchmarks.
2. **iOS tablet prototype:** same ONNX artifacts, schemas, and golden vectors with iOS performance evidence.
3. **Authentication layer:** trusted capture, liveness, anti-replay, multi-frame aggregation, and fallback methods before `authenticated` can exist.
4. **Product integration:** identity/policy synchronization, offline event queues, APIs, attendance rules, authorization, audit, correction, and retention.
5. **Pre-capture video adapter:** a later mobile capture feature may save one still plus up to five seconds immediately preceding the shutter action. Video recognition or template updates remain out of scope until separately researched.
6. **Multi-face photo search:** an independent branch of work only after single-face identification is accurate and stable.

## 16. Design Completion Gate

After the operator approves this consolidated written specification, the next artifact is a detailed Phase-one implementation plan. Implementation must not begin until that plan is reviewed. The plan must choose concrete policy defaults, local data paths, model-candidate discovery tasks, test corpus inventory, encryption/key-provider mechanics, and verification commands without expanding Phase-one scope.
