# Portable Face Core Design

- Status: Phase-1 baseline approved on 2026-09-10; current implementation/evidence status is maintained in `../PROJECT-STATE.md`. Phase-2A research amendment: ADR 0008 and `2026-09-14-mac-live-identification-research-design.md` (docs review; implementation requires a separate plan and go).
- Date: 2026-09-09
- Scope: macOS reference implementation for a future offline Android/iOS open-set face-identification core

## 1. Purpose

Portable Face Core provides local face detection, alignment, embedding, and open-set 1:N identification for applications such as tablet check-in. The final product target is an Android or iOS tablet with no more than about 500 enrolled identities. It must identify locally without network access and return associated identity data only when confidence and quality are sufficient.

Phase 1 is split into two sequential macOS CLI milestones using static images. Phase 1A proves one-shot enrollment, single-face identification, model selection, deterministic structured results, and accuracy evidence without durable biometric storage. Phase 1B adds confirmation-gated adaptive updates, encrypted persistence, rollback, deletion, and governance using the contracts frozen in 1A. Neither milestone claims secure authentication because there is no trusted camera, liveness, or replay protection.

## 2. Product Requirements

- Registration is one capture action and one accepted still photo per person.
- The user does not claim an identity first; the system performs open-set 1:N identification.
- False acceptance is more serious than false rejection. Uncertain inputs return `review` or `unknown`, not a guessed name.
- A successful result can include opaque identity ID, display name, and application metadata.
- Repeated, trustworthy observations add useful templates over time instead of overwriting the previous face.
- Phase-1 learning is confirmation-gated: an identification result cannot create a shadow candidate until a person explicitly marks the proposed identity as correct. Unattended automatic learning is excluded.
- Normal continuity cannot depend on a periodic trusted re-enrollment event. A fresh trusted registration may be offered as manual recovery, but long-term appearance change must be handled by the bounded adaptive-template design.
- Children aging, hairstyle, eyewear, hats, pose, lighting, and background variation are explicit evaluation conditions.
- Masked or seriously occluded faces may be rejected; mask recognition is not a Phase-one success requirement.
- All inference must work on-device and offline.
- Final capacity is no more than about 500 identities.
- Code, libraries, and distributed model artifacts must permit commercial use and redistribution.
- Real biometric data stays out of Git and is not sent to cloud face APIs or telemetry.

## 3. Phase-One Scope

Phase 1 is delivered as two gated milestones. Phase 1B does not begin until the operator reviews the Phase-1A evidence and records a go/no-go decision.

Phase 1A includes:

- one shell entrypoint wrapping a macOS CLI and evaluation harness;
- static-image input only;
- three to five explicitly consented enrolled identities for the initial evaluation gallery;
- consented non-target faces as unknown/negative probes;
- exactly one usable face in every enrollment and probe image;
- one-shot enrollment from one image into an in-memory evaluation session;
- ONNX Runtime inference and exact comparison;
- a versioned JSON result and human-readable summary;
- two to three compliant model candidates evaluated under the same frozen one-shot protocol;
- synthetic embeddings for comparison-capacity and latency measurement at 500 identities;
- deterministic Layer-A fixtures and macOS self-conformance;
- versioned result, policy, template, revision, and repository contracts shaped for Phase-1B persistence.

Phase 1A writes aggregate evaluation reports and non-biometric result records only. It does not create a durable identity database, persist face crops or embeddings, run live adaptive promotion, or implement encrypted export/import.

Phase 1B includes:

- the selected Phase-1A model and operating point;
- persistent identity enrollment and identification through the same versioned contracts;
- confirmation-gated shadow candidates and chronological adaptive replay;
- guarded promotion, bounded active templates, retirement, and rollback;
- encrypted exemplar/template storage through a replaceable repository and `KeyProvider`;
- identity deletion, candidate rejection, trusted re-enrollment, failure recovery, and encrypted export/import.

Both Phase-1 milestones exclude:

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
- **Learning confirmation:** an explicit label that a proposed identity is correct or incorrect. Self-confirmation is useful supervision but is not secure identity proof; operator/staff confirmation carries a distinct actor type.
- **Active template:** a template participating in identity scoring.
- **Retired template:** a non-matching template retained temporarily for rollback.

Phase-one states are:

- `matched`: strong similarity result under the current model and policy;
- `review`: plausible but insufficient for automatic acceptance;
- `unknown`: no trustworthy known identity;
- `invalid_input`: a decoded request was evaluated but cannot produce a recognition decision, such as zero faces, multiple faces, or inadequate quality.

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
  → event record and confirmation opportunity
  → confirmed observation may become a shadow candidate
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
- numerical precision, quantization format, ONNX/ORT artifact format, and permitted execution-provider differences;
- identity aggregation and score direction;
- model and preprocessing manifest;
- template serialization and result JSON.

Versioning these stages is necessary but not sufficient: a stage can be versioned and still be silently wrong for a subset of inputs. The preprocessing chain therefore carries one explicit invariant.

**Source aspect ratio is preserved end to end.** Any resize onto a fixed network input uses a single uniform scale plus padding; face box and landmark coordinates are restored through that same uniform scale and the same padding offset, never through independent per-axis factors. The aligned crop is derived from the landmark constellation, not from a raw detection box resized anisotropically. Consequently the embedding produced for one face must be invariant, within a stated numeric tolerance, to the aspect ratio and letterboxing of the frame that carried it.

Layer-A conformance covers this directly: fixtures include non-square inputs whose restored geometry must match the square-input reference within tolerance. These fixtures stay face-free, so the check adds no privacy surface.

This invariant is stated explicitly because its absence is not hypothetical. A fixed-square detector resize with per-axis coordinate restoration satisfied every versioned-stage requirement above while distorting every non-square input, and the resulting embedding displacement was large enough to collapse open-set margins to near zero.

Phase 1A commits a deterministic fixture generator rather than biometric images. At test time it generates face-free gradients, checkerboards, chroma patterns, EXIF-orientation cases, and synthetic tensors. Committed expected JSON values cover decoding, orientation, color order, resize, crop, padding, interpolation, tensor layout, scaling, normalization, embedding normalization, and numeric encoding. A macOS self-conformance command must reproduce those values.

These Layer-A fixtures do not validate detector or landmark equivalence on real faces. A reproducible and privacy-reviewed real-face conformance carrier remains a Phase-2 entry problem; it must not become an indefinite out-of-band biometric fixture by default. Android/iOS must pass cross-runtime conformance before mobile evaluation begins. Platform accelerators such as NNAPI or Core ML are optional optimizations; CPU correctness is the baseline.

## 7. One-Shot Enrollment

The shared one-shot transformation is:

```text
identity data + one photo
  → decode and orient
  → require exactly one face
  → require minimum quality
  → align and embed
```

Phase 1A places the resulting template in an in-memory, revision-shaped evaluation identity and discards it when the process exits. Phase 1B encrypts the exemplar and template, then atomically creates persistent identity revision 1.

No person is asked to submit many headshots or perform a long scan. A single accepted photo is sufficient to create the identity, but the result is labeled a one-shot initial state rather than proof of production-grade robustness.

Quality policy is versioned and must declare at least sharpness, usable face-pixel size, yaw/pitch bounds, exposure or illumination, occlusion, and detector confidence. The implementation plan selects measurable definitions and defaults rather than treating “high quality” as an undocumented model judgment.

Zero-face, multi-face, low-quality, or incompatible decoded samples return `invalid_input` with reason codes and do not create a partial identity. Unreadable or undecodable input is a process/input failure and likewise never creates a partial identity.

**Capture shape.** Where an enrollment or identification image is produced by a camera under this system's own UI, that UI presents a square alignment guide and yields a square image, so face geometry does not depend on whether the device was held in portrait or landscape.

This is an ergonomic and consistency measure for images this system captures. It is **not** a precondition for correctness. Images from any other source — existing corpora, replay, import, future mobile adapters — carry arbitrary aspect ratios, and the §6 aspect-invariance requirement holds for all of them. A capture-side convention must never be relied upon to keep the numerical path correct.

## 8. Identification Policy

For each probe, the system compares the normalized embedding with all eligible active templates. At the intended capacity, the default is exact comparison rather than an approximate vector index.

The policy considers:

- the best and robust aggregate scores for each identity;
- support across more than one active template when available;
- the absolute top identity score;
- the top-1/top-2 margin when multiple identities exist;
- detector confidence and face quality;
- model, preprocessing, template, and policy versions.

The implementation plan must choose and justify a robust identity aggregation form, such as a median, trimmed mean, or support/second-best rule. A single maximum similarity is insufficient once an identity has multiple templates.

The policy must be calibrated at the identity level because adding identities or templates increases the opportunity for coincidental high scores. Ranking first is never sufficient by itself. Phase 1A enrolls three to five identities so every probe has a real runner-up score and margin; consented non-target probes remain essential to calibrate the unknown boundary.

Open-set false acceptance grows with gallery size. Thresholds calibrated in Phase 1A with three to five enrolled identities are provisional and cannot be extrapolated to 500 identities. Before multi-identity or target-capacity deployment, thresholds and margins must be recalibrated against a representative real gallery; this is a Phase-2 entry gate.

## 9. Adaptive Template Bank

Recognition success never overwrites an existing template and never learns by itself. A matched result first returns a confirmation opportunity. Only an explicit correct confirmation may make the observation eligible for shadow-candidate creation, after which a stricter update threshold than the normal `matched` threshold still applies.

A `not_me` confirmation never creates or promotes a template and records a policy-error event without retaining the probe image. No response, cancellation, or confirmation expiry has the same no-learning outcome. The adapter may keep the observation only in memory while confirmation is pending; Face Core does not persist a pending face crop or embedding before confirmation.

A candidate must be:

- high quality and above the strict update threshold;
- consistent with the identity, not merely one weak template;
- non-duplicative and useful for appearance, age, accessory, pose, or lighting coverage;
- corroborated by independent later events before promotion;
- rejected from automatic promotion when severely occluded or otherwise outside the validated policy.

For a multi-identity gallery, promotion additionally requires identity exclusivity: a candidate must score better for its owning identity than for every other enrolled identity by a separately calibrated promotion margin. Phase 1A records cross-identity score and margin evidence but has no live promotion path. Phase-1B chronological replay exercises this obligation with the selected model; the small gallery remains insufficient to claim target-capacity validation.

For an identity that currently has only one active template, whether from initial enrollment or trusted re-enrollment, the first later event may create a shadow candidate but cannot promote itself. At least one additional, temporally independent event must strongly match both the identity and the candidate cluster before the first promotion. Repeated processing of the same file, burst, or event never counts as independent corroboration.

All similarity-based promotion checks use the same embedding model and therefore are correlated evidence, not independent proof of identity. They can reduce poisoning risk but cannot make self-learning safe against a confident, systematic misidentification. Liveness would establish that a live person is present, not which identity that person has. Phase 1B therefore excludes unattended automatic learning: an explicit confirmation is required before candidate creation, and confirmation does not bypass corroboration or promotion gates.

Self-confirmation contributes a human-provided label but does not establish who pressed the button, so it cannot turn `matched` into `authenticated` or independently authorize attendance. Operator/staff confirmation is recorded separately and may be assigned greater policy weight, but attendance authorization remains outside Face Core.

Because periodic trusted re-enrollment is not guaranteed, the design must preserve useful appearance changes over time without making the original enrollment template permanent. An operator may still perform a fresh trusted registration as an explicit recovery action; that optional path does not replace guarded accumulation during normal operation.

The combination of no guaranteed trusted refresh, no permanent anchor, and correlated similarity evidence leaves long-horizon cumulative drift as an open risk. The Phase-1B implementation plan must propose measurable drift indicators and a bounded response, then evaluate them in chronological replay. Phase 1B must report the remaining limitation rather than claim that promotion thresholds eliminate it.

Active templates are cumulative but bounded. No template, including the initial enrollment template, is permanent. When the bank reaches capacity, all templates are rescored using the following utility factors:

```text
quality
independent-event support
recency
appearance and pose coverage
redundancy
mismatch or outlier risk
```

The implementation plan must define how each factor is normalized, its weight or ordering, and deterministic tie-breaking. Every revision records the component values used for its decision. The lowest-utility template retires only after the new candidate passes promotion. Distance from a centroid alone cannot drive eviction because a useful profile or eyewear sample may be intentionally different. Every promotion and retirement creates a new atomic revision. Retired templates do not participate in matching, remain available for a limited rollback policy, and are deleted after retention expiry.

Chronological evaluation must prevent future probes from leaking into earlier identity state.

## 10. Data Model and Storage

- `Identity`: opaque ID, display name, optional application metadata, lifecycle state.
- `ModelManifest`: exact artifact hashes, licenses, provenance status, input/output contract, embedding dimensions, numerical precision, quantization, and ONNX/ORT format.
- `TemplateGeneration`: identity, model/preprocessing generation, compatibility state.
- `FaceTemplate`: active or retired normalized embedding, quality, support, utility inputs, source reference.
- `CandidateTemplate`: encrypted candidate embedding/crop, evidence, expiry, decision state.
- `EncryptedExemplar`: a bounded pre-alignment face region plus a versioned `exemplar_margin`, clipped to the source image; never the full frame or unrestricted background. Phase 1B may select a minimal margin rather than retaining extra pixels for a migration it does not exercise.
- `PolicyProfile`: quality, match, review, unknown, update, aggregation, retention, and capacity settings.
- `MatchEvent`: request, result, scores, versions, timestamp, and candidate side effect; no image.
- `TemplateRevision`: atomic membership and policy history used for audit and rollback.

Phase 1A uses an in-memory repository implementation and persists no biometric database, while freezing revision-shaped repository and data contracts. Phase 1B may use SQLite, but core services depend on repository and `KeyProvider` interfaces. Encryption keys remain outside the database. Later mobile adapters can use platform-secure storage without changing core semantics.

Names are application metadata, not identity keys. Export/import uses a versioned encrypted format and rejects incompatible model generations.

## 11. CLI and Result Contract

The Phase-1A shell entrypoint exposes the stateless evaluation flow:

```text
./facecore.sh init
./facecore.sh evaluate --enrollment enroll.jpg --probe probe.jpg
./facecore.sh bakeoff --corpus <manifest>
./facecore.sh conformance
```

Enrollment and probes remain in one process-local evaluation session; Phase 1A writes only aggregate reports and non-biometric result JSON.

Phase 1B adds persistent and adaptive subcommands equivalent to:

```text
./facecore.sh identity add --id person-001 --name "Test Person" --image enroll.jpg
./facecore.sh identify --image probe.jpg
./facecore.sh identify --image probe.jpg --confirm-learning
./facecore.sh identity show --id person-001
./facecore.sh identity re-enroll --id person-001 --image trusted-refresh.jpg
./facecore.sh candidates list
./facecore.sh candidates reject --id <candidate-id> --reason <code>
./facecore.sh identity rollback --id <identity-id> --to-revision <n>
./facecore.sh identity delete --id <identity-id>
./facecore.sh status
```

Trusted re-enrollment accepts one operator-supplied photo under the enrollment quality rules. It atomically creates a new revision with the new template as the sole active starting template and retires the previously active templates under the limited rollback retention policy; it preserves the identity ID and application metadata. It never overwrites history in place.

The `identity add` command is create-only. If the ID already exists, it fails without mutation and directs the operator to `identity re-enroll`; it never becomes an implicit overwrite or alternate refresh path.

With `--confirm-learning`, the CLI first prints the identification result and then asks `correct` or `not_me` while the decoded observation remains in process memory. It creates a shadow candidate only after `correct`; cancellation, EOF, timeout, or `not_me` exits without learning. Stable JSON records whether confirmation was requested, its actor type, and the resulting candidate side effect.

Re-enroll, reject, rollback, and delete use the same atomic revision machinery as automatic changes and return stable JSON. A manual recovery action records actor `operator` and is never treated as a training or calibration signal. A learning confirmation is recorded as supervision evidence for its candidate but never as authentication proof.

Commands provide a human-readable summary and stable JSON. Phase 1A uses the same result shape with a process-local revision; a successful Phase-1B match resembles:

```json
{
  "schema_version": 1,
  "status": "matched",
  "identity": {
    "id": "person-001",
    "display_name": "Test Person",
    "metadata": {}
  },
  "decision": {
    "score": 0.82,
    "runner_up_score": null,
    "threshold": 0.76,
    "margin": null,
    "reason_codes": ["match_threshold_met"]
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

`review`, `unknown`, and `invalid_input` never fill a guessed display name. Decision-level reason codes distinguish threshold failure, insufficient margin, quality rejection, and other policy outcomes.

Normal recognition outcomes, including decoded `invalid_input`, exit 0. Every command still emits JSON on failure, but process failures have no recognition decision state: unreadable or undecodable input exits 2, model integrity or compatibility failure exits 3, store or key failure exits 4, invalid configuration exits 5, and unexpected internal failure exits 7.

No REST API is delivered in Phase 1. Future API or mobile layers wrap the same domain/result contract.

## 12. Privacy, Security, and Failure Handling

- Full probe/background images are not retained by Face Core.
- A bounded set of approved and candidate face crops and embeddings is encrypted at rest.
- Logs at every level exclude raw images, face crops, embeddings, full local paths, and personal data. A future diagnostic export containing sensitive material requires a separate explicit, audited workflow; increasing log verbosity never enables it.
- Identity deletion immediately removes active, candidate, retired, exemplar, and identity-link biometric data. Identity-linked personal data in match events is deleted or irreversibly anonymized; a non-biometric audit tombstone may remain only if an application policy requires it.
- Network availability cannot weaken thresholds or select an unsafe fallback.
- Model checksum, model/preprocessing incompatibility, corrupt storage, unavailable keys, or incomplete migrations fail closed.
- Template promotions, retirements, and event side effects are atomic and idempotent.
- Interrupted work must not expose half-created identities or half-applied revisions.
- Model changes create a new template generation and require re-embedding retained exemplars. Old and new generations never compare as if compatible. If a detector or alignment change cannot be reproduced from the stored exemplar geometry and margin, affected identities become `re_enrollment_required`; the system never silently re-embeds them with mismatched geometry.

Phase 1 is vulnerable to a printed or displayed photo because it has no liveness or replay protection. Its output is deliberately `matched`, not `authenticated`.

## 13. Model Bake-Off

Two to three candidates must use the same one-shot enrollment gallery, probe order, policy interface, and output measurements. Every candidate receives the same single registration photo for each enrolled identity. A candidate first passes hard gates:

- exact code and weight licenses allow commercial use and redistribution;
- artifact source, version, checksum, and known training-data provenance are recorded;
- inference runs under ONNX Runtime on macOS;
- ONNX Runtime Mobile checks record operator, shape, fallback, and accelerator findings;
- preprocessing and output contracts are reproducible and versioned.

Selection priorities are:

1. lowest observed non-target false acceptance;
2. lowest cross-identity confusion and the safest top-1/top-2 margin distribution;
3. highest direct `matched` rate for ordinary unoccluded target probes;
4. useful `review/unknown` behavior rather than unsafe guessing;
5. separately reported performance for eyewear, hats, profiles, complex backgrounds, age change, masks, blur, and occlusion;
6. macOS latency, memory, and artifact size;
7. Android/iOS feasibility and later real-device performance.

If fewer than two candidates satisfy licensing and provenance gates, the project reports the shortage rather than weakening those gates. The report lists every excluded candidate, its failed gate, and supporting evidence; the operator then decides whether to pause or explicitly revise the bake-off requirement.

## 14. Evaluation and Acceptance

Evaluation data has three isolated roles:

- one registration photo for each of three to five explicitly consented enrolled identities;
- later target probes grouped by identity that never participate in initial enrollment;
- consented non-target probes used to test the unknown boundary.

The production registration UX still uses one photo per person. Additional target probes are development evidence that simulate later encounters, not required user submissions. All evaluation biometric files remain local and outside Git. The corpus owner records the applicable consent, guardian authorization when required, retention deadline, and deletion procedure outside the Face Core identity schema.

Evaluation is staged:

1. **Phase 1A frozen one-shot:** every model candidate uses only the initial registration template. This produces the model and operating-point evidence used for the Phase-1A go/no-go decision.
2. **Phase 1B chronological adaptive replay:** later events are processed in time order with confirmation-gated shadow candidates enabled for the selected model. The corpus ground-truth label supplies the explicit `correct` or `not_me` confirmation solely inside this evaluation harness; it is never a production confirmation source. Replay must exercise and measure candidate creation, corroboration, promotion, retirement, and rejection behavior against the frozen Phase-1A baseline.

The Phase-1A report compares per-identity and aggregate target match/review/unknown counts, top-1/top-2 margins, cross-identity confusion, and non-target false acceptance for every candidate model. The Phase-1B report compares the selected model's frozen baseline with confirmation-gated candidate creation/promotion, template churn, and drift. Both always report exact denominators; zero observed failures in a small corpus cannot be described as a zero real-world error rate.

The corpus inventory reports sample counts for age change, hairstyle, eyewear, hats, pose, lighting, complex background, masks, blur, and occlusion. A condition with zero samples is labeled `untested`; this requirement documents coverage and does not require collecting additional biometric data merely to fill a category. Because Phase 1 accepts exactly one usable face per image, its evaluation is biased toward posed single-person inputs and must disclose that limitation.

For every candidate model, the Phase-1A evaluation report provides a frozen one-shot threshold-sweep operating table with target `matched`/`review`/`unknown` counts and non-target false acceptances, all with exact denominators. Phase 1A sets no accuracy number before calibration data exists. Functional acceptance alone does not constitute model acceptance; the operator selects or rejects a model and operating point after reviewing this evidence. Phase 1B then measures the selected operating point under chronological adaptive replay against the frozen baseline.

Phase-1A functional acceptance requires:

- one accepted enrollment photo creates a complete process-local evaluation identity;
- invalid enrollment leaves no partial session state;
- target and non-target probes produce deterministic versioned results;
- every enrolled identity has per-identity outcomes, runner-up margins, and a confusion-matrix row in the report;
- uncertain probes never become successful business events;
- model integrity or incompatibility fails closed;
- no face crop, embedding, or identity database persists after the evaluation process exits;
- real biometric files remain untracked;
- synthetic 500-identity comparison is measured within the predeclared comparison-latency budget;
- deterministic Layer-A fixtures, committed expected JSON, and macOS self-conformance pass;
- the winning model and operating point are selected with documented evidence, limitations, and no production-authentication claim.

Phase-1B functional acceptance additionally requires:

- persistent enrollment, identification, confirmation, and recovery commands follow the frozen contracts;
- only an explicit correct confirmation can create a shadow candidate;
- replay ground-truth confirmations exercise candidate creation, corroboration, promotion, rejection, retirement, and rollback;
- adaptive updates append revisions instead of overwriting templates;
- initial and trusted re-enrollment templates can retire under the same utility policy as later templates;
- rejected, unconfirmed, or ambiguous events never enter active templates;
- database, model, and key corruption fails closed;
- encrypted persistence, deletion, rollback, and export/import behavior pass failure injection;
- chronological adaptive replay is compared with the frozen Phase-1A baseline without temporal leakage.

The synthetic 500-identity gallery measures comparison capacity and latency only. It does not estimate false acceptance, ranking quality, margin behavior, or policy-trigger frequency. The implementation plan must define an absolute comparison-latency budget before benchmarking and the report must show comparison time separately from end-to-end time.

Cross-platform conformance remains a Phase-2 entry gate.

Phase-1A tests cover schema/reason codes, zero/one/multiple-face behavior, preprocessing and embedding normalization, model-generation incompatibility, score/margin boundaries, no-network inference, deterministic fixtures, and failure handling without durable biometric state. Phase-1B adds confirmation paths, candidate corroboration, utility eviction, atomic revision rollback, encrypted export/import, deletion, log redaction, storage/key failure injection, and temporal-leakage tests. Temporal-leakage tests assert that the decision for event N cannot read any template revision or evidence timestamped after event N. With the small Phase-1 corpus, capacity eviction, rollback depth, and multi-identity promotion exclusion have synthetic or unit-level evidence only and must not be reported as production validation.

## 15. Future Roadmap

Phase 2 is not synonymous with mobile implementation. [ADR 0008](../decisions/0008-mac-live-identification-research.md) adds a bounded **Phase 2A Mac live-identification research prototype** before the mobile prototypes below. Its [design](2026-09-14-mac-live-identification-research-design.md) is the source of truth for camera/session experiments and the separate consented research-recorder exception to full-frame retention. Phase-1 scope and learning-confirmation restrictions remain unchanged. Multi-frame similarity may be researched without an authentication claim; it does not establish liveness or anti-replay. The representative-gallery gate in §8 remains mandatory for deployment/target-capacity claims, and the real-face carrier/cross-runtime gates in §§6/14 remain mandatory before mobile evaluation. These gates do not require a large hand-curated corpus before bounded Mac research may begin. Research design approval alone authorizes neither implementation nor participant recording.


1. **Android tablet prototype:** camera capture, offline 1:N identification, secure storage, and real 500-person device benchmarks.
2. **iOS tablet prototype:** same ONNX artifacts, schemas, and golden vectors with iOS performance evidence.
3. **Authentication layer:** trusted capture, liveness, anti-replay, multi-frame aggregation, and fallback methods before `authenticated` can exist.
4. **Product integration:** identity/policy synchronization, offline event queues, APIs, attendance rules, authorization, audit, correction, and retention.
5. **Pre-capture video adapter:** a later mobile capture feature may save one still plus up to five seconds immediately preceding the shutter action. Bounded Mac video-identification research is scoped by ADR 0008; mobile video recognition and video-driven production template updates remain outside that research authorization.
6. **Multi-face photo search:** an independent branch of work only after single-face identification is accurate and stable.
7. **Real-face cross-platform fixtures:** before Phase 2 evaluation, choose a reproducible carrier with explicit consent, retention, deletion, and redistribution rules; do not silently turn Phase-1 samples into permanent fixtures.

## 16. Resolved Review Decisions

No product decision from the review remains open. The operator selected:

1. no guaranteed periodic trusted re-enrollment;
2. confirmation-gated supervised learning with no unattended automatic learning;
3. separate Phase-1A accuracy and Phase-1B governance milestones;
4. a three-to-five-identity consented Phase-1A evaluation gallery.

The small gallery makes ranking and margin behavior observable but remains non-representative of a 500-person deployment. It does not relax the Phase-2 representative-gallery recalibration gate.

The initial enrollment template remains governed by the same bounded utility and retention rules as later templates. The rejected alternative of retaining it indefinitely as a hidden drift anchor would contradict that requirement and does not independently solve gradual poisoning.

## 17. Design Completion Gate

The operator approved the Phase-1 baseline on 2026-09-10. Its implementation plans and P0 authorization are recorded in PROJECT-STATE; these historical prerequisites must not be reinterpreted as an unfulfilled present-day implementation gate. Current model acceptance and research limitations remain explicit there. Each new research phase requires its own reviewed design, implementation plan and operator go; documentation approval alone is not execution authority.

Only after the Phase-1A evidence receives an operator go/no-go decision may a separate Phase-1B implementation plan define persistent local data paths, encryption and `KeyProvider` mechanics, confirmation and revision workflows, long-horizon drift indicators and response, rollback/deletion behavior, and its own verification commands.
