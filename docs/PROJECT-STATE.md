# Project State

Last updated: 2026-09-09 Asia/Taipei

## Current Status

Portable Face Core is in **design draft / research initialization**. Documentation and repository boundaries exist; implementation, dependency installation, model download, training, enrollment, and private-data processing have not started.

The operator has approved the high-level split:

- PhotoPrism is evaluated separately as the local photo manager and export tool.
- This repository owns only a reusable face-recognition core and its contracts.

The detailed design in `docs/specs/2026-09-09-portable-face-core-design.md` still requires operator review before an implementation plan is written.

## Approved Requirements

- Local filesystem processing; paths may live in Google Drive, iCloud Drive, Dropbox, or mounted NAS folders.
- The core itself must not call a cloud face-recognition API or upload telemetry/biometric data.
- Multiple identities are supported by the data model; the first prototype may validate one identity.
- Initial enrollment samples contain exactly one clear target face per image. Zero-face or multi-face seeds must be rejected with reasons rather than guessed.
- Similarity results use two configurable thresholds: high-confidence `confirmed`, middle-band `review`, and below-threshold `rejected`.
- False matches may be retained as negative evaluation evidence, but must not be silently treated as positive identity templates.
- Code, core libraries, and distributed model artifacts must allow commercial use and redistribution.
- Phase-one daily operation may expose a single shell entrypoint, but public core interfaces must not depend on shell or Python.
- Future targets include macOS, Windows, Android, and iOS.

## Important Corrections

- Enrollment is usually embedding/template creation, not retraining the foundation neural network.
- A photo manager and a reusable recognition engine are different products.
- Attendance is not equivalent to finding a person in archived photos. It adds live capture, anti-spoofing, identity policy, authorization, event deduplication, audit, retention, and manual fallback.
- A model repository license alone may not settle training-data provenance. Public/commercial release requires an explicit model compliance record.

## Current Candidate Direction

Use a replaceable backend interface around a portable pipeline:

```text
image/frame
  → face detector
  → landmark alignment and quality checks
  → embedding model
  → template matcher
  → score + decision band + diagnostics
```

OpenCV with YuNet and SFace is the leading prototype candidate because official implementations exist in Python and C++ and OpenCV targets desktop and mobile. This is not yet a locked dependency. Exact weights, versions, checksums, licenses, provenance, accuracy on consented fixtures, and mobile runtime compatibility remain release gates.

## Pending Decisions

1. Operator approval or revision of the draft design.
2. Project license for original source code, likely Apache-2.0 or MIT.
3. Exact detector and embedding model artifacts after compliance review.
4. Template aggregation strategy: all embeddings, representative medoids, centroid plus outlier retention, or a hybrid.
5. Encryption/key-management boundary for local template storage.
6. Phase-one benchmark dataset and measurable acceptance thresholds.
7. Whether the first implementation plan includes only file-based verification/identification or also a camera adapter spike.

## Next Session

1. Read `AGENTS.md` and the documents it lists.
2. Review the design with the operator section by section.
3. Resolve ambiguities and update the spec; scan for placeholders, contradictions, and scope leaks.
4. Ask the operator to approve the written spec.
5. Only after approval, write a detailed implementation plan. Do not implement in the same step.
