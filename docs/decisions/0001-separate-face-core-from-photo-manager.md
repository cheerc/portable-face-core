# ADR 0001: Separate Face Core From Photo-Library Management

- Status: accepted at architecture level
- Date: 2026-09-09

## Context

The original idea was a shell-driven classifier that copied school photos into `confirmed`, `review`, and `rejected` directories. This would duplicate full-size originals and combine photo-library management, face inference, human review, and export into one tool.

The operator clarified two distinct goals:

1. Keep one physical original, browse and label it locally, and export selected photos for Google Photos or another consumer.
2. Build a reusable face-recognition capability that may later support applications such as attendance.

## Decision

- Evaluate PhotoPrism for indexing, browsing, person labeling, search, and explicit export of the photo library.
- Build Portable Face Core as an independent library/service contract for detection, alignment, embedding, verification, and identification.
- Do not make either project depend on the other's internal database.
- Treat batch-photo processing and future attendance as adapters/applications using the core, not as core responsibilities.

## Consequences

- The photo library can keep one original plus derived metadata and thumbnails instead of persistent full-size classification copies.
- Portable Face Core remains testable without PhotoPrism or private school photos.
- Attendance-specific security and policy require their own future spec.
- Some inference concepts and model research may be shared, but deployment, UX, storage, and acceptance criteria remain separate.
