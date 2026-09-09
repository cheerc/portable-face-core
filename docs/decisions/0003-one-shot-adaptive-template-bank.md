# ADR 0003: One-Shot Enrollment and Adaptive Template Bank

- Status: accepted
- Date: 2026-09-09

## Context

Student registration is a single capture action, not a multi-minute Face ID-style scan. Children and other users change over time, so replacing one stored face with the latest face would forget earlier useful appearances, while unlimited automatic accumulation would increase false-match and poisoning risk.

## Decision

- Initial enrollment accepts exactly one registration photo containing one usable face.
- Successful recognition does not overwrite the initial or current template.
- Strict, high-quality results create shadow candidates. Independent later events must corroborate a candidate before promotion.
- Active templates are versioned and cumulative but bounded. All templates, including the initial enrollment template, are eligible for retirement.
- At capacity, utility combines quality, independent support, recency, appearance/pose coverage, redundancy, and known mismatch risk. The lowest-utility template retires.
- Retirement is reversible for a limited period; after expiry, retained crop and embedding data are deleted.

## Consequences

- Matching must be calibrated at the identity level because more templates create more opportunities for a coincidental high score.
- An unusual but useful profile or accessory sample must not be evicted merely because it is far from the centroid.
- Evaluation must compare frozen one-shot performance with chronological adaptive replay and must prevent future probes from leaking into earlier decisions.
