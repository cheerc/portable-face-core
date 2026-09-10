# ADR 0007: Small Consented Phase-1A Evaluation Gallery

- Status: accepted
- Date: 2026-09-10

## Context

A single enrolled identity can exercise target-versus-unknown thresholds but has no real runner-up. Selecting an open-set model from that evidence would leave top-1/top-2 margin behavior and cross-identity confusion unobserved. A full representative gallery is inappropriate for the first prototype.

## Decision

- Phase 1A enrolls three to five explicitly consented identities, each from one registration photo.
- Later target probes are grouped by identity; separate consented non-target probes test the unknown boundary.
- Every enrollment and probe image still contains exactly one usable face. Multi-face image handling remains excluded.
- Reports include per-identity outcomes, runner-up margins, and a confusion-matrix row for each enrolled identity.
- Evaluation biometric files stay local and outside Git. The corpus owner records consent, guardian authorization when required, retention, and deletion outside the Face Core identity schema.
- The gallery is evidence for prototype ranking and margin behavior only. It is not representative of 500 identities and cannot establish production false-acceptance or accuracy rates.

## Consequences

- Phase 1A can compare model candidates using non-degenerate identity competition.
- The evaluation corpus has additional privacy and consent obligations.
- Phase 2 still requires recalibration against a representative real gallery before multi-identity or target-capacity deployment.
- This decision changes the number of enrolled identities, not the one-face-per-input boundary.
