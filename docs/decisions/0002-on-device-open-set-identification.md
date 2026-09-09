# ADR 0002: On-Device Open-Set Identification

- Status: accepted
- Date: 2026-09-09

## Context

The intended tablet experience identifies a person without requiring them to select a name, scan a badge, or claim an identity first. The expected population is no more than about 500 people, and recognition must continue when the network is unavailable.

## Decision

- Use open-set 1:N identification, not Face ID-style claimed-identity 1:1 verification, as the primary product flow.
- Perform capture processing, quality checks, inference, exact comparison, and decision policy locally on the tablet.
- Treat false acceptance as more serious than false rejection. Insufficient score, runner-up margin, or quality returns `review` or `unknown` rather than a guessed name.
- Network communication is reserved for versioned identity/policy synchronization and business events. Offline events queue durably.
- Phase 1 returns `matched`, not `authenticated`, because file input provides no liveness or anti-replay evidence.

## Consequences

- A one-person prototype still requires non-target probes to measure false acceptance.
- The identity score must consider absolute similarity, identity-template aggregation, and top-1/top-2 margin when multiple identities exist.
- Production authentication needs a later security layer beyond Face Core similarity.
