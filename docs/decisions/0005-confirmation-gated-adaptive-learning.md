# ADR 0005: Confirmation-Gated Adaptive Learning

- Status: accepted
- Date: 2026-09-10

## Context

The product cannot assume that every identity will receive a trusted registration refresh each year. It therefore needs to accumulate useful later appearances, but an embedder can make the same confident identity mistake across multiple events. Unattended similarity-based promotion would allow those correlated errors to poison the active template bank.

## Decision

- Normal continuity does not depend on periodic trusted re-enrollment; trusted re-enrollment remains an optional operator recovery action.
- Phase 1B requires an explicit `correct` confirmation before an observation can become a shadow candidate.
- `not_me`, cancellation, timeout, or missing confirmation never learns.
- Confirmation does not bypass quality, temporal independence, corroboration, exclusion, capacity, or promotion policy.
- Self-confirmation is supervision evidence, not authentication or attendance authorization. Operator/staff confirmation uses a distinct actor type.
- Unattended automatic learning is excluded from Phase 1B.
- Chronological replay uses corpus ground truth as confirmation only inside the evaluation harness so candidate and promotion behavior remains measurable.

## Consequences

- The interactive adapter keeps the observation in memory only while confirmation is pending; no pending face crop or embedding is persisted.
- A confirmed observation may still remain a shadow candidate or be rejected by later gates.
- The initial and later templates remain bounded and replaceable; no permanent enrollment anchor is introduced.
- Long-horizon drift remains an explicit open risk for the Phase-1B plan and chronological evaluation.

