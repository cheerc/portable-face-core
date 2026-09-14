# ADR 0006: Split Phase 1 Into Accuracy and Governance Milestones

- Status: accepted
- Date: 2026-09-10

## 現況與後續關係

本 ADR 的分階段決策維持 accepted。Phase-1B 開工前 P0 已放行（`d-20260912041106780391-34`），不得將本 ADR 解讀成尚待首次開工批准。工程交付與模型選型／真實學習效果是不同 gate；最新 evidence 和未驗限制見 [PROJECT-STATE](../PROJECT-STATE.md)。原 plan 的「close ADR 0006」是收尾審閱用語，不把 accepted 狀態再當成待批准，也不等於批准下一階段實作。Mac 動態研究安排由 [ADR 0008](0008-mac-live-identification-research.md) 定義，不回溯擴張 Phase-1 靜態邊界。

## Context

The first question is whether one-shot face identification is accurate and useful with compliant portable models. Building encrypted persistence, adaptive-template governance, rollback, and migration before answering that question would increase cost and make model feasibility harder to isolate.

## Decision

- Phase 1A proves the static-image pipeline, one-shot accuracy, model bake-off, deterministic result contract, conformance fixtures, and capacity/latency evidence.
- Phase 1A uses a process-local in-memory repository and persists no biometric database.
- Phase 1A freezes revision-shaped result, policy, template, and repository contracts so Phase 1B can add persistence without destructive redesign.
- The operator reviews Phase-1A evidence and records a go/no-go decision before Phase 1B begins.
- Phase 1B implements persistent identity operations, confirmation-gated adaptive learning, encrypted storage, rollback, deletion, re-enrollment, export/import, and chronological adaptive evaluation.
- Phase 1A and Phase 1B require separate implementation plans and verification gates.

## Consequences

- Model selection and frozen one-shot evidence are not blocked on governance implementation.
- Phase 1A is demonstrable but is not the persistent local product.
- Confirmation and ground-truth-driven adaptive replay belong to Phase 1B.
- Failure of the Phase-1A accuracy gate can stop the project before it accumulates persistence and governance cost.
