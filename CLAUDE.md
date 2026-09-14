# Agent Instructions

本 repo 已有 Phase-1A 靜態辨識與 Phase-1B 治理實作。現況、未驗限制與下一步授權以 `docs/PROJECT-STATE.md` 為入口；不要把歷史 plan 的待開工敘述當成目前狀態。Mac 動態研究設計見 `docs/specs/2026-09-14-mac-live-identification-research-design.md` 與 ADR 0008；文件合併不代表實作或資料蒐集獲授權。

## Start Here

Every new session must read these files in order before proposing or making changes:

1. `docs/PROJECT-STATE.md` — current status, approved decisions, pending gates, and next action.
2. `docs/specs/2026-09-09-portable-face-core-design.md` — the approved design and scope.
3. `docs/research/2026-09-09-open-source-face-stack.md` — evidence and licensing notes for candidate stacks.
4. `docs/decisions/0001-separate-face-core-from-photo-manager.md` — why this project excludes photo-library management.

After reading, inspect live repository state with `git status` and recent commits. Treat the documents as stale-tolerant anchors, not proof of current dependencies or model licenses.

## Hard Boundaries

- Do not add real faces, personal photos, face embeddings, identity databases, attendance records, secrets, or exported photos to Git.
- Do not treat a person's name or folder name as consent to biometric processing.
- Do not implement attendance as a simple photo match. Attendance requires a separately approved threat model, liveness/anti-spoofing design, event rules, authorization, audit, retention, and manual fallback.
- Do not bundle or download a model until its exact code license, weight license, redistribution right, checksum, and training-data provenance have been reviewed and recorded.
- Do not couple the portable core to PhotoPrism, Google Photos, a school-album downloader, a UI framework, or a single database engine.
- Do not call cloud face-recognition APIs or transmit face data unless a future spec explicitly changes this privacy boundary.
- Do not begin implementation until the operator approves the design spec and a separate implementation plan is written.

## Design Principles

- One core pipeline: detect → align → embed → compare.
- Stable interfaces around replaceable detection and embedding backends.
- Embeddings are biometric data; minimize, encrypt where appropriate, version, and delete deliberately.
- The approved product flow is open-set one-to-many identification; do not silently substitute a claimed-identity one-to-one flow.
- Return scores and decision bands, not only a boolean.
- Phase 1 uses `matched`, never `authenticated`; liveness and replay protection are not yet present.
- Keep policy outside the model: applications decide what `matched`, `review`, and `unknown` mean.
- Enrollment is one-shot: one registration action and one accepted photo per initial identity.
- New templates accumulate through a guarded shadow-candidate flow; no initial or later template is permanent or overwritten in place.
- Preserve portability by defining model-independent data contracts and keeping platform adapters thin.
- Tests use consented or synthetic fixtures only.

## Documentation Discipline

Update `docs/PROJECT-STATE.md` whenever an approved decision, current phase, or next-session entry point changes. Record architectural decisions under `docs/decisions/`. Keep research claims linked to primary sources and dated. Mark drafts and unverified claims explicitly.
