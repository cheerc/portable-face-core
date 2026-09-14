# Phase 1B Implementation Plan

> 現況註記（2026-09-14）：本檔保留原實作工作令與驗收要求，不是新 session 待派清單。P0 已由 `d-20260912041106780391-34` 放行；PR-C–G 全波段授權為 `d-20260912090115587658-42`，工程已 merge。原凍結 SHA 與 pending premise 是當時 dispatch 的歷史前提，不適用後續新階段。真圖 replay 已由 PR #38 補入；其零 creation 並不證明真實 promotion 或長期改善。最新已完成／未驗項與下一步見 [PROJECT-STATE](../PROJECT-STATE.md)。Task 11 所稱「close ADR 0006」指研究收尾審閱，不是重新要求 P0，也不自動關 selection gate 或授權 mobile。新 Mac 研究依 [ADR 0008](../decisions/0008-mac-live-identification-research.md) 另寫 plan、另取實作 go。

**Source of truth:** `docs/specs/2026-09-09-portable-face-core-design.md` §§9, 10, 11, 12, 14, approved by the operator on 2026-09-10 and merged to `main` at `f2a68a83c9548d61396ddf3cafa80e9c3d951c11`.

**Governing decisions:**
- ADR 0004: ONNX-first portability and runtime architecture.
- ADR 0006: Split Phase 1 into 1A (accuracy/one-shot) and 1B (governance/adaptive lifecycle).
- ADR 0007: Small consented evaluation gallery with exact denominators.
- Decision `d-20260911181002229319-23`: Operator selects Candidate A (SFace Pair 1 provisional; frozen gates unchanged; selection gate remains OPEN pending 1B chronological replay).
- Decision `d-20260911181317619341-25`: Phase-1B implementation and go/no-go held pending written plan.
- Decision `d-20260911174928737696-16`: ONNX weights download dual-gate blocker (S1 clear + operator decision required).
- Decision `d-20260911171253541089-12`: Pair 2 comparison preconditions (matrix stage).
- Decision `d-20260911184146033747-27`: Team discuss arbitration (D1–D5 rulings, 7-PR structure, layered premises, frozen manifest).
- Decision `d-20260911185748092685-28`: PR #19 review rework instructions (P1/P2 fixes).
- Decision `d-20260911190712794016-29`: PR #19 r1 review rework round 2.
- Decision `d-20260911191746908253-30`: PR #19 r2 review rework round 3.
- Decision `d-20260911192552843761-31`: PR #19 r3 review rework round 4.
- Decision `d-20260911193316595438-32`: PR #19 r4 review rework round 5 (canonical 9-field manifest predicate in protocol and Task 6 including runtime contract, candidate migration all-status lifecycle, Task 11/PR-G operator backup manifest dependency, versioned Argon2id runtime dependency contract with fail-closed UnsupportedKdfError).

**Goal:** Extend the verified in-memory Phase-1A face recognition pipeline into a production-grade on-device governance engine: persistent encrypted identity and candidate storage, `KeyProvider` abstraction with journaled tombstone atomicity and crash recovery, confirmation-gated shadow candidate learning, multi-event corroboration, utility-driven bounded template bank with capacity eviction, atomic revisions and identity recovery CLI (add, show, re-enroll, rollback, delete, candidate reject), active and candidate template retained exemplar lifecycle with all-status generation migration, outer-authenticated export/import key re-homing with pinned versioned Argon2id KDF, canonical 9-field model manifest compatibility verification including runtime contracts, long-horizon drift indicators with bounded response, and a chronological adaptive replay harness with strict temporal-leakage prevention that measures adaptive behavior against the frozen Phase-1A baseline.

**Tech stack:** Python, Encrypted Storage Repository (engine and cipher scheme determined by Spike S1B decision manifest), `KeyProvider` abstraction, NumPy, Pillow, pytest, ruff, mypy. Pure on-device offline execution; zero network access; zero external ORM or unvetted native dependencies.

---

## 1. Frozen Source Manifest & Freshness Verification

In accordance with arbitration `d-20260911184146033747-27` item (8), this implementation plan freezes against immutable source artifacts. Every subsequent implementation dispatch under Phase 1B must verify these identities before beginning mutation:

- **Canonical repository:** `cheerc/portable-face-core` (`https://github.com/cheerc/portable-face-core.git`)
- **Main merge base SHA:** `98b202a516dfd4722326a7601073f24a008bfddb` (Phase-1A closeout merge PR #18)
- **Base tree SHA:** `71e3323c4c24249fdfcd881ea679b45f693b5a9f`
- **Spec blob SHA:** `1e555b5f9674d972da45c602d12818e2cdf2cafe` (`docs/specs/2026-09-09-portable-face-core-design.md`)
- **1A plan blob SHA:** `a9b2bdec8da17eb61cac26ce66dac149342cee0d` (`docs/plans/2026-09-10-phase-1a-implementation-plan.md`)
- **PROJECT-STATE blob SHA:** `f5dcfa4f41137e69bd6ca5d6b8eda6873eae66e5` (`docs/PROJECT-STATE.md`)
- **README blob SHA:** `9f3c8f6971812211c64603521e246321c79e6764` (`README.md`)
- **Active board decisions:**
  - `d-20260911181002229319-23` (Operator selects provisional SFace Pair 1)
  - `d-20260911181317619341-25` (Phase-1B on hold pending plan)
  - `d-20260911174928737696-16` (ONNX weights dual-gate requirement)
  - `d-20260911171253541089-12` (Pair 2 comparison preconditions)
  - `d-20260911184146033747-27` (Phase-1B discuss arbitration)
  - `d-20260911185748092685-28` (PR #19 review rework instructions)
  - `d-20260911190712794016-29` (PR #19 r1 review rework round 2)
  - `d-20260911191746908253-30` (PR #19 r2 review rework round 3)
  - `d-20260911192552843761-31` (PR #19 r3 review rework round 4)
  - `d-20260911193316595438-32` (PR #19 r4 review rework round 5)

**Dispatch verification rule:** If the provider `main` tip, relevant blob SHAs, or active governing decisions differ from this manifest during task intake, the implementer must halt immediately, report the drift to the Lead, and await explicit re-base instruction.

---

## 2. Global Constraints & Non-Negotiable Boundaries

These rules govern all Phase-1B implementation work without exception (spec §§9–12, 14; arbitration item 10):

1. **Implementation Gate (Prerequisite P0):** Zero production code or persistent database modification is permitted prior to operator approval of Prerequisite P0 (ADR 0006 Go/No-Go Decision Gate). All implementation tasks and implementation PRs (PR-C through PR-G) depend strictly on P0.
2. **Strictly offline operation:** Network availability or latency must never alter thresholds, execution paths, or recognition results.
3. **Zero plaintext biometric data at rest:** All face crops (`EncryptedExemplar`) and embeddings (`FaceTemplate`, `CandidateTemplate`) stored on disk must be encrypted at rest using authenticated encryption (AEAD). Unencrypted biometric blobs must never touch the filesystem.
4. **Zero-disk before confirmation:** During `--confirm-learning`, observation frames, crops, and embeddings remain strictly in process memory. Cancellation, EOF, timeout, process interruption, or `not_me` exits immediately without writing any candidate or biometric data to disk (spec §9 line 179).
5. **No `authenticated` state:** Recognition results are strictly constrained to `matched`, `review`, `unknown`, or `invalid_input` (spec §11 lines 237, 270–292). Self-confirmation is human supervision for candidate generation, never cryptographic or identity authentication (spec §9 lines 194–195).
6. **Strict temporal-leakage prevention:** In chronological replay and evaluation, the decision state and template bank at event $N$ must never read or depend upon any template revision, candidate, or observation timestamped $> N$ (spec §14 line 390).
7. **Fail-closed error handling:** Storage corruption, key unavailability, model incompatibility, unsupported KDF, or tampering must fail closed and emit structured exit codes: undecodable input exits `2`, model mismatch exits `3`, storage/key/KDF failure exits `4`, invalid configuration exits `5`, and internal error exits `7` (spec §11 lines 298–299).
8. **Single-face boundary maintained:** Exactly one usable face per registration and probe input. Zero-face or multi-face inputs return `invalid_input` with reason codes and never create or alter identity records (spec §7 lines 153–154).
9. **Real biometric assets remain outside Git:** Real face images, consented evaluation galleries, decrypted databases, and reports containing per-probe identifiable details are denied by `.gitignore` and must never be committed.

---

## 3. Evidence and Open Hypotheses (Layered Premises)

In accordance with arbitration `d-20260911184146033747-27` and rework decisions `d-20260911185748092685-28`, `d-20260911190712794016-29`, `d-20260911191746908253-30`, and `d-20260911193316595438-32`, project premises are strictly partitioned into **Operator-level** and **Commander/Lead-level**.

### Operator-level Premises (Provisional & Non-Binding until Operator Decision Manifest)

**CRITICAL POLICY ENFORCEMENT:** All numerical thresholds, retention periods, key custody models, backup ceilings, and policy values listed below and in Section 6 are **provisional recommendations only**. They are **non-binding** and do not authorize implementation. Every implementation task that consumes these values (**Tasks 2, 3, 4, 5, 6, 7, 8, and 11**) carries a hard structural dependency on an **Operator Decision Manifest** recording explicit sign-off on each item.

1. **Formal Phase-1B Go/No-Go Decision (ADR 0006 Gate):**
   - *Status:* Pending operator review of this plan and Phase-1A evidence (`d-20260911181317619341-25`).
   - *Requirement:* Hard gate for PR-C through PR-G.
2. **ONNX Weights Re-download Authorization (Dual Gate):**
   - *Status:* Worktree release removed local weights; S1 license is CLEAR but SFace #313 provenance is open (`d-20260911174928737696-16`).
   - *Requirement:* Operator must record explicit approval before weights may be retrieved.
3. **P1 Corpus Extension & Chronological Probe Supply:**
   - *Status:* Phase 1A had 1 usable target probe (`PROJECT-STATE.md`). 1B requires chronological replay.
   - *Requirement:* Operator provides multi-timestamp consented probe sequences, OR formally authorizes synthetic temporal streams as the primary governance verification vehicle while real replay is labeled `blocked` with open selection gate (`d-20260911181002229319-23`).
4. **Exemplar Crop Margin & Storage Authorization (`exemplar_margin`):**
   - *Status:* Spec §10 line 222 defines `EncryptedExemplar` as a bounded pre-alignment region plus `exemplar_margin`.
   - *Provisional recommendation:* Minimal crop margin `0.0` (tight crop, no extra background) to honor data minimization. Binding value set in Operator Decision Manifest. Consumed by Tasks 1, 2, 7.
5. **Promotion-Confirmation Semantics:**
   - *Status:* Spec §9 line 176 requires explicit `correct` confirmation for candidate creation.
   - *Provisional recommendation:* Candidate creation strictly requires explicit `correct`; once created and corroborated by subsequent independent events, promotion to active template is deterministic and automated. Binding value set in Operator Decision Manifest. Consumed by Tasks 3, 4.
6. **Retention TTL and Rollback Revision Depth Limit:**
   - *Status:* Spec §9 line 211 states retired templates remain available for a limited rollback policy and are deleted after retention expiry.
   - *Provisional recommendation:* Maximum rollback depth = 5 revisions; retired template retention TTL = 90 days. Binding values set in Operator Decision Manifest. Consumed by Tasks 2, 4, 5, 8.
7. **Key Custody Model:**
   - *Status:* Spec §10 line 227 mandates keys stay outside DB via `KeyProvider`.
   - *Provisional recommendation:* Software-backed external key store for macOS development CLI (`FileKeyProvider`), with interface seam for Phase-2 mobile KeyStore/KeyChain. Binding value set in Operator Decision Manifest. Consumed by Tasks 2, 5, 6.
8. **Backup Retention Ceiling:**
   - *Status:* Arbitration item (7) mandates backup retention limit.
   - *Provisional recommendation:* `backup_max_count = 5`. Binding value set in Operator Decision Manifest. Consumed by Tasks 2, 11.
9. **Actor Taxonomy in Audit Trail:**
   - *Status:* Spec §11 lines 263–265 distinguishes `user` (supervision) and `operator` (recovery/re-enroll).
   - *Provisional recommendation:* `"user"` for live probe confirmation; `"operator"` for CLI administrative actions. Binding value set in Operator Decision Manifest. Consumed by Task 5.

### Commander/Lead-level Premises (Architectural Decisions Resolved in Spikes)

1. **Storage Engine & Encryption Scheme Selection (Spike S1B):**
   - *Requirement:* Neutral architecture specification. The system requires an `EncryptedStorageRepository` satisfying spec §10 and §12 (confidentiality at rest, authenticated integrity, fail-closed exit code 4, ACID transactions, atomic revision commits, zero plaintext on disk).
   - *Spike Responsibility:* Spike S1B evaluates storage routes (e.g., standard library SQLite with application-layer AEAD field encryption vs alternatives) and produces the binding `storage-crypto-manifest.md`. Task 2 depends strictly on S1B's decision manifest; no specific engine, cipher mode, or AAD format is preselected in this plan.
2. **Transaction Isolation & Journaled Tombstone Protocol:**
   - SQLite transactions execute atomically (`BEGIN IMMEDIATE`). Key destruction in external `KeyProvider` is coordinated via a **Journaled Tombstone & Recovery Protocol** (Section 7) ensuring crash-consistent, idempotent, and fail-closed atomicity across SQLite/KeyProvider/WAL boundaries, including explicit recovery when keys were already destroyed before crash.
3. **Event Independence & Composite Ordering Contract:**
   - In accordance with arbitration `d-20260911184146033747-27` item (6) and finding P2-13, event ordering and independence are determined by the composite key `(timestamp, sequence_number, event_uuid, source_sha256)`.
   - Equal timestamps are valid and processed strictly in ascending `sequence_number` order. Consecutive events within `min_corroboration_interval_secs` (provisional 60s) or with matching image digests are treated as a single burst and cannot satisfy independent corroboration. A backward `sequence_number` is an error and rejected.
4. **Utility Factor Formulas & Deterministic Tie-Breaking (Finding P1-2):**
   - All six factors defined by exact mathematical formulas normalized to `[0.0, 1.0]`. Composite score is explicitly clamped to $[0.0, 1.0]$. Tie-breaking uses earliest creation timestamp, with `template_id` lexicographical comparison as the final deterministic tie-breaker.
5. **Identity Collision Defense:**
   - `identity add` is create-only and enforces strict uniqueness at the storage layer; existing IDs fail immediately without mutation (exit code 4).
6. **Active and Candidate Retained Exemplar Lifecycle & All-Status Migration (Finding P1-2):**
   - Both active templates (`face_templates`) and candidate templates (`candidate_templates`) retain their associated `encrypted_exemplar` and crop metadata throughout their lifecycle. When a model manifest changes, stored active exemplars are re-embedded, and candidates across all statuses undergo defined transitions: pending candidates are either re-embedded to $G_2$ or marked `rejected` (reason: `model_generation_retired`); promoted candidates are archived as `generation_retired` terminal records; rejected/expired candidates are preserved as historical records permanently barred from matching or promotion under $G_2$.
7. **Canonical 9-Field ModelManifest Compatibility Predicate (Finding P1-1):**
   - A single shared predicate governs both export/import re-homing (Section 8) and CLI import acceptance (Task 6), explicitly comparing: (1) embedder artifact hash, (2) detector generation, (3) preprocessing generation, (4) tensor layout, (5) normalization contract (scale/mean/std), (6) embedding dimension, (7) numerical precision, (8) quantization type, and (9) execution runtime/provider contract.

---

## 4. Documentation Impact Check

Classification: **`area`**.

Evidence: Adding `docs/plans/2026-09-12-phase-1b-implementation-plan.md` updates the repository map in `README.md:29-45`, and tracking Phase-1B progress updates `docs/PROJECT-STATE.md`. Neither changes existing architectural decisions ADR 0004/0006/0007.

- Task 0 updates `README.md` and `docs/PROJECT-STATE.md` to record Phase-1B planning.
- Task 11 performs the final `project-docs-maintain` pass before Phase-1B completion.

---

## 5. Prerequisites and Spikes

In accordance with arbitration `d-20260911184146033747-27` item (1), implementation tasks depend on three explicit spikes/gates:

### Prerequisite P0 — Phase-1B Go/No-Go Decision Gate
- **Nature:** Non-coding procedural gate. **Owner:** Operator.
- **Dependency:** Hard blocking prerequisite for Tasks 1–11 and PR-C through PR-G. PR-A (this plan) and PR-B (spikes) are docs-only and exempt.
- **Enforcement:** Implementation tasks fail closed if P0 is unresolved. Task 11 includes an integration test confirming CLI subcommands fail closed if the environment/database is unapproved.

### Spike S1B — Storage, Cryptography, and KeyProvider Architecture
- **Time box:** One working session. **Owner:** Implementer.
- **Deliverable:** `docs/research/2026-09-12-storage-crypto-manifest.md` on branch `docs/storage-crypto-manifest`.
- **Scope (Arbitration Items 1, 3, 4; Findings P1-1, P1-2, P2-KDF):**
  1. Evaluate storage and cipher options (standard SQLite + AEAD field encryption vs alternatives) without preselection.
  2. Define authenticated encryption scheme, AAD structure, nonce generation, and format versioning.
  3. Detail `KeyProvider` interface and deliver macOS software key provider (`FileKeyProvider`).
  4. Specify external `KeyProvider` placement for record-specific Data Encryption Keys (DEKs). SQLite stores only opaque `key_id` references; DEKs are managed, stored, and destroyed exclusively within the KeyProvider.
  5. Specify the Journaled Tombstone & Recovery Protocol for two-phase atomic key destruction and SQLite mutation, detailing the key-already-absent reconciliation path.
  6. Pin and verify the implementation dependency for Argon2id (e.g. standard wheels via `cryptography` AEAD / Argon2 CFFI) and record exact package version in `storage-crypto-manifest.md`.
  7. Detail transaction boundaries, crash consistency, and fail-closed WAL checkpoint handling.
  8. Explicitly document threat model limitations: Logical deletion + cryptographic key erasure via KeyProvider; acknowledge physical SSD/flash wear-leveling remanence limitations. Key rotation is deferred to Phase 2.
- **Blocks:** Task 2 (Encrypted Storage Repository).

### Spike S2B — Model Weights and Corpus Chronology Readiness
- **Time box:** One working session. **Owner:** Implementer.
- **Deliverable:** `docs/research/2026-09-12-weights-corpus-readiness.md` on branch `docs/weights-corpus-readiness`.
- **Scope (Arbitration Items 1, 6):**
  1. Record immutable SHA-256, license status, and file paths for SFace 2021dec and YuNet 2023mar.
  2. Inventory consented evaluation probes, timestamps, and identity groupings.
  3. If weights or multi-timestamp probes remain unavailable under operator dual-gate rules: formally freeze the synthetic adversarial stream specifications for Task 9, and record real replay as `blocked-with-reason`.
- **Blocks:** Task 9 (Replay Harness) and Task 10 (Evaluation Run).

---

## 6. Governance Policy Defaults (`governance_policy_version: 1`)

**STATUS: PROVISIONAL RECOMMENDATIONS — NON-BINDING PENDING OPERATOR DECISION MANIFEST.**

The values below represent the architect's recommendations. They become binding only when ratified in the Operator Decision Manifest.

| Parameter | Type / Unit | Provisional Value | Specification Reference & Definition |
|---|---|---|---|
| `candidate_update_threshold` | float | `>= 0.88` | Spec §9 line 176: Stricter than normal `match_threshold` (0.80) to prevent template poisoning. |
| `additional_corroboration_min_events` | integer | `1` | Spec §9 line 190: Minimum count of *additional* temporally independent events beyond creation seed. |
| `burst_suppression_min_interval_secs`| float (seconds) | `60.0` | Spec §9 line 191: Consecutive probe events within 60s cannot satisfy independent corroboration. |
| `promotion_margin` | float | `>= 0.12` | Spec §9 line 188: Candidate score margin over all other enrolled identities. |
| `template_bank_capacity` | integer | `5` | Spec §9 line 200: Maximum active templates per identity; triggers utility eviction. |
| `utility_weight_quality` ($w_Q$) | float | `0.30` | Spec §9 line 203: Weight for normalized image and detection quality. |
| `utility_weight_support` ($w_S$) | float | `0.30` | Spec §9 line 204: Weight for independent-event corroboration count. |
| `utility_weight_recency` ($w_R$) | float | `0.20` | Spec §9 line 205: Weight for observation recency. |
| `utility_weight_coverage` ($w_C$) | float | `0.20` | Spec §9 line 206: Weight for appearance diversity from active template centroid. |
| `utility_penalty_redundancy` ($w_{Red}$) | float | `0.15` | Spec §9 line 207: Penalty weight for near-duplicate active templates. |
| `utility_penalty_outlier` ($w_O$) | float | `0.15` | Spec §9 line 208: Penalty weight for proximity to non-target decision boundary. |
| `drift_max_centroid_shift` | float | `0.20` | Spec §9 line 198: Maximum permissible cosine distance of active centroid from reference anchor. |
| `drift_max_initial_distance` | float | `0.25` | Spec §9 line 198: Maximum distance of individual template from reference anchor (active while anchor exists). |
| `rollback_max_depth` | integer | `5` | Spec §9 line 211: Maximum historical revisions preserved for recovery rollback. |
| `retired_retention_days` | integer | `90` | Spec §9 line 211: Retention TTL for retired templates before cryptographic deletion. |
| `revision_history_max_count` | integer | `20` | Arbitration item (7): Upper bound on preserved revision records per identity. |
| `match_events_max_count` | integer | `10000` | Arbitration item (7): Storage retention ceiling for non-biometric event log. |
| `backup_max_count` | integer | `5` | Arbitration item (7): Upper bound on preserved historical database backups. |
| `exemplar_margin` | float | `0.0` | Spec §10 line 222: Minimal crop margin retained to satisfy data minimization. |

### Exact Utility Rescoring Formulas & Composite Normalization (Finding P1-2)

When the active template bank reaches capacity ($K = 5$), every active template $i$ is rescored using a strictly bounded utility function:

$$U(i) = \text{clamp}\left(S_{\text{base}}(i) - w_{Red} \cdot U_{Red}(i) - w_O \cdot U_O(i), 0.0, 1.0\right)$$

where the positive base score $S_{\text{base}}(i)$ is normalized by construction because positive weights sum to $1.00$ ($w_Q + w_S + w_R + w_C = 0.30 + 0.30 + 0.20 + 0.20 = 1.00$):

$$S_{\text{base}}(i) = w_Q \cdot U_Q(i) + w_S \cdot U_S(i) + w_R \cdot U_R(i) + w_C \cdot U_C(i) \in [0.0, 1.0]$$

Each individual component is strictly mapped to $[0.0, 1.0]$:
1. **Quality ($U_Q$):**
   $$U_Q(i) = \text{clamp}\left(0.5 \cdot \frac{\text{VarLap}(i)}{120.0} + 0.5 \cdot \text{det\_conf}(i), 0.0, 1.0\right)$$
2. **Support ($U_S$):**
   $$U_S(i) = \text{clamp}\left(\frac{\text{additional\_corroboration\_count}(i)}{3.0}, 0.0, 1.0\right)$$
   (Base case: $0.0$ if no additional events).
3. **Recency ($U_R$):**
   $$U_R(i) = \text{clamp}\left(e^{-\frac{\ln(2)}{180.0} \cdot \Delta t}, 0.0, 1.0\right)$$
   where $\Delta t$ is elapsed days since template creation.
4. **Coverage ($U_C$):** Cosine similarity $\cos \in [-1.0, 1.0]$ is mapped linearly to $[0.0, 1.0]$:
   $$U_C(i) = \text{clamp}\left(\frac{1.0 - \cos(\mathbf{e}_i, \mathbf{c}_{-i})}{2.0}, 0.0, 1.0\right)$$
   - Identical to centroid ($\cos = 1.0$) $\implies U_C = 0.0$.
   - Orthogonal ($\cos = 0.0$) $\implies U_C = 0.5$.
   - Anti-parallel ($\cos = -1.0$) $\implies U_C = 1.0$.
   - Base case (bank has 1 template): $1.0$.
5. **Redundancy ($U_{Red}$):**
   $$U_{Red}(i) = \text{clamp}\left(\max_{j \ne i} \cos(\mathbf{e}_i, \mathbf{e}_j), 0.0, 1.0\right)$$
   (Negative cosines clamp to $0.0$; base case if bank has 1 template: $0.0$).
6. **Outlier Risk ($U_O$):**
   $$U_O(i) = \text{clamp}\left(\frac{\text{sim\_runner\_up}(i) - 0.70}{0.15}, 0.0, 1.0\right)$$
   (Base case if no runner-up: $0.0$).

**Mathematical Bound Guarantee:** Because $S_{\text{base}}(i) \in [0.0, 1.0]$ and penalties $w_{Red} \cdot U_{Red} \ge 0, w_O \cdot U_O \ge 0$, the composite expression bounded by $\text{clamp}(\cdot, 0.0, 1.0)$ guarantees $U(i) \in [0.0, 1.0]$ unconditionally across all boundary conditions (including anti-parallel, duplicate, empty-bank, and all-penalty scenarios).

**Deterministic Tie-Breaking:** If $U(i) == U(j)$, the template with the earlier creation timestamp $\min(\text{created\_at})$ is preserved. If timestamps are identical, the template with the lexicographically smaller `template_id` is preserved.

---

## 7. Storage, Deletion, and History Retention Mechanics

### Data Model Schema (`schema_version: 1`)

The storage layer enforces strict relational isolation. **Key Placement Rule:** All Data Encryption Keys (DEKs) are generated, stored, and managed exclusively by the external `KeyProvider`. The database stores only opaque key identifiers (`key_id`) and encrypted byte payloads:

1. `identities`: `id` (TEXT PRIMARY KEY), `display_name` (TEXT), `status` (TEXT: `active`, `re_enrollment_required`, `deleted`), `current_revision` (INTEGER), `created_at` (TEXT), `updated_at` (TEXT).
2. `template_revisions`: `id` (INTEGER PRIMARY KEY AUTOINCREMENT), `identity_id` (TEXT), `revision` (INTEGER), `active_template_ids` (TEXT: JSON list), `retired_template_ids` (TEXT: JSON list), `policy_version` (INTEGER), `created_at` (TEXT), `actor` (TEXT: `user`, `operator`).
3. `face_templates`: `id` (TEXT PRIMARY KEY), `identity_id` (TEXT), `generation_id` (TEXT), `status` (TEXT: `active`, `retired`), `key_id` (TEXT), `encrypted_embedding` (BLOB), `nonce` (BLOB), `encrypted_exemplar` (BLOB), `exemplar_crop_box` (TEXT: JSON), `exemplar_landmarks` (TEXT: JSON), `exemplar_margin` (REAL), `quality_score` (REAL), `utility_score` (REAL), `additional_corroboration_count` (INTEGER), `created_at` (TEXT), `retired_at` (TEXT NULL).
4. `candidate_templates`: `id` (TEXT PRIMARY KEY), `identity_id` (TEXT), `generation_id` (TEXT), `status` (TEXT: `pending`, `promoted`, `rejected`, `expired`, `generation_retired`), `key_id` (TEXT), `encrypted_embedding` (BLOB), `encrypted_exemplar` (BLOB NULL), `exemplar_crop_box` (TEXT NULL), `exemplar_landmarks` (TEXT NULL), `nonce` (BLOB), `quality_score` (REAL), `additional_corroboration_count` (INTEGER DEFAULT 0), `evidence_log` (TEXT: JSON), `expires_at` (TEXT), `created_at` (TEXT).
5. `match_events`: `id` (TEXT PRIMARY KEY), `timestamp` (TEXT), `sequence_number` (INTEGER), `status` (TEXT), `decision_score` (REAL), `runner_up_score` (REAL NULL), `matched_identity_id` (TEXT NULL), `candidate_created` (INTEGER), `actor` (TEXT NULL). (Zero image or embedding data).
6. `deletion_tombstones`: `id` (INTEGER PRIMARY KEY AUTOINCREMENT), `target_type` (TEXT: `record`, `identity`), `target_id` (TEXT), `key_id` (TEXT NULL), `status` (TEXT: `pending_key_destruction`, `key_destroyed`), `created_at` (TEXT).

### Retained Exemplar Lifecycle (Finding P1-2)

To satisfy spec §12 line 313 (model migration and re-embedding), every active and candidate template retains an encrypted crop:
1. **Enrollment:** Initial registration photo produces active template 1. Its pre-alignment face crop is encrypted under a unique record DEK and saved in `face_templates.encrypted_exemplar` alongside crop coordinates and landmarks.
2. **Promotion:** When a candidate promotes to active template, its `encrypted_exemplar` is preserved and linked to the new active template record.
3. **Trusted Re-enrollment:** Generates a new active template containing the new photo's encrypted exemplar.
4. **Retirement & Purge:** Retired templates preserve their encrypted exemplar until retention expiry (90 days), after which the exemplar DEK is destroyed via KeyProvider and the record is purged.

### Journaled Tombstone & Two-Phase Erasure Protocol (Findings P1-1, P1-Tombstone)

Because SQLite transactions cannot span external `KeyProvider` operations, deletion and purge execute via a durable **Journaled Tombstone Protocol** with explicit recovery for already-destroyed keys:

1. **Step 1 (Journaled Tombstone in SQLite):**
   - SQLite transaction begins (`BEGIN IMMEDIATE`).
   - For `identity delete`: `identities.status` is set to `deleted`, and records in `deletion_tombstones` are inserted with `status = 'pending_key_destruction'`. Plaintext buffers in memory are scrubbed.
   - For record purge: target record is marked with tombstone record in `deletion_tombstones`.
   - Active queries immediately filter out tombstoned identities and records.
   - Transaction commits.
2. **Step 2 (Key Destruction in KeyProvider):**
   - Repository invokes `key_provider.destroy_key(key_id)` or `key_provider.destroy_identity_keys(identity_id)`.
   - DEKs are permanently destroyed in the external key store.
3. **Step 3 (Final Purge & Checkpoint in SQLite):**
   - SQLite transaction begins (`BEGIN IMMEDIATE`).
   - Ciphertext blocks in `face_templates` and `candidate_templates` are overwritten with random bytes (`os.urandom`) and rows deleted.
   - Rows in `identities`, `template_revisions`, and `deletion_tombstones` are deleted.
   - Associated `match_events` have `matched_identity_id` set to `NULL` (audit anonymization).
   - Transaction commits.
   - SQLite `PRAGMA wal_checkpoint(TRUNCATE)` is executed immediately. If checkpointing fails or returns busy, operation raises `StoreCorruptionError` (exit code 4).
4. **Crash Recovery & Reconciliation (`reconcile_tombstones()`):**
   - On repository initialization, `reconcile_tombstones()` scans for records in `deletion_tombstones`:
     - **Branch A (`status = 'pending_key_destruction'`):** Checks if the key still exists in KeyProvider. If key exists, it calls `destroy_key()`. If key is already absent (e.g. crash occurred after key destruction but before Step 3), it treats `key_already_absent` as idempotent success, updates tombstone status to `key_destroyed`, and proceeds to Step 3.
     - **Branch B (`status = 'key_destroyed'`):** Executes Step 3 (SQLite row purge, match event anonymization, tombstone deletion, WAL checkpoint).
   - Guarantees fail-closed idempotent completion across all crash boundaries (before destroy, partial destroy, after destroy, checkpoint failure).
5. **Threat Model Limitation:**
   - *Scope:* Destroying DEKs in the external KeyProvider guarantees that database-only backups, WAL files, and disk slack space become permanently undecryptable.
   - *Limitation 1 (Simultaneous Full Backup):* If an operator copies both the database file AND the external KeyProvider secret file at the exact same instant prior to deletion, that offline combined snapshot retains the keys known at that time. Phase 1B guarantees that live operations render existing database files undecryptable upon key destruction.
   - *Limitation 2 (Hardware Remanence):* Physical flash memory wear-leveling controllers prevent operating systems from guaranteeing zero physical NAND cell remanence.

### Comprehensive History & Backup Retention Policy (Findings P2-11, P2-Backup)

1. **Live Database History Retention:**
   - `match_events` retention is capped at 10,000 rows. Excess rows trimmed at startup checkpoint.
   - `template_revisions` retention is capped at 20 historical revisions per identity.
   - Retired templates exceeding `retired_retention_days` (90 days) have their DEKs destroyed and rows purged during maintenance checkpoints.
   - Periodic WAL checkpointing maintains write-ahead log size below 10 MB.
2. **Backup Retention & Rotation Policy:**
   - Managed database backups (created via CLI backup/export commands) are capped at `backup_max_count = 5`. Creating a 6th backup automatically purges the oldest backup archive.
   - Cryptographic key revocation invalidates deleted identities in historical backups without requiring manual backup rewriting.
   - *Operational Limitation:* Backups created outside Face Core by external operator scripts (e.g., OS-level copies of `facecore.db`) are outside Face Core's process lifecycle. Operators must schedule external backup pruning in alignment with the 90-day retention policy and safeguard `KeyProvider` files separately.

---

## 8. Authenticated Export / Import Key Re-Homing Protocol (Findings P1-1, P1-3, P2-KDF)

To permit seamless migration between distinct KeyProvider environments without raw key exposure, using versioned memory-hard key derivation, outer-authenticated envelopes, and canonical 9-field manifest verification:

1. **Passphrase Key Derivation (Versioned Argon2id Memory-Hard KDF Contract):**
   - Export passphrase protection uses **Argon2id** (memory-hard KDF) with explicit versioning in the package header:
     - `kdf_algorithm`: `"argon2id"`
     - `kdf_version`: `1`
     - `kdf_salt`: 16 bytes CSPRNG
     - `kdf_memory_cost_kib`: `65536` ($64\text{ MB}$)
     - `kdf_time_cost`: `3`
     - `kdf_parallelism`: `1`
   - Derives two 256-bit sub-keys: $K_{\text{wrap}}$ (DEK wrapping key) and $K_{\text{envelope}}$ (outer archive authentication key).
   - *Fail-closed KDF contract:* If importing runtime lacks Argon2id support or encounters an unsupported KDF algorithm or version, it fails closed immediately with structured error `UnsupportedKdfError` (exit code `4`). Silent fallback to weaker KDFs is strictly forbidden.
   - If operator supplies an external high-entropy key via KeyProvider, Argon2id derivation is bypassed.
2. **Export Packaging (Source KeyProvider A):**
   - Source repository verifies complete state: `policy_profile`, full `model_manifest`, identities, revisions, active templates, retired templates, candidates across all statuses (`pending`, `promoted`, `rejected`, `expired`, `generation_retired`) with evidence logs, exemplars, and audit tombstones.
   - Inner Layer (Key Wrapping): Source `KeyProvider` un-wraps each record's DEK in memory, and encrypts it under $K_{\text{wrap}}$ using AES-256-GCM (`wrapped_dek`, `wrap_nonce`, `wrap_tag`).
   - Outer Layer (Archive-Wide Authenticated Envelope): The complete canonical package (all metadata, revision associations, candidate statuses, policy profile, model manifest, wrapped DEKs, and biometric ciphertext payloads) is sealed in an authenticated envelope (AEAD AES-256-GCM) keyed by $K_{\text{envelope}}$. The outer AAD binds `export_version`, KDF parameters, salt, and creation timestamp.
3. **Import Unpacking & Re-Homing (Destination KeyProvider B):**
   - **Step 1 (Envelope Verification):** Destination repository verifies the outer AEAD envelope MAC over the entire archive using $K_{\text{envelope}}$. Any modification to metadata, policy, candidate statuses, key associations, or payloads fails immediately with exit code `4` prior to any database write.
   - **Step 2 (Canonical 9-Field Model Manifest Compatibility Check):** Evaluates the canonical field-by-field compatibility predicate:
     1. `embedder_artifact_hash` (weights SHA-256)
     2. `detector_generation` (detector ID, weights SHA, landmark alignment geometry)
     3. `preprocessing_generation` (resize interpolation, RGB channel order, input tensor dimensions)
     4. `tensor_layout` (e.g. `NCHW`)
     5. `normalization_contract` (scale factor, mean array, std array)
     6. `embedding_dimension` (e.g. 512)
     7. `numerical_precision` (e.g. `fp32`)
     8. `quantization_type` (e.g. `none`, `int8bq`)
     9. `execution_runtime` (e.g. `onnxruntime-cpu-arm64`, ORT version / provider contract)
     - If ANY of fields (1, 4, 5, 6, 7, 8, 9) differ: Incompatible! Aborts immediately with exit code `3` (`ModelIncompatibilityError`).
     - If fields (2, 3) differ (detector/preprocessing generation changed) while embedder and runtime are compatible: Triggers Task 7 migration re-embedder over stored exemplars, or fails closed if exemplars cannot be re-aligned.
   - **Step 3 (DEK Re-Homing):** Destination KeyProvider un-wraps each DEK in memory using $K_{\text{wrap}}$. For each record, it generates a new destination-local `key_id`, re-encrypts the DEK under the destination KeyProvider master key, and stores it in destination key custody.
   - **Step 4 (Database Insert):** Destination SQLite inserts records referencing the newly created destination `key_id` values. Biometric ciphertexts are preserved without re-encryption.

---

## 9. 500-Identity Capacity & Storage Benchmark

In accordance with arbitration `d-20260911184146033747-27` item (9):

- **Target:** 500 enrolled identities with 5 active templates each (2,500 total active templates) in encrypted storage.
- **Query mechanism:** Bounded exact linear scan over active templates using vectorized NumPy operations. No ANN index; no ORM cache.
- **Latency budget (Apple M1 reference machine):**
  - Database active template fetch & decryption: p95 ≤ 15.0 ms.
  - 2,500-vector exact comparison & margin ranking: p95 ≤ 3.0 ms.
  - Total identification query overhead (excluding ONNX inference): p95 ≤ 25.0 ms.
- **Reporting:** Benchmark reports comparison time, storage fetch time, and end-to-end time separately.

---

## 10. File and Dependency Map

```text
src/facecore/
├── contracts/
│   ├── candidate.py        # CandidateTemplate, CandidateStatus, EvidenceRecord
│   ├── confirmation.py     # ConfirmationRequest, ConfirmationVerdict, ActorType
│   ├── crypto.py           # EncryptedBlob, KeyReference, KeyProviderProtocol, WrappedKey
│   ├── drift.py            # DriftMetrics, DriftStatus, DriftPolicy
│   ├── export.py           # ExportContainer, ExportMetadata, ExportManifest (full state)
│   ├── migration.py        # ModelMigrationManifest, MigrationResult, GenerationStatus
│   └── policy.py           # Extended GovernancePolicy (thresholds, weights, TTL)
├── storage/
│   ├── base.py             # PersistentRepository protocol
│   ├── cipher.py           # Authenticated cipher abstraction (AEAD)
│   ├── key_provider.py     # KeyProvider protocol, FileKeyProvider, MockKeyProvider
│   ├── sqlite_repo.py      # SQLite implementation with Journaled Tombstones & WAL
│   ├── export.py           # Authenticated export/import serializer and re-homer
│   └── migrations/         # Schema DDL (v1)
├── governance/
│   ├── candidate.py        # Candidate creation, update gate, expiry check
│   ├── corroboration.py    # Multi-event corroborator, burst suppressor
│   ├── promotion.py        # Exclusivity margin checker, promotion manager, generation gate
│   ├── utility.py          # 6-factor utility rescoring engine & tie-breaking
│   ├── eviction.py         # Capacity checker and atomic retirement manager
│   ├── lifecycle.py        # Identity create, re-enroll, rollback, delete
│   ├── migration.py        # Model generation migration & exemplar re-embedder (active + candidate)
│   └── drift.py            # Lifecycle-safe centroid tracking & drift detector
├── eval/
│   ├── replay.py           # Chronological adaptive replay harness (A/B testing)
│   ├── replay_report.py    # Replay vs 1A frozen baseline comparison table
│   └── benchmark_1b.py     # 500-identity encrypted storage & comparison benchmark
└── cli.py                  # CLI commands: identity, candidates, migration, status, --confirm-learning
```

---

## 11. Tasks

### Task 0: Land this plan and repoint project docs
- **Files:**
  - Create: `docs/plans/2026-09-12-phase-1b-implementation-plan.md`
  - Modify: `README.md` (repository map); `docs/PROJECT-STATE.md` ("Current Status" & "Next Session")
- **Interfaces:** Consumes spec §§9–12, 14, Phase-1A closeout state (`98b202a`), arbitration `d-20260911184146033747-27`, and rework decisions `d-20260911185748092685-28`, `d-20260911190712794016-29`, `d-20260911191746908253-30`, `d-20260911193316595438-32`.
- **Acceptance:**
  - Repository map in `README.md` lists `docs/plans/2026-09-12-phase-1b-implementation-plan.md`.
  - `docs/PROJECT-STATE.md` records Phase-1B plan drafted and pending operator review, with implementation locked until ADR 0006 go/no-go.
  - Zero Python source code added.
- **Test-first evidence:**
  - Failing case: Repository map diff check fails when plan file exists on disk but is absent from `README.md`.
  - RED: `diff <(sed -n '/```text/,/```/p' README.md | grep 'phase-1b') <(echo "    └── 2026-09-12-phase-1b-implementation-plan.md")` → returns exit code 1.
  - Minimal behavior: Add plan entry to `README.md` and update `docs/PROJECT-STATE.md`.
  - GREEN: Same diff command returns 0.
  - Project verification: `git diff --check origin/main...HEAD` clean; markdown link checks clean.

### Task 1: Governance & Lifecycle Contracts Expansion
- **Depends on:** Prerequisite P0 (ADR 0006 Go/No-Go Approval).
- **Files:**
  - Create: `src/facecore/contracts/candidate.py`, `src/facecore/contracts/confirmation.py`, `src/facecore/contracts/crypto.py`, `src/facecore/contracts/drift.py`, `src/facecore/contracts/export.py`, `src/facecore/contracts/migration.py`
  - Modify: `src/facecore/contracts/policy.py`, `src/facecore/contracts/template.py`, `src/facecore/contracts/result.py`
  - Test: `tests/contracts/test_governance_contracts.py`
- **Interfaces:** Produces `CandidateTemplate`, `ConfirmationRequest`, `EncryptedBlob`, `KeyProviderProtocol`, `DriftMetrics`, `ExportContainer`, `ModelMigrationManifest`, `GovernancePolicy` (`governance_policy_version: 1`).
- **Acceptance:**
  - `FaceTemplate` models `encrypted_exemplar`, crop bounding box, landmarks, and `key_id`.
  - `CandidateTemplate` models `additional_corroboration_count` initialized to `0` at creation.
  - `ExportContainer` schema models complete governance state: active templates, retired templates, candidates in all statuses (`pending`, `promoted`, `rejected`, `expired`, `generation_retired`) with evidence logs, exemplars, revisions, full policy profile, canonical 9-field model manifest, outer authenticated envelope, and audit tombstones.
  - `GovernancePolicy` marks all operational values as provisional pending Operator Decision Manifest.
- **Test-first evidence:**
  - Failing case: Constructing `CandidateTemplate` without `additional_corroboration_count = 0` or constructing `FaceTemplate` without exemplar fields raises validation error.
  - RED: `pytest tests/contracts/test_governance_contracts.py -q` → `ModuleNotFoundError: No module named 'facecore.contracts.candidate'`.
  - Minimal behavior: Implement dataclass models, enums, JSON encoders, and validation predicates.
  - GREEN: `pytest tests/contracts/test_governance_contracts.py -q` → all pass.
  - Project verification: `mypy src tests` and `ruff check src tests` clean.

### Task 2: Encrypted Storage Repository & KeyProvider Implementation
- **Depends on:** Prerequisite P0, Spike S1B decision manifest, Operator Decision Manifest (Key Custody, Backup Retention Ceiling), Task 1.
- **Files:**
  - Create: `src/facecore/storage/cipher.py`, `src/facecore/storage/key_provider.py`, `src/facecore/storage/sqlite_repo.py`, `src/facecore/storage/migrations/v1.sql`
  - Test: `tests/storage/test_cipher.py`, `tests/storage/test_key_provider.py`, `tests/storage/test_sqlite_repo.py`, `tests/storage/test_stress.py`
- **Interfaces:** Produces `KeyProvider` protocol, `FileKeyProvider`, `SQLitePersistentRepository` implementing `PersistentRepository`.
- **Acceptance:**
  - Implements storage and cipher scheme selected in Spike S1B manifest (AEAD field encryption with unique nonces).
  - DEKs are managed exclusively by `KeyProvider`; SQLite stores only opaque `key_id` references.
  - Deletion and purge follow the Journaled Tombstone Protocol (Section 7).
  - Recovery test: Simulating crash when keys were already destroyed before crash correctly reconciles tombstones, completes SQLite purge, and truncates WAL.
  - Comprehensive crash injection tests: (a) crash before key destruction, (b) crash after each key in multi-key batch, (c) crash after all keys destroyed but before tombstone update, (d) crash after tombstone updated, (e) checkpoint failure.
  - Database corruption, missing key, or invalid auth tag raises `StoreCorruptionError` or `KeyNotFoundError` (exit code 4).
  - 10,000-event + 20-revision + 5-backup rotation retention stress test passes with stable query latency.
- **Test-first evidence:**
  - Failing case: Simulating crash immediately after key destruction leaves orphaned unusable rows that fail startup reconciliation.
  - RED: `pytest tests/storage/test_sqlite_repo.py -k "test_tombstone_reconciliation_after_crash" -q` → `Failed: DID NOT RECONCILE TOMBSTONES`.
  - Minimal behavior: Implement SQLite schema, AEAD cipher, external KeyProvider DEK management, journaled tombstone reconciliation with key-already-destroyed handling, and fail-closed error handling.
  - GREEN: `pytest tests/storage/ -q` → all pass.
  - Project verification: `mypy src` clean; memory leak checks pass under 10k iterations.

### Task 3: Confirmation-Gated Shadow Candidate Pipeline
- **Depends on:** Prerequisite P0, Task 1, Task 2, Operator Decision Manifest (Confirmation Semantics).
- **Files:**
  - Create: `src/facecore/governance/candidate.py`
  - Modify: `src/facecore/cli.py` (`--confirm-learning` handler)
  - Test: `tests/governance/test_candidate_pipeline.py`
- **Interfaces:** Produces `CandidatePipeline.evaluate_observation(result, decoded_face, confirmation) -> CandidateDecision`.
- **Acceptance:**
  - Candidate creation strictly requires: (a) match score $\ge$ `candidate_update_threshold` (provisional 0.88), (b) quality accepted, (c) explicit `correct` confirmation.
  - Newly created candidate initializes `additional_corroboration_count = 0` and status `pending`.
  - `not_me`, cancellation, timeout, or EOF leaves zero records in `candidate_templates` and leaves disk unmodified.
  - Pending observation held strictly in memory; process kill leaves zero unencrypted artifacts.
- **Test-first evidence:**
  - Failing case: Supplying `not_me` confirmation or observation below update threshold creates a candidate row.
  - RED: `pytest tests/governance/test_candidate_pipeline.py -k "test_not_me_creates_no_candidate" -q` → `assert 1 == 0`.
  - Minimal behavior: Implement strict gating, memory-only pending buffers, and candidate insertion with `additional_corroboration_count = 0`.
  - GREEN: `pytest tests/governance/test_candidate_pipeline.py -q` → all pass.
  - Project verification: Worktree clean; zero uncommitted SQLite files.

### Task 4: Corroboration Engine, Promotion Exclusivity, and Utility Eviction
- **Depends on:** Prerequisite P0, Task 3, Operator Decision Manifest (Promotion Margin & Utility Weights).
- **Files:**
  - Create: `src/facecore/governance/corroboration.py`, `src/facecore/governance/promotion.py`, `src/facecore/governance/utility.py`, `src/facecore/governance/eviction.py`
  - Test: `tests/governance/test_corroboration.py`, `tests/governance/test_promotion.py`, `tests/governance/test_utility_eviction.py`
- **Interfaces:** Produces `CorroborationEngine`, `PromotionManager`, `UtilityRescorer`, `EvictionManager`.
- **Acceptance:**
  - Probes within `burst_suppression_min_interval_secs` (60s) or duplicate image hashes increment zero corroboration counts.
  - Promotion strictly requires: (a) `additional_corroboration_count >= additional_corroboration_min_events` (1), (b) cross-identity exclusivity margin $\ge$ `promotion_margin` (0.12), (c) candidate generation matches active model generation (`candidate.generation_id == current_model_generation`).
  - Single seed confirmation remains `pending`; only a second independent event can promote it.
  - When promoted, active template copies the candidate's `encrypted_exemplar` and crop metadata into `face_templates`.
  - Active bank capacity (5) triggers 6-factor utility rescoring using exact Section 6 formulas. Output score is strictly in $[0.0, 1.0]$.
  - Initial template enjoys no permanent exemption and retires under the same utility policy.
  - Deterministic tie-breaking: timestamp followed by `template_id`.
- **Test-first evidence:**
  - Failing case: Candidate with old generation G1 promotes under active generation G2; or utility score falls outside $[0.0, 1.0]$.
  - RED: `pytest tests/governance/test_promotion.py -k "test_old_generation_candidate_refused_promotion" -q` → `assert candidate.status == 'pending' (got 'promoted')`.
  - Minimal behavior: Implement generation check, `additional_corroboration_count` checks, exclusivity query, clamped 6-factor formulas, exemplar propagation, and atomic retirement.
  - GREEN: `pytest tests/governance/test_promotion.py tests/governance/test_utility_eviction.py -q` → all pass.
  - Project verification: Boundary tests pass for anti-parallel ($\cos = -1 \implies U_C = 1.0$), duplicate ($\cos = 1 \implies U_{Red} = 1.0$), empty-bank, and all-penalty maximum suppression resulting in clean $0.0$.

### Task 5: Identity Lifecycle CLI & Recovery Operations
- **Depends on:** Prerequisite P0, Task 2, Task 4, Operator Decision Manifest (Rollback Depth, Retention TTL, Actor Taxonomy).
- **Files:**
  - Create: `src/facecore/governance/lifecycle.py`
  - Modify: `src/facecore/cli.py` (add `identity add/show/re-enroll/rollback/delete`, `candidates list/reject`, `status`)
  - Test: `tests/cli/test_identity_lifecycle.py`, `tests/storage/test_crypto_erasure.py`
- **Interfaces:** CLI subcommands emitting stable JSON per spec §11.
- **Acceptance:**
  - `identity add` is create-only; duplicate ID fails without mutation (exit code 4). Stores initial photo's `encrypted_exemplar` in `face_templates`.
  - `identity re-enroll` atomically retires active templates and installs new photo as sole active template with new exemplar; resets drift reference.
  - `identity rollback --to-revision <n>` restores revision $n$, retiring newer templates; refuses if revision exceeds `rollback_max_depth`.
  - `identity delete` executes cryptographic erasure via the Journaled Tombstone Protocol: inserts deletion tombstone, destroys DEKs via `key_provider.destroy_identity_keys(identity_id)`, overwrites template rows, anonymizes match events, and executes `PRAGMA wal_checkpoint(TRUNCATE)`.
  - Decryption test proves residual ciphertext blocks in backups, WAL, or free pages cannot be decrypted once DEKs are erased in KeyProvider.
- **Test-first evidence:**
  - Failing case: Attempting to decrypt backup ciphertext using KeyProvider after `identity delete` succeeds.
  - RED: `pytest tests/storage/test_crypto_erasure.py -k "test_backup_ciphertext_undecryptable_after_dek_destruction" -q` → `Failed: DEK was not destroyed`.
  - Minimal behavior: Implement KeyProvider DEK destruction, journaled tombstone delete, checkpoint fail-closed handling, and CLI commands.
  - GREEN: `pytest tests/cli/test_identity_lifecycle.py tests/storage/test_crypto_erasure.py -q` → all pass.
  - Project verification: Correct exit codes: 0 normal, 2 bad input, 4 store error.

### Task 6: Encrypted Export / Import with Key Re-Homing & Canonical Manifest Compatibility
- **Depends on:** Prerequisite P0, Task 5, Operator Decision Manifest (Export Key / Passphrase Model).
- **Files:**
  - Create: `src/facecore/storage/export.py`
  - Modify: `src/facecore/cli.py` (add `export`, `import`)
  - Test: `tests/storage/test_export_import.py`
- **Interfaces:** Produces `export_identities(path, passphrase) -> ExportManifest`, `import_identities(path, passphrase, dest_key_provider) -> ImportResult`.
- **Acceptance:**
  - Export archive serializes complete governance state: `policy_profile`, full `model_manifest`, identities, template revisions, active templates, retired templates, candidates across all statuses (`pending`, `promoted`, `rejected`, `expired`, `generation_retired`) with evidence logs and expiry, exemplars, and anonymized match events.
  - Implements the Authenticated Key Re-Homing Protocol (Section 8) using pinned Argon2id memory-hard KDF ($64\text{ MB}, 3\text{ iterations}$) and an outer AEAD envelope. Outer MAC is verified before unpacking; any metadata or association tamper fails immediately with exit code `4`. If runtime lacks Argon2id or encounters unsupported version, fails closed with `UnsupportedKdfError` (exit code 4).
  - Canonical 9-Field Manifest Compatibility Predicate: Verifies all nine fields identically to Section 8: (1) embedder artifact hash, (2) detector generation, (3) preprocessing generation, (4) tensor layout, (5) normalization contract (scale/mean/std), (6) embedding dimension, (7) numerical precision, (8) quantization type, and (9) execution runtime/provider contract. If ANY of (1, 4, 5, 6, 7, 8, 9) differ, import fails closed with exit code `3` (`ModelIncompatibilityError`) with zero database writes. (Finding P1-1).
  - Round-trip test: Exporting an identity from KeyProvider A with modified policy, candidates across all statuses, and retired templates; importing into KeyProvider B; asserting that custom policy is restored, all candidate statuses and evidence logs are preserved, and subsequent rollback or candidate rejection functions identically.
- **Test-first evidence:**
  - Failing case: Importing archive into KeyProvider B with mismatched execution runtime or mismatched tensor layout succeeds; or importing archive with unsupported KDF algorithm succeeds. (Findings P1-1, P2-KDF).
  - RED: `pytest tests/storage/test_export_import.py -k "test_mismatched_runtime_contract_refused" -q` → `Failed: DID NOT RAISE ModelIncompatibilityError`.
  - Minimal behavior: Implement Argon2id KDF verification, outer AEAD envelope verification, canonical 9-field manifest compatibility predicate, key re-homing, and transactional import unpacking.
  - GREEN: `pytest tests/storage/test_export_import.py -q` → all pass.
  - Project verification: Tamper tests pass for metadata classes, nonce/key associations, unsupported KDF formats, and each manifest field individually.

### Task 7: Model Generation Migration & Exemplar Re-Embedding
- **Depends on:** Prerequisite P0, Task 5, Task 6, Operator Decision Manifest (`exemplar_margin`). (Finding P1-3).
- **Files:**
  - Create: `src/facecore/governance/migration.py`
  - Modify: `src/facecore/cli.py` (add `migration migrate-model`, `migration status`)
  - Test: `tests/governance/test_model_migration.py`
- **Interfaces:** Produces `ModelMigrationManager.migrate_generation(new_manifest) -> MigrationReport`.
- **Acceptance:**
  - When runtime model/preprocessing upgrades to a new generation ($G_1 \to G_2$), migration re-embeds stored `EncryptedExemplar` records from active, retired, and candidate templates atomically.
  - Candidate Migration All-Status Rule (Finding P1-2):
    - `pending`: re-embeds exemplar to $G_2$; if geometry fails, transitions status to `rejected` with reason `model_generation_retired` and blocks promotion.
    - `promoted`: transitions status to `generation_retired` (terminal archived state) with `generation_id = 'G1'`; active template in `face_templates` is migrated separately; permanently excluded from future matching/promotion.
    - `rejected` & `expired`: historical terminal states preserved with `generation_id = 'G1'`; permanently excluded from matching/promotion.
  - If active template exemplar crop geometry or landmark alignment cannot be reproduced under new detector/aligner, identity status transitions to `re_enrollment_required`.
  - Identification queries against `re_enrollment_required` identities immediately return `review` with reason code `identity_re_enrollment_required`.
  - Cross-generation comparison assertion: Tests verify that embeddings from different generations are never compared directly.
- **Test-first evidence:**
  - Failing case: Candidate in status `promoted` retains G1 embedding and is permitted to match against G2 templates; or mid-migration crash leaves partial state. (Finding P1-2).
  - RED: `pytest tests/governance/test_model_migration.py -k "test_promoted_candidate_transitions_to_generation_retired" -q` → `assert candidate.status == 'generation_retired' (got 'promoted')`.
  - Minimal behavior: Implement exemplar re-embedder for active and candidate templates, all-status candidate lifecycle rules, geometry validation predicate, atomic revision creation, and status transitions.
  - GREEN: `pytest tests/governance/test_model_migration.py -q` → all pass.
  - Project verification: Tests pass across all four candidate statuses (`pending`, `promoted`, `rejected`, `expired`) and mid-migration crash cleanly rolls back.

### Task 8: Long-Horizon Drift Indicators and Bounded Response Policy
- **Depends on:** Prerequisite P0, Task 4, Operator Decision Manifest (Drift Bounds).
- **Files:**
  - Create: `src/facecore/governance/drift.py`
  - Modify: `src/facecore/contracts/drift.py`, `src/facecore/policy/identify.py`
  - Test: `tests/governance/test_drift_policy.py`
- **Interfaces:** Produces `DriftDetector.measure_identity_drift(identity_id) -> DriftMetrics`.
- **Acceptance:**
  - Tracks active template centroid cosine shift from initial reference anchor.
  - Lifecycle-safe reference: Anchor reference expires when initial template is evicted and pruned after 90 days; drift detection smoothly transitions to rolling centroid diversity, never maintaining an unmanaged permanent embedding.
  - `identity re-enroll` resets the drift reference anchor.
  - When centroid shift exceeds `drift_max_centroid_shift` (provisional 0.20):
    1. Identification status downgrades from `matched` to `review` with reason code `drift_boundary_exceeded`.
    2. Identity status sets `re_enrollment_required`.
    3. Automatic candidate creation and promotion are suspended for that identity.
- **Test-first evidence:**
  - Failing case: Initial template eviction crashes drift calculator, or drifted template continues returning `matched`.
  - RED: `pytest tests/governance/test_drift_policy.py -k "test_drift_lifecycle_after_initial_eviction" -q` → `AttributeError: 'NoneType' object has no attribute 'embedding'`.
  - Minimal behavior: Implement lifecycle-safe reference handling, centroid calculation, and policy hook.
  - GREEN: `pytest tests/governance/test_drift_policy.py -q` → all pass.
  - Project verification: `mypy src` clean.

### Task 9: Chronological Adaptive Replay Harness & Temporal-Leakage Prevention
- **Depends on:** Prerequisite P0, Spike S2B, Task 4, Task 8.
- **Files:**
  - Create: `src/facecore/eval/replay.py`
  - Modify: `src/facecore/eval/session.py`
  - Test: `tests/eval/test_replay.py`, `tests/eval/test_temporal_leakage.py`
- **Interfaces:** Produces `ChronologicalReplayHarness.run_replay(manifest, model_manifest, policy) -> ReplayExecutionSummary`.
- **Acceptance:**
  - Processes events ordered by composite key `(timestamp, sequence_number, event_uuid, source_sha256)`. Equal timestamps supported; backward sequence rejected.
  - Ground-truth labels provide supervision strictly inside the test harness.
  - **A/B Replay Temporal-Leakage Test:** Executes replay stream $1..N$, and compares against replay of stream $1..N$ followed by future suffix $N+1..M$. Asserts that decision output, scores, and template state at event $N$ are byte-identical.
  - Pre-seeded future database rows test: Verifies replay query layer cannot observe rows with `created_at > t_N`.
- **Test-first evidence:**
  - Failing case: Appending future probe events alters the identification score or candidate decision at event $N$.
  - RED: `pytest tests/eval/test_temporal_leakage.py -k "test_ab_replay_future_suffix_isolation" -q` → `AssertionError: Decision at event N differed when future suffix was present`.
  - Minimal behavior: Implement composite key ordering, snapshot query bounds, and A/B verification fixtures.
  - GREEN: `pytest tests/eval/test_replay.py tests/eval/test_temporal_leakage.py -q` → all pass.
  - Project verification: Deterministic execution logs across repeated runs.

### Task 10: Phase-1B Evaluation Run & Baseline Comparison Report
- **Depends on:** Prerequisite P0, Task 9.
- **Files:**
  - Create: `src/facecore/eval/replay_report.py`
  - Modify: `facecore.sh` (add `replay` command)
  - Test: `tests/eval/test_replay_report.py`
- **Interfaces:** Produces `./facecore.sh replay --corpus <manifest> --report reports/phase-1b-replay.md`.
- **Acceptance:**
  - Executes chronological replay using Candidate A (SFace Pair 1 provisional) against consented P1 corpus.
  - Emits side-by-side comparison table: Phase-1A frozen baseline vs Phase-1B adaptive bank (match/review/unknown counts, candidate creations, promotions, evictions, drift metrics, exact denominators). Rate formatting forbidden below $N < 30$.
  - **Conditional Closeout Rule:** If real weights or real multi-timestamp probes remain blocked by operator dual gates:
    - Report evaluates synthetic adversarial stream only.
    - Report marks real replay `blocked-with-reason` and explicitly leaves model selection gate **OPEN**.
    - Report is labeled `partial governance validation`, NOT Phase-1B completion.
  - Redaction check: Scan asserts zero raw embeddings, face crops, or local paths.
- **Test-first evidence:**
  - Failing case: Real replay is blocked but report claims model selection gate is closed or Phase 1B is complete.
  - RED: `pytest tests/eval/test_replay_report.py -k "test_blocked_real_replay_leaves_gate_open" -q` → `AssertionError: Model selection gate must remain OPEN when real replay is blocked`.
  - Minimal behavior: Implement baseline delta calculator, exact denominator formatting, conditional gate logic, and redaction scanner.
  - GREEN: `pytest tests/eval/test_replay_report.py -q` → all pass.
  - Project verification: End-to-end replay passes on synthetic test corpus.

### Task 11: 500-Identity Capacity Benchmark & Conditional Docs Closeout
- **Depends on:** Prerequisite P0, Task 5, Task 10, Operator Decision Manifest (Backup Retention Ceiling). (Finding P1-3).
- **Files:**
  - Create: `src/facecore/eval/benchmark_1b.py`
  - Modify: `README.md`, `docs/PROJECT-STATE.md`
  - Test: `tests/eval/test_benchmark_1b.py`
- **Interfaces:** Produces 500-identity capacity report section in `reports/phase-1b-replay.md`.
- **Acceptance:**
  - Benchmarks 500 synthetic identities (2,500 vectors) in encrypted storage: comparison p95 $\le$ 3.0 ms; fetch & decrypt p95 $\le$ 15.0 ms.
  - Validates CLI fail-closed integration when uninitialized or tampered.
  - Verifies managed backup retention ceiling (`backup_max_count = 5`, derived from ratified Operator Decision Manifest).
  - **Conditional Docs Closeout:**
    - If real replay was completed: `docs/PROJECT-STATE.md` records Phase-1B completion and requests operator review to close ADR 0006.
    - If real replay was blocked: `docs/PROJECT-STATE.md` records Phase-1B Governance Engine delivered (provisional) with model selection gate remaining OPEN pending weights/corpus.
  - `README.md` repository map matches disk exactly.
- **Test-first evidence:**
  - Failing case: Benchmark fails to report comparison time separately from storage fetch time, or backup retention exceeds ceiling.
  - RED: `pytest tests/eval/test_benchmark_1b.py -q` → `KeyError: 'sqlite_fetch_p95_ms'`.
  - Minimal behavior: Implement dual-stage timing probes, run benchmark, update documentation conditionally.
  - GREEN: `pytest tests/eval/test_benchmark_1b.py -q` → all pass.
  - Project verification: Map diff check clean; `ruff check` and `mypy` clean.

---

## 12. PR Boundaries

In accordance with arbitration `d-20260911184146033747-27` and rework decisions `d-20260911185748092685-28`, `d-20260911190712794016-29`, `d-20260911191746908253-30`, and `d-20260911193316595438-32`, implementation is partitioned into **7 distinct PRs**.

**Hard Gate:** PR-C through PR-G require prior approval of Prerequisite P0 (ADR 0006 Go/No-Go Decision) and the relevant Operator Decision Manifest items. PR-A and PR-B are docs-only and exempt.

| PR | Included Tasks | Scope & Boundaries | Independent Acceptance Criteria |
|---|---|---|---|
| **PR-A** | Task 0 | Plan only (docs-only). Land `docs/plans/2026-09-12-phase-1b-implementation-plan.md`, update `README.md` map and `docs/PROJECT-STATE.md`. No Python source code. | Map matches disk exactly; `PROJECT-STATE.md` reflects 1B planning status; CI passes. ADR 0006 gate remains intact. |
| **PR-B** | Spikes S1B, S2B | Research manifests (docs-only). Land storage/crypto manifest and model/corpus readiness reports. | Concrete AAD specifications, nonce rules, KeyProvider seam, DEK placement, pinned Argon2id KDF dependency, journaled tombstone protocol, model SHAs, and corpus chronology recorded. |
| **PR-C** | Task 1, Task 2 | Foundation: Contracts, `KeyProvider`, AEAD cipher, encrypted repository, and Journaled Tombstone Protocol. **Requires P0 approval & Operator Key Custody/Backup Manifest.** | All contracts versioned (`schema_version: 1`, `governance_policy_version: 1`). SQLite ACID transactions, tampering detection, KeyProvider DEK destruction tests, journaled tombstone crash reconciliation (including key-already-absent path), and 10k-event + 5-backup stress test pass. Zero CLI mutation. |
| **PR-D** | Task 3, Task 4 | Adaptive Core: Confirmation-gated candidate pipeline, corroboration, promotion exclusivity, 6-factor utility rescoring, exemplar propagation, and capacity eviction. **Requires P0 approval & Operator Confirmation/Promotion Manifest.** | Candidate creation strictly gated by `correct` (`additional_corroboration_count = 0`); burst suppression active; cross-identity margin enforced; lowest-utility template evicted on capacity; active template retains exemplar; generation-mismatched candidate blocked from promotion; utility scores strictly in $[0.0, 1.0]$ across all boundaries; atomic revisions committed. Modules and fixtures split between T3 and T4; T3 tests pass independently of T4. |
| **PR-E** | Task 5, Task 6, Task 7 | Administrative CLI, Recovery & Generation Migration: `identity add/show/re-enroll/rollback/delete`, `candidates list/reject`, outer-authenticated export/import key re-homing with Argon2id, exemplar re-embedding. **Requires P0 approval & Operator Retention/Actor/Export/Margin Manifest.** | `identity add` collision blocked; `re-enroll` and `rollback` atomic; `delete` proves KeyProvider DEK destruction, journaled tombstones, and backup unrecoverability; export/import verifies outer AEAD envelope and canonical 9-field ModelManifest compatibility; model migration re-embeds active and candidate exemplars across all statuses and handles geometry mismatch fallback. Failure matrices split across tasks. |
| **PR-F** | Task 8, Task 9 | Drift Monitoring & Replay Engine: Lifecycle-safe drift metrics, bounded response policy, chronological replay harness, A/B temporal-leakage isolation tests. **Requires P0 approval & Operator Drift Bounds Manifest.** | Drift threshold breach triggers `review` and `re_enrollment_required`; replay executes in strict composite key order; A/B temporal leakage tests prove future suffix does not alter event $N$ decisions. |
| **PR-G** | Task 10, Task 11 | Evaluation, Benchmark & Conditional Closeout: SFace Pair 1 replay run vs 1A frozen baseline, 500-identity capacity benchmark, conditional `project-docs-maintain` pass. **Requires P0 approval & Operator Backup Retention Manifest.** | Replay report emitted with exact denominators and delta comparison; if real replay blocked, selection gate stays OPEN and docs reflect partial status; 500-identity benchmark meets latency budget; backup retention ceiling enforced; redaction check clean. |

**Partial Red Rule:** Within multi-task PRs (PR-D, PR-E), tests are partitioned into distinct test files (`test_candidate_pipeline.py` vs `test_promotion.py`; `test_identity_lifecycle.py` vs `test_model_migration.py`). A failing test under Task 4 or Task 7 does not prevent Task 3 or Task 5 from demonstrating independent functional correctness.

---

## 13. Out of Scope for Phase 1B

Restating explicit boundaries to prevent scope creep (spec §§9–12, 14; arbitration item 10):

- **No camera hardware or live video streams:** Static image decode only; zero RTSP, V4L2, AVFoundation, or continuous video tracking.
- **No liveness or anti-replay presentation attack detection:** Output is strictly `matched`, never `authenticated`.
- **No Android or iOS native code:** All code is written in cross-platform Python on macOS; mobile platform integration is Phase 2.
- **No mobile Keychain / KeyStore implementation:** Phase 1B delivers the `KeyProvider` interface and a macOS local file provider only.
- **No attendance or business rules:** Attendance policies, payroll integrations, class schedules, and authorization logic remain outside Face Core.
- **No multi-face search:** Every image must contain exactly one usable face; multi-face photo discovery is a separate future roadmap item.
- **No automated key rotation:** Cryptographic key rotation without downtime requires external key managers and is deferred to Phase 2.
- **No approximate nearest neighbor (ANN) vector indices:** Capacity (500 identities) is fully satisfied by exact linear scan.

---

## 14. Highest Risks & Stop Conditions

In accordance with arbitration `d-20260911184146033747-27` and rework decisions `d-20260911185748092685-28`, `d-20260911190712794016-29`, `d-20260911191746908253-30`, and `d-20260911193316595438-32`, the plan defines five concrete risks with pre-planned mitigations and **three non-negotiable STOP conditions**:

### Explicit STOP Conditions:
1. **STOP Condition 1 (Unsupervised Learning Breach):** If any code path, test, or replay execution allows an unconfirmed observation (or one confirmed with `not_me`, canceled, or timed out) to create an active template, candidate, or persistent biometric artifact, or allows a candidate seed event to promote itself without an additional independent corroboration event, or allows an old-generation candidate to promote under a new model generation → **HALT IMMEDIATELY**. Reviewer must reject the PR.
2. **STOP Condition 2 (Fail-Open or Plaintext Storage Breach):** If any unit test, failure injection, or manual inspection reveals that storage corruption, key absence, unsupported KDF, or model incompatibility fails open (exits 0 instead of 3/4), or leaves unencrypted embeddings, face crops, or un-erased DEKs on disk, or allows unauthenticated export archive tampering → **HALT IMMEDIATELY**. Do not proceed until cryptographic fail-closed behavior is restored.
3. **STOP Condition 3 (Gate Weakening or Premature Gate Closure):** If ONNX weights or multi-timestamp real evaluation probes are missing under operator dual-gate rules, the team must **NOT** weaken the gates, fabricate data, or close the model selection gate. The replay task must mark real replay `blocked-with-reason`, evaluate synthetic streams only, keep the selection gate OPEN, and refuse to claim full Phase-1B completion.

### Operational Risks & Mitigations:
1. **Risk: Cumulative template drift and self-poisoning over long horizons.**
   - *Mitigation:* Spec §9 line 198 acknowledges that similarity-based gates share the same embedder and cannot eliminate drift. Task 8 bounds this risk by measuring centroid shift from a lifecycle-safe reference anchor, automatically halting candidate promotion and triggering `re_enrollment_required` when boundaries are breached.
2. **Risk: ONNX weights download blocked by provenance concerns.**
   - *Mitigation:* Decision `d-20260911174928737696-16` strictly enforces the dual gate. If operator approval is withheld, Task 9 and Task 10 execute against synthetic adversarial vectors, proving governance mechanics without violating licensing boundaries, leaving selection gate OPEN.
3. **Risk: SQLite WAL locking and concurrency issues during recovery operations.**
   - *Mitigation:* Spike S1B standardizes on WAL mode with busy timeouts and explicit `BEGIN IMMEDIATE` transactions. Unit tests verify crash consistency and atomic rollback under mid-transaction process kills.
4. **Risk: Over-interpreting synthetic 500-identity benchmarks.**
   - *Mitigation:* Reports and code documentation explicitly state that synthetic benchmarks validate latency, memory, and database scalability only, and provide zero evidence regarding false acceptances, margin distributions, or recognition accuracy.
5. **Risk: Temporal leakage in chronological replay invalidating baseline comparisons.**
   - *Mitigation:* Task 9 implements an adversarial A/B replay test suite comparing execution with and without future suffixes, proving that future observations cannot alter past decisions.
