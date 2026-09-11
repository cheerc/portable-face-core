# Phase 1B Implementation Plan

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

**Goal:** Extend the verified in-memory Phase-1A face recognition pipeline into a production-grade on-device governance engine: persistent encrypted identity and candidate storage, `KeyProvider` abstraction, confirmation-gated shadow candidate learning, multi-event corroboration, utility-driven bounded template bank with eviction, atomic revisions and identity recovery CLI (add, show, re-enroll, rollback, delete, candidate reject), encrypted export/import, long-horizon drift indicators with bounded response, and a chronological adaptive replay harness with strict temporal-leakage prevention that measures adaptive behavior against the frozen Phase-1A baseline.

**Tech stack:** Python, ONNX Runtime, NumPy, Pillow, SQLite (standard library), AES-GCM (via `cryptography` AEAD), pytest, ruff, mypy. Pure on-device offline execution; zero network access; zero external ORM or complex C extensions.

---

## 1. Frozen Source Manifest & Freshness Verification

In accordance with arbitration `d-20260911184146033747-27` item (8), Phase-1B implementation plan freezes against immutable source artifacts. Every subsequent implementation dispatch under Phase 1B must verify these identities before beginning mutation:

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

**Dispatch verification rule:** If the provider `main` tip, relevant blob SHAs, or active governing decisions differ from this manifest during task intake, the implementer must halt immediately, report the drift to the Lead, and await explicit re-base instruction.

---

## 2. Global Constraints & Non-Negotiable Boundaries

These rules govern all Phase-1B implementation work without exception (spec §§9–12, 14; arbitration item 10):

1. **Strictly offline operation:** Network availability or latency must never alter thresholds, execution paths, or recognition results.
2. **Zero plaintext biometric data at rest:** All face crops (`EncryptedExemplar`) and embeddings (`FaceTemplate`, `CandidateTemplate`) stored on disk must be encrypted at rest using authenticated encryption (AEAD). Unencrypted biometric blobs must never touch the filesystem.
3. **Zero-disk before confirmation:** During `--confirm-learning`, observation frames, crops, and embeddings remain strictly in process memory. Cancellation, EOF, timeout, process interruption, or `not_me` exits immediately without writing any candidate or biometric data to disk (spec §9 line 179).
4. **No `authenticated` state:** Recognition results are strictly constrained to `matched`, `review`, `unknown`, or `invalid_input` (spec §11 lines 237, 270–292). Self-confirmation is human supervision for candidate generation, never cryptographic or identity authentication (spec §9 lines 194–195).
5. **Strict temporal-leakage prevention:** In chronological replay and evaluation, the decision state and template bank at event $N$ must never read or depend upon any template revision, candidate, or observation timestamped $> N$ (spec §14 line 390).
6. **Fail-closed error handling:** Storage corruption, key unavailability, model incompatibility, or tampering must fail closed and emit structured exit codes: undecodable input exits `2`, model mismatch exits `3`, storage/key failure exits `4`, invalid configuration exits `5`, and internal error exits `7` (spec §11 lines 298–299).
7. **Single-face boundary maintained:** Exactly one usable face per registration and probe input. Zero-face or multi-face inputs return `invalid_input` with reason codes and never create or alter identity records (spec §7 lines 153–154).
8. **Real biometric assets remain outside Git:** Real face images, consented evaluation galleries, decrypted databases, and reports containing per-probe identifiable details are denied by `.gitignore` and must never be committed.

---

## 3. Evidence and Open Hypotheses (Layered Premises)

In accordance with arbitration `d-20260911184146033747-27` items (7) and (10), project premises are strictly partitioned into **Operator-level** (business, legal, ethics, product authority) and **Commander/Lead-level** (architecture, interface, protocol authority):

### Operator-level Premises (Blocked until Operator Decision)

1. **Formal Phase-1B Go/No-Go Decision (ADR 0006 Gate):**
   - *Status:* Pending operator review of this plan and Phase-1A evidence (`d-20260911181317619341-25`).
   - *Resolution:* Operator records go/no-go decision. Implementation tasks remain blocked until approved.
2. **ONNX Weights Re-download Authorization (Dual Gate):**
   - *Status:* Worktree release removed local weights; S1 license is CLEAR but SFace #313 provenance is open (`d-20260911174928737696-16`).
   - *Resolution:* Operator explicitly approves downloading and caching weights under `models/` for Phase-1B evaluation.
3. **P1 Corpus Extension & Chronological Probe Supply:**
   - *Status:* Phase 1A had 1 usable target probe (`PROJECT-STATE.md`). 1B requires chronological replay.
   - *Resolution:* Operator provides multi-timestamp consented probe sequences, OR formally authorizes synthetic temporal streams as the primary governance verification vehicle while real replay is labeled `blocked` with open selection gate (`d-20260911181002229319-23`).
4. **Exemplar Crop Margin & Storage Authorization (`exemplar_margin`):**
   - *Status:* Spec §10 line 222 defines `EncryptedExemplar` as a bounded pre-alignment region plus `exemplar_margin`.
   - *Resolution:* Operator decides whether Phase 1B retains minimal pre-alignment crop (`margin: 0.0`) to minimize biometric storage, or retains a moderate margin (e.g., `0.15`) for hypothetical future re-alignment across detector revisions. (Default recommendation: minimal `margin: 0.0`).
5. **Promotion-Confirmation Semantics (Arbitration Item 4):**
   - *Status:* Spec §9 line 176 requires explicit `correct` confirmation for candidate creation. Spec is ambiguous on whether an already-confirmed, corroborated candidate requires an additional operator action to promote, or auto-promotes deterministically.
   - *Resolution:* Operator clarifies: Candidate creation requires explicit `correct`; promotion from candidate to active template is deterministic and automated once corroboration and exclusivity gates are met.
6. **Retention TTL and Rollback Revision Depth Limit:**
   - *Status:* Spec §9 line 211 states retired templates remain available for a limited rollback policy and are deleted after retention expiry.
   - *Resolution:* Operator sets parameters: Maximum rollback depth = 5 revisions; retired template retention TTL = 90 days.
7. **Key Custody Model:**
   - *Status:* Spec §10 line 227 mandates keys stay outside DB via `KeyProvider`.
   - *Resolution:* Operator approves development key custody model: software-backed key file / environment secret for macOS CLI, with clear interface seam for Phase-2 mobile KeyStore/KeyChain.
8. **Actor Taxonomy in Audit Trail:**
   - *Status:* Spec §11 lines 263–265 distinguishes `user` (supervision) and `operator` (recovery/re-enroll).
   - *Resolution:* Operator confirms actor strings: `"user"` for live probe confirmation; `"operator"` for CLI administrative actions.

### Commander/Lead-level Premises (Architectural Decisions Resolved in Spikes)

1. **Storage Engine & Encryption Scheme (Spike S1B):**
   - Standard SQLite database paired with application-layer AEAD field encryption (AES-256-GCM via `cryptography`). SQLite handles relations and ACID transactions; sensitive biometric blobs (crops, embeddings) are stored as encrypted byte payloads with authenticated associated data (AAD) binding to `identity_id` and `revision`. Avoids external native C dependencies (such as SQLCipher).
2. **Transaction Isolation & WAL Durability (Spike S1B):**
   - SQLite configured with `PRAGMA journal_mode = WAL` and `PRAGMA synchronous = NORMAL`. Every promotion, re-enrollment, rollback, and deletion executes within an atomic transaction (`BEGIN IMMEDIATE`).
3. **Event Independence & Burst Suppression Contract (Arbitration Item 6):**
   - Event independence is determined by: immutable evaluation sequence order + unique event UUID + source image SHA-256 digest + minimum time gap (`min_corroboration_interval_secs = 60.0`). Repeated processing of identical hashes or events within the suppression window is filtered out.
4. **Utility Factor Normalization & Deterministic Tie-Breaking:**
   - Six factors normalized to `[0.0, 1.0]`, weighted per `governance_policy_version: 1`. Tie-breaking uses the earliest creation timestamp.
5. **Identity Collision Defense:**
   - `identity add` enforces `UNIQUE(id)` at the database constraint level and wraps operations in transactions, failing immediately with structured exit code `4` if the ID exists (spec §11 lines 260–261).
6. **Status Transition to `re_enrollment_required`:**
   - When stored exemplar cannot be re-embedded or aligned under model update, identity status transitions to `re_enrollment_required`; identification queries against such identities immediately short-circuit to `review` with reason code `identity_re_enrollment_required` (spec §12 lines 313–314).

---

## 4. Documentation Impact Check

Classification: **`area`**.

Evidence: Adding `docs/plans/2026-09-12-phase-1b-implementation-plan.md` updates the repository map in `README.md:29-45`, and tracking Phase-1B progress updates `docs/PROJECT-STATE.md`. Neither changes existing architectural decisions ADR 0004/0006/0007.

- Task 0 updates `README.md` and `docs/PROJECT-STATE.md` to record Phase-1B planning.
- Task 10 performs the final `project-docs-maintain` pass before Phase-1B completion.

---

## 5. Prerequisites and Spikes

In accordance with arbitration `d-20260911184146033747-27` item (1), implementation tasks depend on three explicit spikes:

### Prerequisite P0 — Phase-1B Go/No-Go Decision Gate
- **Nature:** Non-coding procedural gate. **Owner:** Operator.
- **Dependency:** Blocks all implementation PRs (PR-C through PR-G). PR-A (this plan) and PR-B (spikes) may proceed to establish technical readiness, but no persistent code or CLI modification is merged until P0 is approved.
- **Fail-closed verification:** CLI subcommands check for an initialized database; dispatch hooks enforce task dependencies. Task 10 includes an integration test confirming uninitialized or disabled state fails closed.

### Spike S1B — Storage, Cryptography, and KeyProvider Architecture
- **Time box:** One working session. **Owner:** Implementer.
- **Deliverable:** `docs/research/2026-09-12-storage-crypto-manifest.md` on branch `docs/storage-crypto-manifest`.
- **Scope (Arbitration Item 1 & 4):**
  1. Evaluate AES-256-GCM authenticated payload encryption over SQLite tables.
  2. Define AAD construction: `AAD = identity_id || revision_id || field_name`.
  3. Specify 96-bit CSPRNG nonce generation and ensure nonce uniqueness per record write.
  4. Specify `KeyProvider` interface and deliver macOS software key provider (`FileKeyProvider` reading 32-byte secret from environment or restricted `~/.facecore/key`).
  5. Detail atomic transaction boundaries across SQLite tables and verify atomic rollback on failure.
  6. Document secure deletion limitation: Logical deletion + cryptographic erasure of keys/records; explicitly disclose that physical flash memory wear-leveling prevents OS-level guarantees of zero physical remanence.
  7. Key rotation is explicitly deferred; threat model records it as a Phase-2 enhancement.
- **Blocks:** Task 2 (Encrypted Storage Repository).

### Spike S2B — Model Weights and Corpus Chronology Readiness
- **Time box:** One working session. **Owner:** Implementer.
- **Deliverable:** `docs/research/2026-09-12-weights-corpus-readiness.md` on branch `docs/weights-corpus-readiness`.
- **Scope (Arbitration Item 1 & 6):**
  1. Record immutable SHA-256, license status, and file paths for SFace 2021dec and YuNet 2023mar.
  2. Inventory consented evaluation probes, timestamps, and identity groupings.
  3. If weights or multi-timestamp probes remain unavailable under operator dual-gate rules: formally freeze the synthetic adversarial stream specifications for Task 8, and record real replay as `blocked` in Task 9.
- **Blocks:** Task 8 (Replay Harness) and Task 9 (Evaluation Run).

---

## 6. Frozen Governance Policy Defaults (`governance_policy_version: 1`)

In accordance with arbitration `d-20260911184146033747-27` item (3), all operational thresholds and utility weights are declared versioned and auditable:

| Parameter | Type / Unit | Default Value | Specification Reference & Rationale |
|---|---|---|---|
| `candidate_update_threshold` | float (similarity) | `>= 0.88` | Spec §9 line 176: Stricter than normal `match_threshold` (provisional 0.80) to prevent noisy template poisoning. |
| `corroboration_min_events` | integer (count) | `1` | Spec §9 line 190: At least one additional temporally independent event required before first promotion. |
| `burst_suppression_min_interval_secs`| float (seconds) | `60.0` | Spec §9 line 191: Consecutive probe events within 60s are treated as single burst and cannot satisfy independent corroboration. |
| `promotion_margin` | float (similarity) | `>= 0.12` | Spec §9 line 188: Candidate must score higher for owning identity than all other enrolled identities by this margin. |
| `template_bank_capacity` | integer (slots) | `5` | Spec §9 line 200: Maximum active templates per identity; capacity triggers utility rescoring and eviction. |
| `utility_weight_quality` | float (weight) | `0.25` | Spec §9 line 203: Normalized variance of Laplacian and face size. |
| `utility_weight_support` | float (weight) | `0.25` | Spec §9 line 204: Number of independent corroborating events. |
| `utility_weight_recency` | float (weight) | `0.15` | Spec §9 line 205: Exponential decay half-life = 180 days. |
| `utility_weight_coverage` | float (weight) | `0.15` | Spec §9 line 206: Cosine diversity from existing active template centroid. |
| `utility_weight_redundancy` | float (weight) | `-0.10` | Spec §9 line 207: Penalty for near-identical duplicate embeddings. |
| `utility_weight_outlier` | float (weight) | `-0.10` | Spec §9 line 208: Penalty for embeddings near the non-target decision boundary. |
| `drift_max_centroid_shift` | float (distance) | `0.20` | Spec §9 line 198: Maximum permissible cosine distance between active centroid and initial enrollment template. |
| `drift_max_initial_distance` | float (distance) | `0.25` | Spec §9 line 198: Individual active template distance bound from initial enrollment template. |
| `rollback_max_depth` | integer (revisions) | `5` | Spec §9 line 211: Number of historical revisions retained for recovery rollback. |
| `retired_retention_days` | integer (days) | `90` | Spec §9 line 211: Retention period for retired templates before cryptographic deletion. |
| `exemplar_margin` | float (fraction) | `0.0` | Spec §10 line 222: Minimal crop margin retained in Phase 1B to satisfy data minimization. |

**Deterministic Tie-Breaking:** If two active templates achieve identical utility scores during capacity eviction, the template with the earlier creation timestamp is preserved, evicting the newer unproven template.

---

## 7. Storage, Deletion, and History Retention Mechanics

### Data Model Schema (`schema_version: 1`)

SQLite tables partitioned into non-sensitive relational metadata and encrypted biometric payloads:

1. `identities`: `id` (TEXT PRIMARY KEY), `display_name` (TEXT), `status` (TEXT: `active`, `re_enrollment_required`, `deleted`), `current_revision` (INTEGER), `created_at` (TEXT), `updated_at` (TEXT).
2. `template_revisions`: `id` (INTEGER PRIMARY KEY AUTOINCREMENT), `identity_id` (TEXT), `revision` (INTEGER), `active_template_ids` (TEXT: JSON list), `retired_template_ids` (TEXT: JSON list), `policy_version` (INTEGER), `created_at` (TEXT), `actor` (TEXT: `user`, `operator`).
3. `face_templates`: `id` (TEXT PRIMARY KEY), `identity_id` (TEXT), `generation_id` (TEXT), `status` (TEXT: `active`, `retired`), `encrypted_embedding` (BLOB), `nonce` (BLOB), `quality_score` (REAL), `utility_score` (REAL), `corroboration_count` (INTEGER), `created_at` (TEXT), `retired_at` (TEXT NULL).
4. `candidate_templates`: `id` (TEXT PRIMARY KEY), `identity_id` (TEXT), `generation_id` (TEXT), `status` (TEXT: `pending`, `promoted`, `rejected`, `expired`), `encrypted_embedding` (BLOB), `encrypted_exemplar` (BLOB NULL), `nonce` (BLOB), `quality_score` (REAL), `evidence_log` (TEXT: JSON), `expires_at` (TEXT), `created_at` (TEXT).
5. `match_events`: `id` (TEXT PRIMARY KEY), `timestamp` (TEXT), `status` (TEXT), `decision_score` (REAL), `runner_up_score` (REAL NULL), `matched_identity_id` (TEXT NULL), `candidate_created` (INTEGER), `actor` (TEXT NULL). (Zero image or embedding data).

### Cryptographic Erasure & Deletion Lifecycle (Arbitration Item 5)

- **Logical Deletion + Crypto-Erasure:**
  When `identity delete --id <id>` executes:
  1. Transaction `BEGIN IMMEDIATE` acquires lock.
  2. All active, retired, and candidate embeddings and exemplars for `<id>` are overwritten with cryptographic random bytes (`os.urandom`) before row deletion.
  3. Rows in `identities`, `template_revisions`, `face_templates`, and `candidate_templates` are deleted.
  4. Any `match_events` referencing `<id>` have `matched_identity_id` set to `NULL` (anonymized tombstone).
  5. Transaction commits; SQLite `PRAGMA wal_checkpoint(TRUNCATE)` is executed to clear write-ahead log remanence.
  6. **Limitation disclosure:** The plan explicitly notes that underlying SSD/flash storage wear-leveling controllers may retain physical cell remnants beyond OS file visibility. Phase 1B guarantees cryptographic and logical un-recoverability through standard OS file interfaces, but does not claim DOD-level physical degaussing.

### Storage Stress & Retention Limit (Arbitration Item 7)

- To prevent unbounded event log growth, `match_events` retention is capped at 10,000 events. Older events are trimmed during initialization checkpoints.
- Task 2 includes a 10,000-event synthetic stress benchmark validating database integrity, index scan latency, and file size stability under continuous operation.

---

## 8. Synthetic Streams and Adversarial Governance Testing

In accordance with arbitration `d-20260911184146033747-27` item (6), synthetic data in Phase 1B is strictly scoped to **governance state-machine validation** and **adversarial policy testing**, never for biometric accuracy claims:

### Adversarial Test Streams:
1. **Wrong confirmation label:** Probe matches identity A, but operator/user inputs `not_me`. Asserts zero candidates created, policy error recorded.
2. **Duplicate image hash:** Exact same image file submitted repeatedly in succession. Asserts duplicate detection, zero candidate creation, zero corroboration credit.
3. **Burst event stream:** Multiple distinct probes with simulated variations submitted within `< 60s`. Asserts burst suppression prevents multiple corroboration counts.
4. **Out-of-order / Future timestamp stream:** Probe with timestamp $t_{N+100}$ injected into sequence. Asserts replay harness refuses out-of-order events.
5. **Cross-identity margin tie:** Synthetic candidate matches identity A at 0.90 and identity B at 0.89 (margin 0.01 < promotion margin 0.12). Asserts promotion refusal.
6. **Corrupt storage / missing key injection:** SQLite header corrupted or key altered mid-operation. Asserts clean fail-closed exit code `4` with zero partial revision.

---

## 9. 500-Identity Capacity & Storage Benchmark

In accordance with arbitration `d-20260911184146033747-27` item (9):

- **Target:** 500 enrolled identities with full active template banks (2,500 total active templates) stored in encrypted SQLite.
- **Query mechanism:** Bounded exact linear scan over active templates using vectorized NumPy operations. No approximate nearest neighbor (ANN) index; no external caching layer.
- **Latency & Capacity budget (Apple M1 reference machine):**
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
│   ├── crypto.py           # EncryptedBlob, KeyReference, KeyProviderProtocol
│   ├── drift.py            # DriftMetrics, DriftStatus, DriftPolicy
│   ├── export.py           # ExportContainer, ExportMetadata, ExportManifest
│   └── policy.py           # Extended GovernancePolicy (thresholds, weights, TTL)
├── storage/
│   ├── base.py             # PersistentRepository protocol
│   ├── cipher.py           # AES-GCM encryption/decryption, nonce, AAD
│   ├── key_provider.py     # KeyProvider protocol, FileKeyProvider, MockKeyProvider
│   ├── sqlite_repo.py      # SQLite implementation with ACID transactions & WAL
│   ├── export.py           # Encrypted export/import serializer and unpacker
│   └── migrations/         # Schema DDL (v1)
├── governance/
│   ├── candidate.py        # Candidate creation, update gate, expiry check
│   ├── corroboration.py    # Multi-event corroborator, burst suppressor
│   ├── promotion.py        # Exclusivity margin checker, promotion manager
│   ├── utility.py          # 6-factor utility rescoring engine & tie-breaking
│   ├── eviction.py         # Capacity checker and atomic retirement manager
│   ├── lifecycle.py        # Identity create, re-enroll, rollback, delete
│   └── drift.py            # Centroid tracking, initial template distance, drift detector
├── eval/
│   ├── replay.py           # Chronological adaptive replay harness
│   ├── replay_report.py    # Replay vs 1A frozen baseline comparison table
│   └── benchmark_1b.py     # 500-identity encrypted storage & comparison benchmark
└── cli.py                  # CLI commands: identity, candidates, status, --confirm-learning
```

---

## 11. Tasks

### Task 0: Land this plan and repoint project docs
- **Files:**
  - Create: `docs/plans/2026-09-12-phase-1b-implementation-plan.md`
  - Modify: `README.md` (repository map); `docs/PROJECT-STATE.md` ("Current Status" & "Next Session")
- **Interfaces:** Consumes spec §§9–12, 14, Phase-1A closeout state (`98b202a`), and arbitration `d-20260911184146033747-27`.
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
- **Files:**
  - Create: `src/facecore/contracts/candidate.py`, `src/facecore/contracts/confirmation.py`, `src/facecore/contracts/crypto.py`, `src/facecore/contracts/drift.py`, `src/facecore/contracts/export.py`
  - Modify: `src/facecore/contracts/policy.py`, `src/facecore/contracts/template.py`, `src/facecore/contracts/result.py`
  - Test: `tests/contracts/test_governance_contracts.py`
- **Interfaces:** Produces `CandidateTemplate`, `ConfirmationRequest`, `EncryptedBlob`, `KeyProviderProtocol`, `DriftMetrics`, `ExportContainer`, `GovernancePolicy` (`governance_policy_version: 1`).
- **Acceptance:**
  - `IdentificationResult` serializes `candidate_created` dynamically based on confirmation outcome.
  - `CandidateTemplate` models `status` enum (`pending`, `promoted`, `rejected`, `expired`) and prevents instantiation without valid encryption nonce.
  - `GovernancePolicy` defines all 16 parameters in Section 6 with strict typing and schema validation.
- **Test-first evidence:**
  - Failing case: Instantiating `CandidateTemplate` with plaintext embedding or unconfirmed status raises validation error.
  - RED: `pytest tests/contracts/test_governance_contracts.py -q` → `ModuleNotFoundError: No module named 'facecore.contracts.candidate'`.
  - Minimal behavior: Implement dataclass models, enums, JSON encoders, and validation predicates.
  - GREEN: `pytest tests/contracts/test_governance_contracts.py -q` → all pass.
  - Project verification: `mypy src tests` and `ruff check src tests` clean.

### Task 2: KeyProvider Abstraction and Encrypted Storage Repository
- **Depends on:** Spike S1B, Task 1.
- **Files:**
  - Create: `src/facecore/storage/cipher.py`, `src/facecore/storage/key_provider.py`, `src/facecore/storage/sqlite_repo.py`, `src/facecore/storage/migrations/v1.sql`
  - Test: `tests/storage/test_cipher.py`, `tests/storage/test_key_provider.py`, `tests/storage/test_sqlite_repo.py`, `tests/storage/test_stress.py`
- **Interfaces:** Produces `KeyProvider` protocol, `FileKeyProvider`, `AESGCMCipher`, `SQLitePersistentRepository` implementing `PersistentRepository`.
- **Acceptance:**
  - All embeddings and exemplars written to SQLite are encrypted with AES-256-GCM and unique nonces; AAD binds to identity and revision.
  - Database corruption, missing key file, or invalid authentication tag triggers `StoreCorruptionError` or `KeyNotFoundError`, exiting with code `4`.
  - ACID transactions guarantee interrupted writes leave zero partial revisions or orphaned candidate records.
  - 10,000-event stress test proves database integrity, WAL checkpointing, and stable query latency.
- **Test-first evidence:**
  - Failing case: Corrupting SQLite DB header bytes or bit-flipping ciphertext raises `StoreCorruptionError` (exit code 4).
  - RED: `pytest tests/storage/test_sqlite_repo.py -k "test_tampered_ciphertext" -q` → `Failed: DID NOT RAISE StoreCorruptionError`.
  - Minimal behavior: Implement SQLite schema, AES-GCM cipher with AAD verification, and fail-closed error handling.
  - GREEN: `pytest tests/storage/ -q` → all pass.
  - Project verification: `mypy src` clean; memory leak checks pass under 10k iterations.

### Task 3: Confirmation-Gated Shadow Candidate Pipeline
- **Depends on:** Task 1, Task 2.
- **Files:**
  - Create: `src/facecore/governance/candidate.py`
  - Modify: `src/facecore/cli.py` (`--confirm-learning` handler)
  - Test: `tests/governance/test_candidate_pipeline.py`
- **Interfaces:** Produces `CandidatePipeline.evaluate_observation(result, decoded_face, confirmation) -> CandidateDecision`.
- **Acceptance:**
  - Candidate creation strictly requires: (a) match result above `candidate_update_threshold` (0.88), (b) quality gate accepted, (c) explicit `correct` confirmation from user/operator.
  - `not_me`, cancellation, timeout, or EOF leaves zero records in `candidate_templates` and leaves filesystem completely unmodified.
  - Pending observation is held strictly in memory; process termination leaves zero unencrypted biometric artifacts.
- **Test-first evidence:**
  - Failing case: Supplying `not_me` confirmation or observation below update threshold creates a candidate row.
  - RED: `pytest tests/governance/test_candidate_pipeline.py -k "test_not_me_creates_no_candidate" -q` → `assert 1 == 0`.
  - Minimal behavior: Implement strict gating, memory-only pending buffers, and SQLite candidate insertion only on `correct`.
  - GREEN: `pytest tests/governance/test_candidate_pipeline.py -q` → all pass.
  - Project verification: Worktree clean; zero uncommitted SQLite files.

### Task 4: Corroboration Engine, Promotion Exclusivity, and Utility Eviction
- **Depends on:** Task 3.
- **Files:**
  - Create: `src/facecore/governance/corroboration.py`, `src/facecore/governance/promotion.py`, `src/facecore/governance/utility.py`, `src/facecore/governance/eviction.py`
  - Test: `tests/governance/test_corroboration.py`, `tests/governance/test_promotion.py`, `tests/governance/test_utility_eviction.py`
- **Interfaces:** Produces `CorroborationEngine`, `PromotionManager`, `UtilityRescorer`, `EvictionManager`.
- **Acceptance:**
  - Rapid probe submissions within `burst_suppression_min_interval_secs` (60s) or duplicate image hashes increment zero corroboration counts.
  - Candidate promotion requires: (a) `corroboration_count >= corroboration_min_events` (1), (b) exclusivity margin over all enrolled non-target identities `>= promotion_margin` (0.12).
  - When active template bank reaches capacity (5), all templates are rescored via 6 utility factors; lowest-utility template is retired atomically.
  - Initial enrollment template enjoys no special exemption and retires under the same utility policy as later templates.
  - New atomic revision is committed via `append_revision` only after promotion succeeds.
- **Test-first evidence:**
  - Failing case: Candidate with insufficient cross-identity margin (0.05 < 0.12) is promoted to active template bank.
  - RED: `pytest tests/governance/test_promotion.py -k "test_cross_identity_tie_refuses_promotion" -q` → `assert candidate.status == 'promoted'`.
  - Minimal behavior: Implement burst filter, 1:N exclusivity query, 6-factor utility rescoring, and atomic revision commit.
  - GREEN: `pytest tests/governance/test_promotion.py tests/governance/test_utility_eviction.py -q` → all pass.
  - Project verification: Verification of atomic rollback if eviction fails mid-transaction.

### Task 5: Identity Lifecycle CLI & Recovery Operations
- **Depends on:** Task 2, Task 4.
- **Files:**
  - Create: `src/facecore/governance/lifecycle.py`
  - Modify: `src/facecore/cli.py` (add `identity add/show/re-enroll/rollback/delete`, `candidates list/reject`, `status`)
  - Test: `tests/cli/test_identity_lifecycle.py`, `tests/storage/test_crypto_erasure.py`
- **Interfaces:** CLI subcommands emitting stable JSON per spec §11.
- **Acceptance:**
  - `identity add` is create-only; duplicate ID fails without mutation and directs operator to `re-enroll`.
  - `identity re-enroll` atomically retires all current active templates and sets new enrollment photo as sole active template in a new revision.
  - `identity rollback --to-revision <n>` restores active templates to revision $n$, retiring newer templates; refuses if revision exceeds `rollback_max_depth`.
  - `identity delete` executes cryptographic erasure: overwrites embedding blobs, deletes rows, anonymizes match events, and truncates WAL. Direct inspection of SQLite database file reveals zero plaintext biometric strings or valid ciphertext payloads.
  - `candidates reject --id <id> --reason <code>` marks candidate `rejected`, preventing promotion.
- **Test-first evidence:**
  - Failing case: Querying SQLite file binary after `identity delete` detects residual embedding bytes.
  - RED: `pytest tests/storage/test_crypto_erasure.py -q` → `AssertionError: Residual ciphertext block found in SQLite DB`.
  - Minimal behavior: Implement CLI argument parsing, transactional lifecycle actions, CSPRNG overwrite on delete, and WAL truncation.
  - GREEN: `pytest tests/cli/test_identity_lifecycle.py tests/storage/test_crypto_erasure.py -q` → all pass.
  - Project verification: CLI returns correct exit codes: 0 normal, 2 bad args/image, 4 storage error.

### Task 6: Encrypted Export / Import with Generation Compatibility
- **Depends on:** Task 5.
- **Files:**
  - Create: `src/facecore/storage/export.py`
  - Modify: `src/facecore/cli.py` (add `export`, `import`)
  - Test: `tests/storage/test_export_import.py`
- **Interfaces:** Produces `export_identities(path, key) -> ExportManifest`, `import_identities(path, key) -> ImportResult`.
- **Acceptance:**
  - Export archive contains encrypted identities, revisions, active templates, and model manifest signature.
  - Import verifies model manifest: if exported `model_version`, embedding dimension, or preprocessing generation differs from current runtime, import fails closed with exit code `3` (`ModelIncompatibilityError`) and makes zero database mutations.
  - Corrupted archive or incorrect decryption key fails closed with exit code `4`.
- **Test-first evidence:**
  - Failing case: Importing an archive generated with a mismatched model dimension succeeds or partially writes records.
  - RED: `pytest tests/storage/test_export_import.py -k "test_mismatched_model_generation_refused" -q` → `Failed: DID NOT RAISE ModelIncompatibilityError`.
  - Minimal behavior: Implement encrypted tar/zip container, header manifest verification, and transactional import unpacker.
  - GREEN: `pytest tests/storage/test_export_import.py -q` → all pass.
  - Project verification: Import followed by `identify` reproduces identical identification scores.

### Task 7: Long-Horizon Drift Indicators and Bounded Response Policy
- **Depends on:** Task 4.
- **Files:**
  - Create: `src/facecore/governance/drift.py`
  - Modify: `src/facecore/contracts/drift.py`, `src/facecore/policy/identify.py`
  - Test: `tests/governance/test_drift_policy.py`
- **Interfaces:** Produces `DriftDetector.measure_identity_drift(identity_id) -> DriftMetrics`.
- **Acceptance:**
  - Tracks two primary drift indicators: (a) active template centroid cosine shift from initial enrollment template, (b) maximum individual template distance from initial enrollment template.
  - When centroid shift exceeds `drift_max_centroid_shift` (0.20) or individual distance exceeds `drift_max_initial_distance` (0.25), drift detector triggers bounded response:
    1. Identification status downgrades from `matched` to `review` with reason code `drift_boundary_exceeded`.
    2. Identity lifecycle status marks `re_enrollment_required`.
    3. Further automatic candidate creation and promotion for that identity are blocked until trusted re-enrollment.
- **Test-first evidence:**
  - Failing case: Synthetically drifted template bank exceeding 0.20 centroid shift continues to return status `matched`.
  - RED: `pytest tests/governance/test_drift_policy.py -k "test_centroid_drift_triggers_review" -q` → `assert result.status == 'review' (got 'matched')`.
  - Minimal behavior: Implement centroid calculation, drift distance predicates, and policy decision hook.
  - GREEN: `pytest tests/governance/test_drift_policy.py -q` → all pass.
  - Project verification: `mypy src` clean.

### Task 8: Chronological Adaptive Replay Harness & Temporal-Leakage Prevention
- **Depends on:** Spike S2B, Task 4, Task 7.
- **Files:**
  - Create: `src/facecore/eval/replay.py`
  - Modify: `src/facecore/eval/session.py`
  - Test: `tests/eval/test_replay.py`, `tests/eval/test_temporal_leakage.py`
- **Interfaces:** Produces `ChronologicalReplayHarness.run_replay(corpus_manifest, model_manifest, policy) -> ReplayExecutionSummary`.
- **Acceptance:**
  - Harness iterates through probe events in strict monotonically increasing timestamp order.
  - Ground-truth labels from corpus supply supervision solely within the test harness; never exposed to runtime CLI.
  - Strict temporal-leakage test: Injects assertion at event $t_N$ verifying that database queries cannot see templates, revisions, or candidates created at $t > t_N$. Any leak raises `TemporalLeakageError`.
  - Supports synthetic adversarial streams (bursts, bad labels, duplicates) to validate governance state transitions.
- **Test-first evidence:**
  - Failing case: An event at index $k$ reads a template revision created by event $k+1$ without error.
  - RED: `pytest tests/eval/test_temporal_leakage.py -q` → `Failed: DID NOT RAISE TemporalLeakageError`.
  - Minimal behavior: Implement event timestamp ordering, temporal state snapshot isolation, and leakage verification hooks.
  - GREEN: `pytest tests/eval/test_replay.py tests/eval/test_temporal_leakage.py -q` → all pass.
  - Project verification: Replay produces deterministic candidate creation, promotion, and retirement event logs.

### Task 9: Phase-1B Evaluation Run & Baseline Comparison Report
- **Depends on:** Task 8.
- **Files:**
  - Create: `src/facecore/eval/replay_report.py`
  - Modify: `facecore.sh` (add `replay` command)
  - Test: `tests/eval/test_replay_report.py`
- **Interfaces:** Produces `./facecore.sh replay --corpus <manifest> --report reports/phase-1b-replay.md`.
- **Acceptance:**
  - Executes chronological replay using Candidate A (SFace Pair 1 provisional) against the consented P1 corpus (or synthetic adversarial stream if real weights/probes remain blocked by operator dual gate).
  - Emits side-by-side operating comparison table contrasting frozen Phase-1A baseline vs Phase-1B adaptive template bank:
    - Target probe match / review / unknown counts.
    - False acceptance counts on non-target probes.
    - Candidate creation counts, corroboration counts, promotion counts, and eviction counts.
    - Measured drift metrics across evaluation timeline.
    - Exact denominators on every cell (rate formatting forbidden below $N < 30$).
  - If real weights or multi-timestamp probes are unavailable, report explicitly documents real replay as `blocked-with-reason`, evaluates synthetic stream, and leaves model selection gate OPEN (`d-20260911181002229319-23`).
  - Redaction check: Report output scanned before write; asserts zero raw embeddings, face crops, or personal local paths.
- **Test-first evidence:**
  - Failing case: Replay report emits a percentage rate for a denominator of 5, or omits the baseline comparison delta column.
  - RED: `pytest tests/eval/test_replay_report.py -q` → `AssertionError: Denominator 5 must be formatted as 'k/N', not percentage`.
  - Minimal behavior: Implement comparison metrics calculation, table formatter with exact denominators, and redaction assertions.
  - GREEN: `pytest tests/eval/test_replay_report.py -q` → all pass.
  - Project verification: `./facecore.sh replay` executes end to end on synthetic test corpus.

### Task 10: 500-Identity Capacity Benchmark & Project Docs Maintenance
- **Depends on:** Task 5, Task 9.
- **Files:**
  - Create: `src/facecore/eval/benchmark_1b.py`
  - Modify: `README.md`, `docs/PROJECT-STATE.md`
  - Test: `tests/eval/test_benchmark_1b.py`
- **Interfaces:** Produces 500-identity capacity report section in `reports/phase-1b-replay.md`.
- **Acceptance:**
  - Generates 500 synthetic identities with 5 active templates each (2,500 total vectors) stored in encrypted SQLite.
  - Measures 1:N exact comparison latency: comparison p95 ≤ 3.0 ms; SQLite fetch & decrypt p95 ≤ 15.0 ms.
  - Verifies CLI fail-closed integration when uninitialized or corrupt.
  - Executes `project-docs-maintain` pass: `README.md` and `docs/PROJECT-STATE.md` match delivered codebase; records Phase-1B completion and ADR 0006 status.
- **Test-first evidence:**
  - Failing case: Benchmark fails to separate comparison time from storage decryption time, or repository map omits newly added modules.
  - RED: `pytest tests/eval/test_benchmark_1b.py -q` → `KeyError: 'sqlite_fetch_p95_ms'`.
  - Minimal behavior: Implement separate timing probes, run benchmark, update documentation.
  - GREEN: `pytest tests/eval/test_benchmark_1b.py -q` → all pass.
  - Project verification: Map diff check clean; `ruff check src tests` and `mypy src` clean.

---

## 12. PR Boundaries

In accordance with arbitration `d-20260911184146033747-27` item (2), implementation is partitioned into **7 distinct PRs**. PR-A is strictly docs-only to preserve the ADR 0006 review gate. Tasks grouped within implementation PRs maintain strict module, fixture, and failure matrix separation so that partial test failures in one task do not mask or block another:

| PR | Included Tasks | Scope & Boundaries | Independent Acceptance Criteria |
|---|---|---|---|
| **PR-A** | Task 0 | Plan only (docs-only). Land `docs/plans/2026-09-12-phase-1b-implementation-plan.md`, update `README.md` map and `docs/PROJECT-STATE.md`. No Python source code. | Map matches disk exactly; `PROJECT-STATE.md` reflects 1B planning status; CI passes. ADR 0006 gate remains intact. |
| **PR-B** | Spikes S1B, S2B | Research manifests (docs-only). Land storage/crypto manifest and model/corpus readiness reports. | Concrete AAD specifications, nonce rules, KeyProvider seam, model SHAs, and corpus chronology recorded. |
| **PR-C** | Task 1, Task 2 | Foundation: Contracts, `KeyProvider`, AES-GCM cipher, and encrypted SQLite repository. | All contracts versioned (`schema_version: 1`, `governance_policy_version: 1`). SQLite ACID transactions, tampering detection, and 10k-event stress test pass. Zero CLI mutation. |
| **PR-D** | Task 3, Task 4 | Adaptive Core: Confirmation-gated candidate pipeline, corroboration, promotion exclusivity, 6-factor utility rescoring, and capacity eviction. | Candidate creation strictly gated by `correct`; burst suppression active; cross-identity margin enforced; lowest-utility template evicted on capacity; atomic revisions committed. Modules and fixtures split between T3 and T4; T3 tests pass independently of T4. |
| **PR-E** | Task 5, Task 6 | Administrative CLI & Recovery: `identity add/show/re-enroll/rollback/delete`, `candidates list/reject`, encrypted export/import. | `identity add` collision blocked; `re-enroll` and `rollback` atomic; `delete` proves cryptographic erasure across DB and WAL; export/import enforces model generation compatibility (exit code 3). Failure matrices split between T5 (store) and T6 (model). |
| **PR-F** | Task 7, Task 8 | Drift Monitoring & Replay Engine: Long-horizon drift metrics, bounded response policy, chronological replay harness, temporal-leakage isolation tests. | Drift threshold breach triggers `review` and `re_enrollment_required`; replay executes in strict timestamp order; temporal leakage assertions prove $t_N$ cannot read $> t_N$. |
| **PR-G** | Task 9, Task 10 | Evaluation, Benchmark & Closeout: SFace Pair 1 replay run vs 1A frozen baseline, 500-identity capacity benchmark, `project-docs-maintain` documentation pass. | Replay report emitted with exact denominators and delta comparison; 500-identity benchmark meets latency budget; redaction check clean; `README.md` and `docs/PROJECT-STATE.md` updated. |

**Partial Red Rule:** Within multi-task PRs (PR-D, PR-E), tests are partitioned into distinct test files (`test_candidate_pipeline.py` vs `test_promotion.py`). A failing test under Task 4 does not prevent Task 3 from demonstrating independent functional correctness.

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

In accordance with arbitration `d-20260911184146033747-27` item (11), the plan defines five concrete risks with pre-planned mitigations and **three non-negotiable STOP conditions**:

### Explicit STOP Conditions:
1. **STOP Condition 1 (Unsupervised Learning Breach):** If any code path, test, or replay execution allows an unconfirmed observation (or one confirmed with `not_me`, canceled, or timed out) to create an active template, candidate, or persistent biometric artifact → **HALT IMMEDIATELY**. Reviewer must reject the PR.
2. **STOP Condition 2 (Fail-Open or Plaintext Storage Breach):** If any unit test, failure injection, or manual inspection reveals that storage corruption, key absence, or model incompatibility fails open (exits 0 instead of 3/4), or leaves unencrypted embeddings or face crops on disk → **HALT IMMEDIATELY**. Do not proceed until cryptographic fail-closed behavior is restored.
3. **STOP Condition 3 (Gate Weakening or Premature Gate Closure):** If ONNX weights or multi-timestamp real evaluation probes are missing under operator dual-gate rules, the team must **NOT** weaken the gates, fabricate data, or close the model selection gate. The replay task must mark real replay `blocked-with-reason`, evaluate synthetic streams only, and keep the selection gate OPEN.

### Operational Risks & Mitigations:
1. **Risk: Cumulative template drift and self-poisoning over long horizons.**
   - *Mitigation:* Spec §9 line 198 acknowledges that similarity-based gates share the same embedder and cannot eliminate drift. Task 7 bounds this risk by measuring centroid shift and initial enrollment distance, automatically halting candidate promotion and triggering `re_enrollment_required` when boundaries are breached.
2. **Risk: ONNX weights download blocked by provenance concerns.**
   - *Mitigation:* Decision `d-20260911174928737696-16` strictly enforces the dual gate. If operator approval is withheld, Task 8 and Task 9 execute against synthetic adversarial vectors, proving governance mechanics without violating licensing boundaries.
3. **Risk: SQLite WAL locking and concurrency issues during recovery operations.**
   - *Mitigation:* Spike S1B standardizes on WAL mode with busy timeouts and explicit `BEGIN IMMEDIATE` transactions. Unit tests verify crash consistency and atomic rollback under mid-transaction process kills.
4. **Risk: Over-interpreting synthetic 500-identity benchmarks.**
   - *Mitigation:* Reports and code documentation explicitly state that synthetic benchmarks validate latency, memory, and database scalability only, and provide zero evidence regarding false acceptances, margin distributions, or recognition accuracy.
5. **Risk: Temporal leakage in chronological replay invalidating baseline comparisons.**
   - *Mitigation:* Task 8 implements an adversarial temporal leakage test suite that asserts $t_N$ queries cannot observe future revisions, ensuring the validity of evaluation conclusions.
