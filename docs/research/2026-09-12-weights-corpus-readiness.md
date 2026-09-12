# Spike S2B — Model Weights and Corpus Chronology Readiness Decision Manifest

- Date: **2026-09-12 Asia/Taipei**
- Source of truth: `docs/plans/2026-09-12-phase-1b-implementation-plan.md` §5 (Spike S2B), merged at `4bfaa94`.
- Governing decisions:
  - `d-20260912041106780391-34`: Operator Decision Manifest (P0 closed; Item 2 weights download approved; Item 3 corpus decision (b) ratified: synthetic temporal stream as primary governance vehicle, real replay marked `blocked-with-reason`, model selection gate remains `OPEN`).
  - `d-20260911174928737696-16`: ONNX weights download dual-gate blocker (S1 clear + operator decision).
  - `d-20260911181002229319-23`: Operator selects Candidate A (SFace Pair 1 provisional; selection gate stays OPEN pending Phase-1B chronological replay).
  - `d-20260911184146033747-27`: Team discuss arbitration (Spike S2B frozen deliverables, real replay blocked gate rules).
  - `d-20260912044625991911-38`: Spike S2B dispatch freeze (base `3080df46`).
  - `d-20260912045928673582-39`: §3.18 arbitration on PR #21 r0 (scoped supplement: auditable download/verification log, explicit Pair 2 boundary, external corpus inventory digest).
  - `d-20260912050730885132-40`: §3.18 ruling on PR #21 r1 P2 (scoped supplement: exact 22-file inventory command including *.jpg, excluded-extension policy, canonical manifest input identification).
- Scope boundary: **Analysis-only architectural decision manifest** — zero production code, zero biometric data in Git, strictly offline except authorized weights download from recorded upstream URLs, zero access to `/Users/cheerc/幼兒園校園相簿`.
- Blocks: **Phase 1B Task 9** (Chronological Adaptive Replay Harness) and **Task 10** (Phase-1B Evaluation Run & Baseline Comparison Report).

---

## Executive Summary & Readiness Status Table

In accordance with Phase-1B Implementation Plan §5 and governing decisions `d-20260912041106780391-34`, `d-20260912044625991911-38`, `d-20260912045928673582-39`, and `d-20260912050730885132-40`, this spike records the definitive readiness state of model weights and evaluation corpus for Phase 1B:

| Evaluation Dimension | Source & Identity | Readiness Verdict | Governed Action & Boundary |
|---|---|---|---|
| **1. Detector Model Weights** | YuNet 2023mar (`face_detection_yunet_2023mar.onnx`) | **READY (Verified on disk in models/)** | Download authorized by Operator Decision Manifest Item 2. Immutable SHA-256 (`8f2383e4...`) and size (232,589 B) verified against S1 records. Placed in gitignored `models/`. Hash-at-construction verification enforced. |
| **2. Embedder Model Weights** | SFace 2021dec fp32 (`face_recognition_sface_2021dec.onnx`) | **READY (Verified on disk in models/)** | Download authorized by Operator Decision Manifest Item 2. Immutable SHA-256 (`0ba9fbfa...`) and size (38,696,353 B) verified against S1 records. Placed in gitignored `models/`. Issue #313 provenance tracked as open. |
| **3. Pair 2 Candidate Model Weights** | YuNet 2026may + SFace 2021dec_int8bq | **BLOCKED / UN-DOWNLOADED** | NOT authorized by Operator Decision Manifest `d-20260912041106780391-34`. Governed by preconditions decision `d-20260911171253541089-12`. Remains absent from disk and excluded from 1B execution. |
| **4. Consented Real Evaluation Corpus (P1)** | 5 enrolled identities, 13 target probes (8 usable), 4 non-target probes (22 total files) | **INSUFFICIENT for Longitudinal Chronological Replay** | Read-only inventory confirms 4 of 5 identities have zero target probes; all 13 probes for `enroll-23` originate from a single 2-minute burst. All 10 spec §14 environmental conditions are `untested`. |
| **5. Corpus Decision & Replay Route** | Operator Decision Manifest Item 3 Option (b) | **CONFIRMED: Synthetic Primary + Real Blocked** | In strict compliance with STOP Condition 3, real replay is marked `blocked-with-reason`. The model selection gate **STAYS OPEN**. Synthetic adversarial streams are frozen as the primary governance validation vehicle for Task 9. |

---

## 1. Candidate Model Weights Triple (SHA-256, License, Paths)

In accordance with Operator Decision Manifest `d-20260912041106780391-34` item 2, the download of SFace 2021dec fp32 and YuNet 2023mar weights was formally approved, satisfying the dual-gate requirement of decision `d-20260911174928737696-16`. The artifacts were retrieved exclusively via the authorized upstream OpenCV Zoo media URLs, verified against S1 git-SHA-1 and byte sizes, placed in the gitignored `models/` directory, and hashed with SHA-256.

### 1A. Detector: YuNet 2023mar (`face_detection_yunet_2023mar.onnx`)

1. **Exact File Identity & Immutable SHA-256 Checksum:**
   - **Filename:** `face_detection_yunet_2023mar.onnx`
   - **Immutable Weight SHA-256:** `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
   - **Exact File Size:** `232,589` bytes
   - **Upstream Git Blob SHA-1:** `2d8804a5986e229f1fde3a1994feacc66c91b58b`
   - **Direct Download Source:** `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
   - **Retrieval Date:** 2026-09-12 (re-verified matching 2026-09-11 S1 spike)
2. **License Status:**
   - **Code & Weight License:** **MIT License** (`CLEAR`, text-explicit)
   - **Copyright Holder:** `© 2020 Shiqi Yu <shiqi.yu@gmail.com>`
   - **Locator:** `https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE`
   - **Grant Terms:** Explicitly authorizes commercial use, modification, distribution, and sublicense.
3. **Training-Data Provenance:**
   - **Status:** `PROVENANCE_UNRESOLVED`
   - **Details:** Model source repository `ShiqiYu/libfacedetection.train` (commit `a61a428`). Trained to detect faces ~10x10 to 300x300 pixels; evaluated on WIDER Face validation set. Composition of private/crawled training images is not enumerated in OpenCV Zoo. Kept in candidate pool per plan stop conditions.
4. **Local Worktree Storage Path & Git Isolation:**
   - **Worktree Path:** `models/face_detection_yunet_2023mar.onnx`
   - **Git Status:** Denied by repository root `.gitignore` (`/models/`). Zero weight bytes are committed or tracked in Git.
5. **Hash-at-Construction Verification Contract:**
   - At runtime constructor initialization, `YuNetDetector.__init__` reads the model file bytes, computes SHA-256, and asserts equality against `YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"`. Any hash divergence fails closed immediately with `ModelIntegrityError` (exit code `3`).

### 1B. Embedder: SFace 2021dec fp32 (`face_recognition_sface_2021dec.onnx`)

1. **Exact File Identity & Immutable SHA-256 Checksum:**
   - **Filename:** `face_recognition_sface_2021dec.onnx`
   - **Immutable Weight SHA-256:** `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
   - **Exact File Size:** `38,696,353` bytes
   - **Upstream Git Blob SHA-1:** `5817e559d509b2c1d5069f3c49a388bc45d4395f`
   - **Direct Download Source:** `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
   - **Retrieval Date:** 2026-09-12 (re-verified matching 2026-09-11 S1 spike)
2. **License Status:**
   - **Code & Weight License:** **Apache License, Version 2.0** (`CLEAR`, text-explicit)
   - **Locator:** `https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/LICENSE`
   - **Grant Terms:** Sections 1–9 grant commercial use, reproduction, modification, and distribution.
3. **Training-Data Provenance:**
   - **Status:** `PROVENANCE_UNRESOLVED` (Tracked under OpenCV Zoo Issue #313)
   - **Details:** Upstream issue `#313` remains open without maintainer resolution regarding the training dataset provenance for the 2021dec parameter weights. Kept in candidate pool per plan stop conditions; disclosed honestly in project state.
4. **Local Worktree Storage Path & Git Isolation:**
   - **Worktree Path:** `models/face_recognition_sface_2021dec.onnx`
   - **Git Status:** Denied by repository root `.gitignore` (`/models/`). Zero weight bytes are committed or tracked in Git.
5. **Hash-at-Construction Verification Contract:**
   - At runtime constructor initialization, `SFaceRecognizer.__init__` reads the model file bytes, computes SHA-256, and asserts equality against `SFACE_FP32_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"`. Any hash divergence fails closed immediately with `ModelIntegrityError` (exit code `3`).

### 1C. Auditable Download & Hash Verification Log (Arbitration `d-20260912045928673582-39`)

For full reproducibility across fresh or disposable review worktrees where `.gitignore` excludes `models/`, the authorized download and verification steps are recorded as follows:

```bash
# 1. Ensure gitignored models directory exists
mkdir -p models

# 2. Download YuNet 2023mar from authorized OpenCV Zoo media URL
curl -sSL -o models/face_detection_yunet_2023mar.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# 3. Download SFace 2021dec from authorized OpenCV Zoo media URL
curl -sSL -o models/face_recognition_sface_2021dec.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

# 4. Verified Execution Log & Checksum Output (Measured 2026-09-12 12:47:38 Asia/Taipei):
$ shasum -a 256 models/*
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  models/face_detection_yunet_2023mar.onnx
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79  models/face_recognition_sface_2021dec.onnx

$ ls -l models/*
-rw-r--r--  1 cheerc  staff    232589 Sep 12 12:47 models/face_detection_yunet_2023mar.onnx
-rw-r--r--  1 cheerc  staff  38696353 Sep 12 12:47 models/face_recognition_sface_2021dec.onnx
```

### 1D. Explicit Pair 2 Un-Downloaded & Un-Authorized Boundary

- **Pair 2 Candidates:** YuNet 2026may (dynamic-shape detector) and SFace 2021dec_int8bq (quantized embedder).
- **Authorization Status:** **NOT AUTHORIZED for download.** Operator Decision Manifest `d-20260912041106780391-34` Item 2 approved downloading Pair 1 artifacts only.
- **Governing Gate:** Pair 2 remains strictly governed by decision `d-20260911171253541089-12` preconditions (ORT standalone symbolic dimension verification, int8bq tolerance calibration).
- **Physical Absence:** Neither Pair 2 model file has been retrieved. They remain absent from disk and absent from `models/`. No download of Pair 2 is permitted without separate explicit operator authorization.

---

## 2. Consented Evaluation Corpus (P1) Read-Only Inventory

In accordance with ADR 0007 and spec §14, evaluation data is strictly segregated from production code. Biometric files remain local and untracked. The P1 corpus was audited in read-only mode to assess its readiness for chronological adaptive replay.

### 2A. Gallery and Probe Inventory Breakdown

- **Enrolled Gallery Identities ($N = 5$):**
  - `enroll-01`: 1 registration photo accepted (`enrollment refused: 0`).
  - `enroll-02`: 1 registration photo accepted (`enrollment refused: 0`).
  - `enroll-03`: 1 registration photo accepted (`enrollment refused: 0`).
  - `enroll-04`: 1 registration photo accepted (`enrollment refused: 0`).
  - `enroll-23`: 1 registration photo accepted (`enrollment refused: 0`).
  - *Total Enrolled Identifiers:* 5.
- **Target Probes by Identity:**
  - `enroll-01`: **0 target probes** (no follow-up probes available).
  - `enroll-02`: **0 target probes** (no follow-up probes available).
  - `enroll-03`: **0 target probes** (no follow-up probes available).
  - `enroll-04`: **0 target probes** (no follow-up probes available).
  - `enroll-23`: **13 target probes** (`enroll-23-probe-01.jpeg` through `enroll-23-probe-13.png`).
    - 8 probes are usable single faces meeting the frozen 0.90 detector confidence gate.
    - 5 probes are screenshots honestly refused at detector confidence gate (`invalid_input`).
- **Non-Target Evaluation Probes:**
  - 4 consented non-target probe images.
  - Phase-1A false acceptance: `0/4` at provisional anchor 0.85/0.10.

### 2B. Chronology and Timestamp Audit

- **Timestamp Distribution:**
  All 13 target probes for `enroll-23` carry timestamp signatures from a single capture session on 2026-09-11 between `15:00:23` and `15:02:09` (a total time window of 106 seconds).
- **Corroboration Implication:**
  Under the burst suppression policy (`burst_suppression_min_interval_secs = 60.0`), multiple probe submissions within 60 seconds are suppressed and cannot count toward independent event corroboration.
- **Spec §14 Environmental Condition Inventory:**
  All 10 required environmental conditions are **untested** in the real P1 corpus:
  1. Age change: `untested` (0 longitudinal samples).
  2. Hairstyle: `untested`.
  3. Eyewear: `untested`.
  4. Hats: `untested`.
  5. Pose: `untested` (all near-frontal).
  6. Lighting: `untested` (single indoor lighting setting).
  7. Complex background: `untested`.
  8. Masks: `untested`.
  9. Blur: `untested`.
  10. Occlusion: `untested`.

### 2C. Longitudinal Evaluation Gap

The P1 real corpus cannot evaluate:
1. Multi-identity cross-confusion over time (4 out of 5 identities have zero target probes).
2. Long-term template accumulation, utility eviction, and centroid drift over months or years.
3. Multi-session independent corroboration across calendar days.

### 2D. Auditable Corpus Inventory Inspection Record (Arbitration `d-20260912045928673582-39` & `d-20260912050730885132-40`)

To prove the empirical basis of the `INSUFFICIENT` verdict with exact denominator reproducibility across all approved image extensions:

```bash
# Non-sensitive inventory inspection command enumerating all approved image extensions (*.jpg, *.jpeg, *.png):
$ find "$P1_CORPUS_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l
22

# Per-role inventory breakdown (exact sum = 22):
# - Enrolled registration photos: 5 files (enroll-01..04, enroll-23; includes enroll-23.jpg)
# - Target probe photos/screenshots: 13 files (enroll-23-probe-01..13: 7 jpeg + 6 png)
# - Non-target probe photos: 4 files (consented probe unknowns: 4 jpg)
# Total approved role files: 22

# Excluded-Extension Policy:
# Non-image metadata files (*.txt, *.json, *.DS_Store), video formats (*.mp4, *.mov), and
# unapproved raw formats (*.heic, *.raw) are strictly excluded from evaluation inputs.

# Canonical Inventory Input Identification:
# The canonical inventory input is the external, untracked evaluation manifest
# mapping each of the 22 approved relative filenames to its role, identity group, and byte size.
# Timestamp of audit: 2026-09-12 12:40:15 Asia/Taipei
# Inventory Canonical Digest (SHA-256 over the 22 sorted manifest lines):
# b8c19ef684742a033f11cf8d1b19e27c1f808761aa31f98d7ebca105658e6583
```

*Privacy Boundary Guarantee:* No image files, facial crops, personal names, or local folder paths outside Git enter the repository. The audit establishes strictly that longitudinal multi-identity data is absent.

---

## 3. Corpus Decision (b) Ratification & STOP-3 Declaration

In accordance with Operator Decision Manifest `d-20260912041106780391-34` Item 3 and dispatch freeze `d-20260912044625991911-38`:

### Binding Corpus Policy:
1. **Real Replay Status:**
   Phase-1B chronological replay against real evaluation data is formally classified as **`blocked-with-reason`** due to the absence of multi-identity longitudinal probe sequences.
2. **Model Selection Gate Remains `OPEN`:**
   In strict adherence to **STOP Condition 3** (Plan §14):
   > "If ONNX weights or multi-timestamp real evaluation probes are missing under operator dual-gate rules, the team must NOT weaken the gates, fabricate data, or close the model selection gate. The replay task must mark real replay `blocked-with-reason`, evaluate synthetic streams only, keep the selection gate OPEN, and refuse to claim full Phase-1B completion."
   Candidate A (SFace Pair 1) remains **provisional**. The model selection gate is **NOT closed**.
3. **Primary Validation Vehicle:**
   Synthetic adversarial temporal streams are ratified as the **sole authorized primary vehicle** for validating Phase-1B governance mechanics, state machines, burst suppression, capacity eviction, and temporal-leakage isolation in Task 9.

---

## 4. Synthetic Adversarial Temporal Streams Specification (Frozen for Task 9)

To guarantee exhaustive and verifiable testing of all Phase-1B governance mechanics without relying on real biometric data, Task 9 must construct and execute against these seven deterministic synthetic adversarial streams:

### Stream 1: Supervised Learning & Promotion Stream
- **Purpose:** Exercises happy-path candidate creation, corroboration, and promotion.
- **Structure:**
  - Enrolls 5 synthetic identities ($I_1 \dots I_5$) with single 128-d unit embeddings.
  - Generates a sequence of 10 synthetic probe observations for $I_1$ timestamped at $t_1, t_2, \dots, t_{10}$ with $\Delta t \ge 120.0\text{ s}$ (exceeding 60s burst threshold).
  - Probe similarity scores to $I_1$: $0.91 \ge 0.88$ (`candidate_update_threshold`).
  - Probe similarity scores to $I_2 \dots I_5$: $\le 0.70$ (margin $\ge 0.21 \ge 0.12$).
  - Confirmation ground truth: `"correct"`.
- **Expected Outcome:**
  - Event $t_1$: Creates candidate $C_1$ (`status: pending`, `additional_corroboration_count: 0`). Active bank remains 1.
  - Event $t_2$: Corroborates $C_1$ (`additional_corroboration_count: 1`). Exclusivity margin passes. $C_1$ promoted to active template bank ($K=2$). New revision committed.

### Stream 2: Unsupervised / `not_me` Rejection Stream
- **Purpose:** Verifies that unconfirmed observations, explicit `not_me`, timeouts, or cancellations never learn.
- **Structure:**
  - Injects probes matching $I_2$ at score $0.93$.
  - Sub-case 2A: Confirmation is `"not_me"`.
  - Sub-case 2B: User cancels / process EOF.
  - Sub-case 2C: Confirmation timeout elapses.
- **Expected Outcome:**
  - Zero candidate records inserted into SQLite.
  - Active template bank and revision count remain completely unchanged.
  - Policy error event logged without biometric payload.

### Stream 3: Anti-Replay & Burst Suppression Stream
- **Purpose:** Verifies that rapid submissions and duplicate image hashes cannot satisfy independent corroboration.
- **Structure:**
  - Event $t_1$: Candidate $C_2$ created at $t = 100.0\text{ s}$.
  - Event $t_2$: Same image hash resubmitted at $t = 110.0\text{ s}$.
  - Event $t_3$: Distinct image probe submitted at $t = 130.0\text{ s}$ ($\Delta t = 30.0\text{ s} < 60.0\text{ s}$).
- **Expected Outcome:**
  - Event $t_2$ detected as duplicate hash: 0 corroboration credit.
  - Event $t_3$ suppressed by burst filter ($\Delta t < 60\text{ s}$): 0 corroboration credit.
  - $C_2$ remains in `status: pending` with `additional_corroboration_count = 0`. Promotion refused.

### Stream 4: Cross-Identity Margin Ambiguity Stream
- **Purpose:** Verifies that ambiguous candidates with insufficient cross-identity margin are blocked from promotion.
- **Structure:**
  - Candidate $C_3$ created for $I_1$ with match score $0.90$.
  - Evaluation against gallery finds similarity to $I_2$ is $0.88$ (margin $0.02 < 0.12$ `promotion_margin`).
  - Independent corroboration event arrives at $t + 120\text{ s}$.
- **Expected Outcome:**
  - `additional_corroboration_count` increments to 1.
  - Promotion manager evaluates 1:N exclusivity: runner-up margin $0.02$ breaches $0.12$ threshold.
  - Promotion refused; $C_3$ remains `pending` with reason `insufficient_promotion_margin`.

### Stream 5: Bank Capacity & 6-Factor Utility Eviction Stream
- **Purpose:** Verifies that when active bank reaches $K = 5$, lowest-utility template is evicted atomically.
- **Structure:**
  - Drives $I_1$ through successful promotions until active bank has 5 templates.
  - Injects 6th candidate $C_6$ passing promotion.
  - Pre-configures utility attributes:
    - Template 1 (Initial enrollment): high quality, age = 300 days (recency penalty).
    - Templates 2–4: recent, high support.
    - Template 5: duplicate near Template 2 ($\cos = 0.99$, redundancy penalty).
- **Expected Outcome:**
  - 6-factor utility rescoring executes across all 5 active templates using exact Section 6 clamped formulas.
  - Lowest-utility template is selected, its DEK is destroyed via `KeyProvider`, and row is retired atomically.
  - Initial enrollment template enjoys no special exemption.
  - Active bank size remains exactly 5; new revision recorded.

### Stream 6: Temporal-Leakage Adversarial Stream (A/B Isolation)
- **Purpose:** Verifies that decisions at event $N$ cannot read or depend upon future events $> N$.
- **Structure:**
  - Run A: Executes stream with events $e_1, e_2, \dots, e_N$.
  - Run B: Executes stream with events $e_1, e_2, \dots, e_N, e_{N+1}, \dots, e_{N+5}$ (future suffix).
  - Pre-seeds database with future candidate rows timestamped $t > t_N$.
- **Expected Outcome:**
  - Decision state, matching scores, candidate status, and active templates at event $N$ are 100% byte-identical between Run A and Run B.
  - Queries at event $N$ strictly filter `created_at <= t_N`. Any detection of future state raises `TemporalLeakageError`.

### Stream 7: Long-Horizon Drift & Bounded Response Stream
- **Purpose:** Verifies that gradual appearance drift triggers `review` and halts self-learning.
- **Structure:**
  - Synthetically drifts active templates by rotating embedding vectors away from initial anchor:
    - Step 1: Centroid shift $= 0.10 \le 0.20$ (normal matching).
    - Step 2: Centroid shift $= 0.22 > 0.20$ (`drift_max_centroid_shift`).
- **Expected Outcome:**
  - Identification status downgrades from `matched` to `review` with reason code `drift_boundary_exceeded`.
  - Identity status transitions to `re_enrollment_required`.
  - Further shadow candidate creation is blocked until trusted re-enrollment.

---

## 5. Handover to Phase 1B Implementation (Task 9 & Task 10 Readiness)

With this decision manifest complete, the technical and corpus prerequisites for Task 9 and Task 10 are frozen:

1. **Model Weights Verified:** YuNet 2023mar (`8f2383e4...`) and SFace 2021dec (`0ba9fbfa...`) verified on disk in `models/` with SHA-256 and size matching S1 specifications.
2. **Pair 2 Excluded:** YuNet 2026may and SFace int8bq remain strictly un-downloaded and un-authorized.
3. **Real Corpus Status:** Real replay is formally marked `blocked-with-reason` with documented audit digest.
4. **Gate Status:** Model selection gate remains **`OPEN`** (provisional SFace Pair 1).
5. **Synthetic Replay Authorized:** The seven synthetic adversarial stream specifications in Section 4 govern Task 9 implementation and Task 10 reporting.
6. **Task 9 & 10 Unblocked:** Task 9 may proceed to implement the replay harness against the synthetic streams, and Task 10 will emit the evaluation report contrasting Phase 1A baseline with Phase 1B governance under synthetic load.
