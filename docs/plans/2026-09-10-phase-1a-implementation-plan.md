# Phase 1A Implementation Plan

**Source of truth:** `docs/specs/2026-09-09-portable-face-core-design.md`, approved by the operator on 2026-09-10 and merged to `main` at `f2a68a83c9548d61396ddf3cafa80e9c3d951c11`. Governing decisions: ADR 0004 (ONNX-first), ADR 0006 (Phase-1A/1B split), ADR 0007 (small consented gallery).

**Goal:** A macOS static-image CLI and evaluation harness that, for two to three compliant ONNX model candidates, produces a threshold-sweep operating table with exact denominators over a three-to-five-identity consented gallery, so the operator can select or reject a model and operating point — while writing no durable biometric state.

**Architecture:** A Python package `facecore` exposing five separable layers — adapter (file decode), face pipeline (quality, single-face detection, alignment, embedding), repository (revision-shaped, in-memory only in Phase 1A), identification policy (aggregation, open-set bands), and an evaluation harness (bake-off, threshold sweep, capacity benchmark, conformance). A `facecore.sh` shell entrypoint wraps the CLI. Every contract that Phase 1B will persist is defined and versioned in Phase 1A but backed only by an in-memory repository.

**Tech stack:** Python, ONNX Runtime, NumPy, Pillow for decode/EXIF, pytest, ruff, mypy. No database, no network at inference time, no server.

## Global Constraints

- Phase 1A persists no biometric database, face crop, or embedding. Process exit must leave no biometric state on disk.
- Exactly one usable face per enrollment and probe image. Zero-face and multi-face inputs are `invalid_input`, never a partial identity.
- Real biometric files, corpus manifests naming real people, and reports containing per-image identity detail stay outside Git.
- No Phase-1B governance: no persistence, no `KeyProvider`, no encryption, no shadow candidates, no confirmation flow, no promotion, no retirement, no rollback, no deletion, no export/import, no chronological replay.
- No `authenticated` state. Decision states are exactly `matched`, `review`, `unknown`, `invalid_input`.
- Network availability must never change a threshold or a result. Inference runs offline.
- No model artifact is downloaded, vendored, or pinned as a dependency until Spike S1 records its exact code license, weight license, redistribution right, SHA-256, and training-data provenance.
- Every `tokio::spawn`-equivalent long-lived background task is out of scope; Phase 1A is synchronous and single-process.

## Evidence and Open Hypotheses

**Established (source-backed):**

- The approved spec fixes the CLI surface, decision states, exit codes, result schema shape, quality-policy obligations, evaluation staging, and Phase-1A functional acceptance list — `docs/specs/2026-09-09-portable-face-core-design.md` §§3, 4, 7, 8, 11, 13, 14.
- ONNX Runtime is the approved runtime; ONNX Runtime Mobile is the intended mobile runtime — ADR 0004 and `docs/research/2026-09-09-open-source-face-stack.md`.
- YuNet (detector) states MIT and SFace (embedder) states Apache-2.0 in their OpenCV Zoo model directories — `docs/research/2026-09-09-open-source-face-stack.md`, "Candidate Model Stack".
- OpenCV Zoo issue #313 leaves SFace weight training-data provenance unresolved; the directory's Apache-2.0 claim stands but a recorded review is required before redistribution — same file, "Compliance Caveat".
- dlib models are CC0-1.0 / public-domain-released and are a valid fallback comparison backend — same file, "Alternative: dlib".
- DeepFace's MIT license does not extend to the models it wraps, so "DeepFace" is not a satisfiable license answer — same file, "Not Safe as a Blanket Dependency Choice".
- The repository `.gitignore` already denies `*.jpg/*.png/*.heic/...`, `/models/`, `/data/`, `/embeddings/`, `/exports/`, and SQLite artifacts — `.gitignore` at `f2a68a8`.
- The reference machine for this plan is Apple M1, 8 cores, arm64, macOS 26.6.2 — measured 2026-09-10 via `uname -m`, `sysctl -n machdep.cpu.brand_string`, `sw_vers`.

**Unproven — each has a named resolution method:**

- *Which artifacts clear the licensing and provenance gates.* Two to three must clear, or spec §13 requires reporting the shortage rather than weakening the gate. Resolved by Spike S1.
- *Exact artifact SHA-256, version, and direct URL for every candidate.* Not recorded anywhere yet. Resolved by Spike S1.
- *ONNX Runtime Mobile operator support, dynamic-shape behavior, and accelerator findings per candidate.* Resolved by Spike S1 using the model usability checker.
- *Whether an `onnxruntime` wheel exists for the system Python 3.14.7 on macOS arm64.* The system interpreter is 3.14.7; ONNX Runtime wheel coverage for a very recent CPython is not verified. Resolved by Spike S1 step 7; if absent, the plan pins a supported interpreter in `pyproject.toml` rather than assuming.
- *Whether three to five consented identities with usable enrollment photos are actually obtainable.* This is a human and consent dependency, not a code dependency. Resolved by Prerequisite P1 before Task 9 can run.
- *Every threshold value.* Phase 1A sets no accuracy number before calibration data exists (spec §14). Thresholds are swept, not chosen.

## Documentation impact check

Classification: **`area`**.

Evidence: adding `docs/plans/` changes the repository map in `README.md:29-45`, and completing Phase 1A changes `docs/PROJECT-STATE.md` "Current Status" and "Next Session". Both are area-level project documents, not architecture or ADR changes — the architectural decisions are already recorded in ADR 0004/0006/0007 and this plan introduces no new one.

- The `README.md` repository-map and `docs/PROJECT-STATE.md` next-session updates for *this plan document* are part of Task 0 and land in the same commit as the plan.
- A `project-docs-maintain` check covering `README.md` and `docs/PROJECT-STATE.md` is **Task 13**, and it is a **blocking dependency of the Phase-1A completion report** — not merely an acceptance line.

## Prerequisite P1 — consented evaluation corpus (blocks Task 9 only)

Not a coding task. Owner: operator.

- Obtain three to five explicitly consented identities, one registration photo each, plus consented non-target probe faces, plus later target probes grouped by identity.
- Record consent, guardian authorization where required, retention deadline, and deletion procedure **outside** the Face Core identity schema and outside Git (spec §14).
- Files live under a local path the operator chooses; nothing enters the repository.
- Until P1 completes, Tasks 1–8 and 10–13 proceed unblocked using deterministic synthetic fixtures; only Task 9 (real-gallery bake-off execution) waits.

**Stop condition:** if P1 cannot be satisfied, Task 9 does not run and the Phase-1A report states the gallery as `unavailable`; the model-selection gate stays open rather than being decided on synthetic data.

## Spike S1 — model candidate discovery and licensing gate (time-boxed, blocks Task 6)

**Question:** which two to three detector+embedder artifact pairs clear the spec §13 hard gates, and what are their exact identities?

**Owner:** implementer. **Time box:** one working session. **Branch/artifact:** `docs/research/2026-09-10-model-candidate-gate.md` on branch `docs/model-candidate-gate`, merged before Task 6 begins.

**Expected evidence per candidate**, following the nine research gates already listed in `docs/research/2026-09-09-open-source-face-stack.md`:

1. exact version, immutable SHA-256, source repository, release/tag, direct artifact URL, retrieval date;
2. code license and weight license recorded separately, each with a locator;
3. explicit statement of commercial use, modification, and redistribution rights;
4. known training datasets and any unresolved provenance (SFace must cite issue #313 status as of the retrieval date);
5. `python -m onnxruntime.tools.check_onnx_model_mobile_usability` output: unsupported operators, dynamic-shape issues, fallback and accelerator findings;
6. embedding dimension, input tensor layout, expected scale/mean/std, and score direction;
7. confirmation that an `onnxruntime` wheel exists for the interpreter this project will pin on macOS arm64, with the exact version.

**Stop conditions:**

- Fewer than two candidates clear the license and provenance gates → stop; write the shortage report spec §13 requires, list every excluded candidate with its failed gate and evidence, and escalate the pause-or-revise decision to the operator. Do not weaken a gate to reach two.
- A candidate's provenance is merely *unclear* rather than *refused* → record it as `provenance_unresolved`, keep it in the bake-off, and surface it in the operator's selection report. Uncertainty is disclosed, not silently resolved either way.

**Formal work reuses the spike:** Task 6's `ModelManifest` fixtures are populated verbatim from S1's recorded fields; no manifest value is retyped from memory.

## Frozen Phase-1A defaults

These are the measurable definitions spec §7 and §14 require the plan to choose. All are **provisional and versioned**; the bake-off may show any of them to be wrong, which is the point of running it.

**Quality policy `quality_policy_version: 1`:**

| Gate | Definition | Default | Reason code on failure |
|---|---|---|---|
| detector confidence | detector's own score | `>= 0.90` | `quality_detector_confidence` |
| usable face size | face bounding-box shorter side, pixels | `>= 112` | `quality_face_too_small` |
| sharpness | variance of Laplacian over the aligned crop | `>= 60.0` | `quality_blurry` |
| exposure | mean luma of aligned crop in `[40, 215]` and clipped-pixel fraction `<= 0.05` | as stated | `quality_exposure` |
| yaw | absolute yaw estimated from the five landmarks | `<= 30°` | `quality_pose_yaw` |
| pitch | absolute pitch estimated from the five landmarks | `<= 20°` | `quality_pose_pitch` |
| occlusion | count of landmarks below the detector's per-landmark confidence | `0 of 5 below 0.5` | `quality_occluded` |

**Identity aggregation `aggregation_strategy: single_template_passthrough`.** In Phase 1A every identity holds exactly one template, so the identity score is that template's score; median, trimmed mean, and support rules are degenerate. The aggregation interface is frozen so Phase 1B can add `trimmed_mean_top_k` without a contract change, but no multi-template strategy is implemented now. This is the justified choice spec §8 asks for: with one template per identity, a robust aggregate and a maximum are the same number, and implementing an unexercised strategy would be untested code.

**Comparison stage — exact boundary.** The budget below is meaningless without saying what it times, so the boundary is defined by a principle rather than a list, to keep it from being gamed: **every step whose cost grows with the number of enrolled identities is inside the comparison stage.**

- Inside: similarity computation of the probe embedding against all active templates, per-identity aggregation, ranking, top-1/top-2 margin computation, and decision-band assignment.
- Outside: decode, EXIF orientation, quality gating, detection, alignment, ONNX embedding inference, report assembly, and file I/O. None of these scale with identity count.

An implementer may not move an N-scaling step into the end-to-end series to make the comparison series pass. Task 11's test asserts the boundary by scaling the gallery from 50 to 500 and requiring the comparison series to grow while the end-to-end remainder stays flat within noise.

**Comparison-latency budget** (spec §14 requires this be declared *before* benchmarking): for the comparison stage as defined above, one probe against 500 active templates, single process, on the reference machine (Apple M1, arm64, macOS 26.6.2): **p95 ≤ 1.0 ms, p99 ≤ 2.0 ms**.

Basis, so the number is auditable rather than arbitrary: a measured NumPy floor for the 500×512 matmul plus top-2 selection is p95 ≈ 0.013 ms on the reference machine (2026-09-10). The budget allows roughly 75× that floor to cover per-identity aggregation, banding, and interpreter overhead, while still failing loudly if the comparison stage is implemented as a per-identity Python loop — which would land near 0.5 ms and consume most of the allowance. An earlier draft of this plan set 10 ms, which is roughly 790× the measured floor and could not fail; that budget would have been decorative.

The report must record machine, interpreter, NumPy and ONNX Runtime versions, and must show comparison time separately from end-to-end time. A run on a materially different machine class does not inherit this budget; the report says so and the operator re-declares.

**Decision bands.** `matched` requires identity score `>= match_threshold` **and** top-1/top-2 margin `>= margin_threshold`; `review` requires score `>= review_threshold`; otherwise `unknown`. No numeric value is chosen in this plan — Task 10 sweeps them and the operator picks from the table.

## File and Dependency Map

**Create:**

| Path | Responsibility |
|---|---|
| `pyproject.toml` | package metadata, pinned deps, ruff/mypy/pytest config, pinned interpreter |
| `facecore.sh` | shell entrypoint wrapping `python -m facecore.cli` |
| `src/facecore/__init__.py` | package marker, `SCHEMA_VERSION = 1` |
| `src/facecore/cli.py` | argparse surface for `init`, `evaluate`, `bakeoff`, `conformance`; exit-code mapping |
| `src/facecore/errors.py` | exception types carrying spec §11 exit codes |
| `src/facecore/contracts/result.py` | `IdentificationResult`, `Decision`, `Quality`, reason-code enums, JSON encoder |
| `src/facecore/contracts/policy.py` | `PolicyProfile` incl. quality thresholds, bands, aggregation strategy, capacity |
| `src/facecore/contracts/template.py` | `FaceTemplate`, `TemplateGeneration`, `TemplateRevision` (revision-shaped, not persisted) |
| `src/facecore/contracts/manifest.py` | `ModelManifest`: id, hashes, licenses, provenance status, IO contract, dims, precision |
| `src/facecore/repository/base.py` | `Repository` protocol — revision-shaped read/write surface Phase 1B will persist |
| `src/facecore/repository/memory.py` | `InMemoryRepository`, process-local, discarded at exit |
| `src/facecore/pipeline/decode.py` | decode, EXIF orientation, color order normalization |
| `src/facecore/pipeline/quality.py` | the `quality_policy_version: 1` gates above |
| `src/facecore/pipeline/detect.py` | detector adapter, exactly-one-face enforcement |
| `src/facecore/pipeline/align.py` | landmark alignment, normalized crop, padding, interpolation |
| `src/facecore/pipeline/embed.py` | ONNX Runtime session, tensor layout/scale/normalization, L2-normalized output |
| `src/facecore/policy/identify.py` | identity aggregation, margin, open-set band assignment, reason codes |
| `src/facecore/eval/session.py` | in-memory evaluation session: enroll N identities, identify probes |
| `src/facecore/eval/corpus.py` | corpus-manifest schema + loader (manifest itself stays outside Git) |
| `src/facecore/eval/bakeoff.py` | run all candidates over one frozen gallery and probe order |
| `src/facecore/eval/sweep.py` | threshold-sweep operating table with exact denominators |
| `src/facecore/eval/capacity.py` | synthetic 500-vector comparison capacity and latency benchmark |
| `src/facecore/eval/report.py` | non-biometric report writer + redaction assertions |
| `src/facecore/conformance/fixtures.py` | deterministic face-free fixture generator (created in Task 3, first consumer; reused by Task 12) |
| `src/facecore/conformance/check.py` | macOS self-conformance against committed expected JSON |
| `tests/conformance/expected/*.json` | committed expected values (non-biometric) |
| `tests/` | one test module per source module below |
| `docs/plans/2026-09-10-phase-1a-implementation-plan.md` | this plan |

**Modify:**

- `README.md:29-45` — repository map gains `docs/plans/`, `src/`, `tests/`, `facecore.sh`, `pyproject.toml` as they land.
- `docs/PROJECT-STATE.md` — "Current Status" and "Next Session" at plan landing and again at Phase-1A completion.
- `.gitignore` — add `/reports/`, `/corpus/`, `/build/` (Task 10 owns this edit). `.venv/` is already covered.

**Interfaces and dependency order:**

```text
contracts/*          (no deps)
  → repository/base  → repository/memory
  → pipeline/decode → pipeline/quality → pipeline/detect → pipeline/align → pipeline/embed
  → policy/identify  (consumes repository + contracts)
  → eval/session     (consumes policy + repository + pipeline)
  → eval/bakeoff, eval/sweep, eval/capacity (consume eval/session)
  → eval/report      (consumes all eval outputs)
conformance/*        (consumes pipeline/decode + align + embed only)
cli.py               (consumes everything; owns exit codes)
```

**Data / auth / migration / rollback / runtime:** no auth, no migration, no rollback — all Phase 1B. Runtime is a single synchronous process. Data is process-local and discarded at exit.

## Tasks

### Task 0: Land this plan and repoint project docs

**Files:**
- Create: `docs/plans/2026-09-10-phase-1a-implementation-plan.md`
- Modify: `README.md` repository map; `docs/PROJECT-STATE.md` "Current Status" and "Next Session"

**Interfaces:** Consumes the approved spec at `f2a68a8`. Produces the reviewed plan every later task cites.

**Acceptance:**
- The repository map lists `docs/plans/`.
- `docs/PROJECT-STATE.md` states that the Phase-1A plan exists and is pending review, and that implementation begins only after that review.
- No source tree is added by this task.

**Test-first evidence:**
- Failing case: repository map omits a directory that exists on disk.
- RED: `diff <(sed -n '/```text/,/```/p' README.md | grep -oE '^[a-z]+/' | sort -u) <(find . -mindepth 1 -maxdepth 1 -type d -not -name '.*' | sed 's|^\./||;s|$|/|' | sort)` → shows `docs/plans/` missing from the map. (BSD `find` on macOS has no `-printf`; the `sed` form is portable.)
- Minimal behavior: add the `plans/` entry and the two PROJECT-STATE edits.
- GREEN: the same command → no difference.
- Project verification: `grep -rn "AGENTS.md" . --include="*.md"` → no hits; `git diff --check origin/main...HEAD` → clean.

### Task 1: Frozen contracts

**Files:**
- Create: `src/facecore/__init__.py`, `src/facecore/contracts/result.py`, `src/facecore/contracts/policy.py`, `src/facecore/contracts/template.py`, `src/facecore/contracts/manifest.py`, `src/facecore/errors.py`, `pyproject.toml`
- Test: `tests/contracts/test_result.py`, `test_policy.py`, `test_template.py`, `test_manifest.py`

**Interfaces:**
- Produces: `IdentificationResult(schema_version:int, status:Literal["matched","review","unknown","invalid_input"], identity:Identity|None, decision:Decision, quality:Quality, model_version:str, template_revision:int, candidate_created:bool)`; `PolicyProfile`; `FaceTemplate`; `TemplateRevision`; `ModelManifest`; `FaceCoreError` subclasses carrying `exit_code`.

**Acceptance:**
- A `matched` result serializes to exactly the key set in spec §11's example, `schema_version: 1`.
- `review`, `unknown`, and `invalid_input` serialize with `identity: null` and never a guessed `display_name`.
- `candidate_created` is present and always `false` in Phase 1A.
- `template_revision` is present and is `1` for a one-shot evaluation identity.
- Exit-code mapping matches spec §11 exactly: normal outcomes `0`; undecodable input `2`; model integrity/compatibility `3`; store/key `4` (reserved, unreachable in 1A); invalid configuration `5`; unexpected internal `7`.

**Test-first evidence:**
- Failing case: constructing a result with `status="authenticated"`.
- RED: `.venv/bin/python -m pytest tests/contracts -q` → `ModuleNotFoundError: No module named 'facecore'`, then after scaffolding, `Failed: DID NOT RAISE <class 'ValueError'>`.
- Minimal behavior: `status` constrained to the four states; `to_json()` emitting the frozen key order; `FaceCoreError.exit_code` per subclass.
- GREEN: `.venv/bin/python -m pytest tests/contracts -q` → all pass.
- Project verification: `.venv/bin/ruff check src tests` and `.venv/bin/mypy src` → clean.

### Task 2: Revision-shaped repository and in-memory implementation

**Files:**
- Create: `src/facecore/repository/base.py`, `src/facecore/repository/memory.py`
- Test: `tests/repository/test_memory.py`

**Interfaces:**
- Consumes: `contracts/template.py`, `contracts/manifest.py`
- Produces: `Repository` protocol with `create_identity`, `get_identity`, `list_active_templates`, `current_revision`, `append_revision`; `InMemoryRepository` implementing it.

**Acceptance:**
- `append_revision` never mutates an existing revision; revision numbers increase monotonically.
- The protocol exposes no persistence, encryption, candidate, promotion, retirement, or deletion method — Phase 1B adds those behind the same protocol.
- Two `InMemoryRepository` instances share no state.
- Nothing is written to disk: a test asserts the process's working tree is byte-identical before and after a full enroll+identify cycle.

**Test-first evidence:**
- Failing case: enrolling an identity, then asserting no file appeared anywhere under the repository root.
- RED: `.venv/bin/python -m pytest tests/repository -q` → `AttributeError: 'InMemoryRepository' object has no attribute 'append_revision'`.
- Minimal behavior: dataclass-backed dict store; `append_revision` copies rather than mutates.
- GREEN: `.venv/bin/python -m pytest tests/repository -q` → all pass.
- Project verification: `.venv/bin/mypy src` → clean, including protocol conformance.

### Task 3: Decode, orientation, and color-order normalization

**Files:**
- Create: `src/facecore/pipeline/decode.py`, `src/facecore/conformance/fixtures.py`
- Test: `tests/pipeline/test_decode.py`

The deterministic fixture generator lands here rather than in Task 12 because Task 3 is its first consumer — the eight EXIF-orientation cases below have to be generated, and `.gitignore` denies committing them as images. Task 12 reuses the same generator for the full Layer-A set.

**Interfaces:**
- Produces: `decode_image(data: bytes) -> DecodedImage` with deterministic RGB order and EXIF orientation applied.

**Acceptance:**
- All eight EXIF orientation values produce the same upright pixel array from equivalent inputs.
- Undecodable bytes raise the exit-code-`2` error type, never `invalid_input`.
- Color order is RGB and asserted against a committed expected value.

**Test-first evidence:**
- Failing case: an EXIF-orientation-6 image decoding sideways.
- RED: `.venv/bin/python -m pytest tests/pipeline/test_decode.py -q` → assertion diff on the rotated array.
- Minimal behavior: Pillow decode, `ImageOps.exif_transpose`, explicit RGB conversion.
- GREEN: same command → pass.
- Project verification: `.venv/bin/python -m pytest tests/ -q` → no regression.

### Task 4: Quality gate

**Files:**
- Create: `src/facecore/pipeline/quality.py`
- Test: `tests/pipeline/test_quality.py`

**Interfaces:**
- Consumes: `DecodedImage`, detector output, `PolicyProfile`
- Produces: `QualityVerdict(status, reason_codes)` using `quality_policy_version: 1`.

**Acceptance:**
- Each of the seven gates in the frozen-defaults table fires its own reason code, proven by one synthetic case per gate.
- Multiple simultaneous failures return all applicable reason codes, not just the first.
- A passing case returns `status="accepted"`, `reason_codes=[]`.

**Test-first evidence:**
- Failing case: a deliberately blurred synthetic crop scoring below the sharpness floor.
- RED: `.venv/bin/python -m pytest tests/pipeline/test_quality.py -q` → `assert [] == ['quality_blurry']`.
- Minimal behavior: the seven measurable gates, each returning its code.
- GREEN: same command → pass, seven parameterized cases.
- Project verification: `.venv/bin/ruff check src tests` → clean.

### Task 5: Single-face detection and alignment

**Files:**
- Create: `src/facecore/pipeline/detect.py`, `src/facecore/pipeline/align.py`
- Test: `tests/pipeline/test_detect.py`, `tests/pipeline/test_align.py`

**Interfaces:**
- Consumes: `DecodedImage`, detector adapter
- Produces: `DetectedFace(box, landmarks, confidence)`; `align(face) -> AlignedCrop` with versioned resize, crop, padding, and interpolation.

**Acceptance:**
- Zero detected faces → `invalid_input` with `input_no_face`.
- Two or more detected faces → `invalid_input` with `input_multiple_faces`; the request is refused rather than picking the largest face.
- Alignment is deterministic: the same input yields a byte-identical crop across runs.

**Test-first evidence:**
- Failing case: a stubbed detector returning two faces still producing a result.
- RED: `.venv/bin/python -m pytest tests/pipeline/test_detect.py -q` → `assert 'matched' == 'invalid_input'`.
- Minimal behavior: count check before any downstream call; refuse on `!= 1`.
- GREEN: same command → pass.
- Project verification: `.venv/bin/python -m pytest tests/ -q` → no regression.

### Task 6: ONNX embedding with model manifest

**Files:**
- Create: `src/facecore/pipeline/embed.py`
- Test: `tests/pipeline/test_embed.py`

**Interfaces:**
- Consumes: `AlignedCrop`, `ModelManifest` populated from Spike S1
- Produces: L2-normalized `numpy.ndarray` embedding plus the manifest's `model_version`.

**Dependency:** Spike S1 must be merged first.

**Acceptance:**
- Embedding L2 norm is `1.0` within `1e-6`.
- A manifest whose recorded SHA-256 does not match the artifact on disk fails closed with the exit-code-`3` error, before any inference.
- Comparing templates across two different `model_version` values raises rather than returning a score.
- No network call occurs during inference, asserted by a socket-blocking fixture.

**Test-first evidence:**
- Failing case: a tampered artifact whose hash no longer matches its manifest.
- RED: `.venv/bin/python -m pytest tests/pipeline/test_embed.py -q` → `Failed: DID NOT RAISE ModelIntegrityError`.
- Minimal behavior: hash the artifact at session construction, compare to manifest, raise before creating the ORT session.
- GREEN: same command → pass.
- Project verification: `.venv/bin/mypy src` → clean.

### Task 7: Identification policy

**Files:**
- Create: `src/facecore/policy/identify.py`
- Test: `tests/policy/test_identify.py`

**Interfaces:**
- Consumes: probe embedding, `Repository`, `PolicyProfile`
- Produces: `IdentificationResult` with `score`, `runner_up_score`, `margin`, `threshold`, `reason_codes`.

**Acceptance:**
- Band assignment follows the frozen rule: `matched` needs score **and** margin; `review` needs score only; else `unknown`.
- With a single enrolled identity, `runner_up_score` and `margin` are `null` and a `matched` verdict records `margin_unavailable_single_identity` in `reason_codes` — the small-gallery limitation is visible in the record, not silent.
- With multiple identities, `runner_up_score` is the second-best **identity** score, not the second-best template score.
- Reason codes distinguish `match_threshold_met`, `below_match_threshold`, `insufficient_margin`, and `below_review_threshold`.
- `aggregation_strategy` is recorded in the result so a Phase-1B change is detectable.

**Test-first evidence:**
- Failing case: a probe clearing `match_threshold` but failing `margin_threshold` still returning `matched`.
- RED: `.venv/bin/python -m pytest tests/policy -q` → `assert 'matched' == 'review'`.
- Minimal behavior: evaluate margin as a conjunct, not an afterthought.
- GREEN: same command → pass.
- Project verification: `.venv/bin/python -m pytest tests/ -q` → no regression.

### Task 8: Evaluation session and `evaluate` command

**Files:**
- Create: `src/facecore/eval/session.py`, `src/facecore/cli.py`, `facecore.sh`
- Test: `tests/eval/test_session.py`, `tests/cli/test_evaluate.py`

**Interfaces:**
- Consumes: pipeline, policy, `InMemoryRepository`
- Produces: `./facecore.sh init`, `./facecore.sh evaluate --enrollment <path> --probe <path>`.

**Acceptance:**
- `evaluate` prints a human-readable summary and stable JSON.
- Invalid enrollment leaves no session state: a second `evaluate` in the same process behaves as if the first never ran.
- Decoded `invalid_input` exits `0`; undecodable input exits `2`.
- After the process exits, no face crop, embedding, or identity database exists on disk — asserted by comparing a recursive checksum of the working tree before and after.

**Test-first evidence:**
- Failing case: a zero-face enrollment image creating a partial identity.
- RED: `.venv/bin/python -m pytest tests/eval/test_session.py -q` → `assert 1 == 0` on identity count.
- Minimal behavior: build the identity only after the quality and single-face gates both pass.
- GREEN: same command → pass.
- Project verification: generate two fixtures into the gitignored build directory, then `./facecore.sh evaluate --enrollment build/fixtures/synthetic_a.png --probe build/fixtures/synthetic_b.png; echo $?` → `0`. No fixture image is committed; `.gitignore` denies `*.png` and `/build/`.

### Task 9: Corpus manifest and bake-off harness

**Files:**
- Create: `src/facecore/eval/corpus.py`, `src/facecore/eval/bakeoff.py`
- Test: `tests/eval/test_corpus.py`, `tests/eval/test_bakeoff.py`

**Dependency:** Prerequisite P1 for a real run; the tests themselves use a synthetic manifest and do not need P1.

**Interfaces:**
- Consumes: a corpus manifest **stored outside the repository**, `ModelManifest` list from S1
- Produces: `./facecore.sh bakeoff --corpus <manifest>` and a per-candidate result set.

**Acceptance:**
- The manifest schema carries per-file role (`enrollment`, `target_probe`, `non_target_probe`), identity label, and condition labels (age change, hairstyle, eyewear, hats, pose, lighting, complex background, masks, blur, occlusion).
- Loading a manifest that points inside the repository working tree is refused with the exit-code-`5` configuration error — biometric files must not be in Git.
- Every candidate receives the same enrollment gallery, the same probe order, and the same policy interface; a test asserts identical probe ordering across candidates.
- Each identity is enrolled from exactly one registration photo, asserted by the harness.
- A condition with zero samples is emitted as `untested`, never as a zero error rate.
- The bake-off emits a **corpus inventory table with one row per condition for all ten spec §14 conditions** — age change, hairstyle, eyewear, hats, pose, lighting, complex background, masks, blur, occlusion — each carrying its sample count. All ten rows are always present; a condition absent from the manifest renders as `untested`, not as a missing row. The table is emitted even when the corpus is unavailable under Prerequisite P1, in which case every row reads `untested` — coverage is documented, and a gap is visible rather than silent.
- The report states the spec §14 limitation that Phase-1A evaluation is biased toward posed single-person inputs, because exactly one usable face per image is required.

**Test-first evidence:**
- Failing case: a manifest whose enrollment path resolves under the repository root; and separately, a manifest naming only three conditions rendering an inventory table with three rows instead of ten.
- RED: `.venv/bin/python -m pytest tests/eval/test_corpus.py tests/eval/test_bakeoff.py -q` → `Failed: DID NOT RAISE ConfigurationError`, then `assert 3 == 10` on inventory row count.
- Minimal behavior: resolve the path, compare against the repository root, refuse on containment; render the inventory from the fixed ten-condition list rather than from the manifest's observed keys.
- GREEN: same command → pass.
- Project verification: `.venv/bin/python -m pytest tests/ -q` → no regression.

### Task 10: Threshold-sweep operating table and non-biometric report path

**Files:**
- Create: `src/facecore/eval/sweep.py`, `src/facecore/eval/report.py`
- Modify: `.gitignore` — add `/reports/`, `/corpus/`, `/build/`
- Test: `tests/eval/test_sweep.py`, `tests/eval/test_report.py`

**Interfaces:**
- Consumes: bake-off result set
- Produces: per-candidate operating table over a threshold grid.

**Acceptance:**
- Every row carries target `matched`/`review`/`unknown` counts, non-target false acceptances, and **exact denominators**.
- Per-identity outcomes, runner-up margins, and a confusion-matrix row exist for every enrolled identity.
- Zero observed failures is rendered as `0/N`, never as a rate, and never as "zero error rate".
- **Minimum denominator for rate rendering is 30.** Below it the cell shows counts only, as `k/N`. At or above it a rate may appear, but `k/N` is still shown alongside — a rate never replaces its denominator anywhere in the report. Phase-1A galleries are three to five identities, so most cells will be counts-only by construction; that is the intended outcome, not a degraded one.
- The table is produced for every candidate model, including any marked `provenance_unresolved`.
- Reports are written under `./reports/`, which `.gitignore` denies. A report containing an embedding value, a face crop, a full local path, or a per-image identity label fails a redaction assertion in `eval/report.py` before the file is written — the guard is code, not a reviewer's habit.

**Test-first evidence:**
- Failing case: a table row rendering `0.0%` for a denominator of 4; and separately, a report body carrying a raw embedding vector.
- RED: `.venv/bin/python -m pytest tests/eval/test_sweep.py tests/eval/test_report.py -q` → `assert '0.0%' == '0/4'`, and `Failed: DID NOT RAISE RedactionError`.
- Minimal behavior: render counts over denominators and forbid rate formatting below the minimum denominator of 30; scan the assembled report for float arrays, absolute paths, and identity labels before writing.
- GREEN: same command → pass.
- Project verification: `.venv/bin/ruff check src tests` → clean.

### Task 11: Synthetic capacity and latency benchmark

**Files:**
- Create: `src/facecore/eval/capacity.py`
- Test: `tests/eval/test_capacity.py`

**Interfaces:**
- Consumes: embedding dimension from `ModelManifest`
- Produces: comparison-latency distribution at 500 synthetic identities.

**Acceptance:**
- Comparison time is reported separately from end-to-end time.
- The report records machine, interpreter, NumPy and ONNX Runtime versions.
- The declared budget (p95 ≤ 1.0 ms, p99 ≤ 2.0 ms on the reference machine) is asserted, and a breach fails the acceptance rather than being noted.
- The comparison stage matches the boundary defined in the frozen defaults: the gallery is swept at 50, 100, 250, and 500 identities, and the test asserts the comparison series grows with identity count while the end-to-end remainder stays flat within noise. This is what stops an N-scaling step being reclassified as end-to-end.
- The report states in prose that synthetic vectors measure capacity and latency only, and estimate no false acceptance, ranking quality, margin behavior, or policy-trigger frequency.

**Test-first evidence:**
- Failing case: a benchmark reporting a single blended end-to-end number; and separately, a comparison series that does not grow between a 50-identity and a 500-identity gallery, which proves an N-scaling step was timed outside it.
- RED: `.venv/bin/python -m pytest tests/eval/test_capacity.py -q` → `KeyError: 'comparison_p95_ms'`, then `assert 0.98 > 1.5` on the 500-vs-50 growth ratio.
- Minimal behavior: time the comparison stage as bounded above, in isolation; emit both series across the four gallery sizes.
- GREEN: same command → pass.
- Project verification: `./facecore.sh bakeoff --corpus tests/fixtures/synthetic-corpus.json` → capacity section present in the report.

### Task 12: Deterministic Layer-A fixtures and `conformance`

**Files:**
- Create: `src/facecore/conformance/check.py`, `tests/conformance/expected/*.json`
- Consumes: `src/facecore/conformance/fixtures.py` from Task 3, extended here to the full twelve-case Layer-A set
- Test: `tests/conformance/test_check.py`

**Interfaces:**
- Produces: `./facecore.sh conformance`.

**Acceptance:**
- Fixtures are generated at test time and are face-free: gradients, checkerboards, chroma patterns, EXIF-orientation cases, synthetic tensors. No image is committed.
- Committed expected JSON covers decoding, orientation, color order, resize, crop, padding, interpolation, tensor layout, scaling, normalization, embedding normalization, and numeric encoding — twelve named cases.
- `./facecore.sh conformance` reproduces every committed value or exits non-zero naming the first divergence.
- The report states explicitly that Layer-A fixtures do **not** validate detector or landmark equivalence on real faces.

**Test-first evidence:**
- Failing case: a deliberately altered interpolation mode.
- RED: `.venv/bin/python -m pytest tests/conformance -q` → divergence on the resize case.
- Minimal behavior: pin interpolation, regenerate, compare to committed JSON.
- GREEN: same command → pass, twelve cases.
- Project verification: `./facecore.sh conformance; echo $?` → `0`.

### Task 13: Project-docs maintenance pass (blocking)

**Files:**
- Modify: `README.md`, `docs/PROJECT-STATE.md`

**Dependency:** blocks the Phase-1A completion report. Not an acceptance line on another task.

**Acceptance:**
- Invoke `project-docs-maintain` over `README.md` and `docs/PROJECT-STATE.md`.
- The repository map matches the tree on disk.
- `docs/PROJECT-STATE.md` records Phase-1A completion, the selected-or-rejected model, the operating point, and the Phase-1B go/no-go as the next gate.
- `README.md` status line no longer says planning is the next artifact.

**Test-first evidence:**
- Failing case: the repository map omits `src/` after the source tree lands.
- RED: the Task 0 map-vs-disk diff command → shows `src/` and `tests/` missing.
- GREEN: same command → no difference.
- Project verification: `grep -rn "Phase-1A implementation planning is the next artifact" README.md` → no hits.

## PR Boundaries

| PR | Tasks | Independent acceptance |
|---|---|---|
| PR-A | Task 0 | Plan document lands; repository map and PROJECT-STATE match disk. No code. |
| PR-B | Spike S1 | Model candidate gate report merged, or the spec §13 shortage report plus an operator escalation. |
| PR-C | Tasks 1, 2 | Contracts and in-memory repository with their own tests. No user-visible behavior; foundation consumed by every later PR. |
| PR-D | Tasks 3, 4, 5 | Decode, quality, detection, alignment — a probe image becomes a deterministic aligned crop or a reason-coded refusal. |
| PR-E | Task 6 | Embedding with manifest integrity gate. Depends on PR-B and PR-D. |
| PR-F | Tasks 7, 8 | `init` and `evaluate` work end to end on synthetic fixtures. First user-visible behavior. |
| PR-G | Tasks 9, 10 | `bakeoff` produces the operating table. Depends on PR-F. |
| PR-H | Task 11 | Capacity benchmark. Independent of PR-G — it needs only an embedding dimension. |
| PR-I | Task 12 | `conformance` passes. Independent of PR-G and PR-H. |
| PR-J | Task 13 | Docs match the delivered tree. Last. |

Tasks 1 and 2 share PR-C because merging a `Repository` protocol with no implementation, or an implementation with no protocol, leaves a broken state. Tasks 3–5 share PR-D for the same reason: an aligner with no detector output to align is unreachable code. PR-H and PR-I are deliberately separate from PR-G — each is an independently observable behavior change and neither breaks the other.

## Out of Scope for Phase 1A

Restating so review can check for creep: persistent storage, encryption, `KeyProvider`, shadow candidates, confirmation flow, `--confirm-learning`, promotion, corroboration, retirement, utility eviction, rollback, deletion, re-enrollment, export/import, chronological adaptive replay, drift indicators, multi-face images, video, camera, Android, iOS, REST, attendance rules, PhotoPrism coupling, real-face cross-runtime fixtures, and any `authenticated` state.

## Highest Risks

1. **Spike S1 finds fewer than two compliant candidates.** Spec §13 already dictates the response — report the shortage, do not weaken the gate — but it would stall Phase 1A at the model gate. Detected early because S1 blocks Task 6.
2. **Prerequisite P1 does not materialize.** Without a consented gallery there is no runner-up evidence and no defensible operating point. Tasks 1–8 and 10–13 still deliver, so the loss is bounded to the accuracy claim rather than the whole milestone.
3. **The frozen quality defaults are wrong.** They are guesses pending calibration; the bake-off may show the sharpness floor or yaw bound rejects usable photos. Mitigated by versioning the policy and treating the defaults as swept parameters, not constants.
4. **Python 3.14.7 has no ONNX Runtime wheel.** Would force an interpreter pin change before any inference code runs. Detected by S1 step 7, ahead of Task 6.
5. **A small gallery invites over-reading.** Every report path in Tasks 10 and 11 carries an explicit non-extrapolation statement for exactly this reason.
