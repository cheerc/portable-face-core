# Spike S1 — Model Candidate Licensing and Provenance Gate

- Retrieval date: **2026-09-11** (all online evidence below was fetched and re-verified on this date; the 2026-09-09 research note was treated as evidence-not-approval and every URL was re-checked).
- Source of truth: `docs/plans/2026-09-10-phase-1a-implementation-plan.md` Spike S1 (frozen at `79f1e60`), spec §13 hard gates, decision `d-20260910155215736226-0`.
- Scope boundary: **read-only investigation only** — no dependency installed, no model artifact downloaded. Every item that requires the artifact bytes is marked `unverified` with a named completion step, never recorded as passing.
- Pairs in bake-off: **2** (Pair 1, Pair 2). Both carry `provenance_unresolved` flags and stay in the bake-off per the plan's stop conditions. No shortage report is triggered (shortage requires *fewer than two* pairs surviving license+provenance; zero pairs were *refused*).

## Gate legend

- `CLEAR` — evidence on file, independently re-verifiable via the locator.
- `PROVENANCE_UNRESOLVED` — not refused; kept in bake-off, surfaced to the operator.
- `UNVERIFIED` — cannot be established without the artifact bytes or a local run; must complete before Task 6. Never counted as a pass.
- `EXCLUDED (failed gate)` — refused for the stated gate, with evidence.

---

## Pair 1 (leading) — YuNet 2023mar (detector) + SFace 2021dec (embedder)

### 1A. Detector: `face_detection_yunet_2023mar.onnx`

1. **Exact version / SHA-256 / source / tag / URL / retrieval date**
   - File: `face_detection_yunet_2023mar.onnx`, hosted in `opencv/opencv_zoo`, path `models/face_detection_yunet/`, branch `main` (no release tag; Zoo releases `4.10.0`/`4.9.0` carry no model assets — verified on the releases page 2026-09-11).
   - Direct artifact URL: `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` (documented `curl -L -o` pattern); media URL `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`.
   - Git blob identity (locator only, **not** a weight checksum): git-SHA-1 `2d8804a5986e229f1fde3a1994feacc66c91b58b`, size 232,589 bytes, via `api.github.com/repos/opencv/opencv_zoo/contents/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` on 2026-09-11.
   - Weight SHA-256: `UNVERIFIED` — no download per scope; Task 6 must hash the artifact at session construction and compare to the manifest (plan Task 6 acceptance already requires this fail-closed check).
   - Retrieval date: 2026-09-11.
2. **Code license vs weight license, separately with locators** — `CLEAR`
   - Model directory license file: `https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE` — **MIT License**, copyright `© 2020 Shiqi Yu <shiqi.yu@gmail.com>`, grant text: *"to use, copy, modify, merge, publish, distribute, sublicense, and/or sell"* (first 8 lines fetched verbatim 2026-09-11).
   - The Zoo directory presents this LICENSE as covering the model files (same-directory license convention, stated in each wrapper header: *"subject to the license terms in the LICENSE file found in the same directory"* — `yunet.py`, `sface.py` headers fetched 2026-09-11). Code (wrapper/demo) and weights share the one stated license; recorded here as a single MIT claim with that caveat, not as two independent licenses.
3. **Commercial use / modification / redistribution** — `CLEAR` (text-explicit)
   - The MIT grant text explicitly names commercial-adjacent rights (`sell`), modification (`modify`, `merge`), and redistribution (`publish`, `distribute`, `sublicense`). This is a statement of what the license *text* says, not legal advice.
4. **Known training datasets / unresolved provenance** — `PROVENANCE_UNRESOLVED`
   - Known: model source `https://github.com/ShiqiYu/libfacedetection.train` (commit `a61a428` cited by the Zoo README); detects faces ~10×10–300×300 px *"due to the training scheme"*; evaluated 0.834/0.824/0.708 AP-easy/med/hard on **WIDER Face validation** (eval set, not claimed as training set).
   - Unresolved: the exact training corpus composition is not enumerated in the Zoo directory; the README defers to the external training repo (*"For details on training this model, please visit https://github.com/ShiqiYu/libfacedetection.train"*). The training repo's file list was not audited in this spike. Kept in bake-off with this flag.
5. **onnxruntime mobile-usability checker output** — `UNVERIFIED`
   - Not run (requires the `.onnx` bytes). Required command before Task 6: `python -m onnxruntime.tools.check_onnx_model_mobile_usability --model <artifact>`. Record unsupported ops, dynamic-shape findings, fallback and accelerator recommendations verbatim at that time.
6. **Embedding dims / IO contract** (detector output contract)
   - Output: `N × 15` `CV_32F` rows of `[x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y, rcm_x, rcm_y, lcm_x, lcm_y, score]` — box + 5 landmarks (right eye, left eye, nose tip, right/left mouth corner) + confidence (corroborated by the upstream `demo.py` visualizer parsing `det[0:4]`, `det[4:14]`, `det[-1]`, fetched 2026-09-11).
   - `2023mar` file has **fixed input shape** (OpenCV 4.x DNN infers on the input image shape; see `opencv_zoo` issue #44 cited in README). Constructor params from `yunet.py`: `inputSize=[w,h]`, `confThreshold` (demo default 0.9), `nmsThreshold` (0.3), `topK` (5000); `setInputSize` must track the current image.
   - Direction: higher `score` = likelier face; filtered by `score_threshold`.
7. **onnxruntime wheel for the pinned interpreter on macOS arm64** — `CLEAR (existence)` / local install `UNVERIFIED`
   - Reference machine interpreter measured 2026-09-11: system Python **3.14.7**, `arm64`, macOS 26.6.2.
   - PyPI JSON (`https://pypi.org/pypi/onnxruntime/json`, fetched 2026-09-11): latest **1.29.0** (released 2026-08-17), `requires_python >= 3.11`, classifiers include Python 3.14, and asset **`onnxruntime-1.29.0-cp314-cp314-macosx_14_0_arm64.whl`** (CPython 3.14, macOS 14.0+ ARM64) exists. Existence confirmed without installing anything; the local venv install is Task 1's business.

### 1B. Embedder: `face_recognition_sface_2021dec.onnx`

1. **Exact version / SHA-256 / source / tag / URL / retrieval date**
   - File: `face_recognition_sface_2021dec.onnx`, `opencv/opencv_zoo`, path `models/face_recognition_sface/`, branch `main`, no release tag (same releases-page finding as above).
   - Media URL: `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`.
   - Git blob identity (locator only): git-SHA-1 `5817e559d509b2c1d5069f3c49a388bc45d4395f`, size 38,696,353 bytes, via the same contents API on 2026-09-11.
   - Weight SHA-256: `UNVERIFIED` — same no-download reason and Task 6 hash-at-construction requirement as 1A.
   - Retrieval date: 2026-09-11.
2. **Code license vs weight license, separately with locators** — `CLEAR`
   - Model directory license file: `https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/LICENSE` (11,358 bytes) — full **Apache License, Version 2.0, January 2004** text (first 12 lines fetched verbatim 2026-09-11); README states *"All files in this directory are licensed under Apache 2.0 License."*
   - Same same-directory-license caveat as 1A: one stated license covering code + weights, recorded as such.
3. **Commercial use / modification / redistribution** — `CLEAR` (text-explicit, with provenance caveat below)
   - Apache-2.0 §§1–9 grant use, reproduction, modification, distribution (the fetched header confirms the license identity; the full grant is in the 11,358-byte file at the locator). This records the license text's grant, **not** a resolution of the training-data question in item 4 — redistribution of the weights remains gated on item 4's review per CLAUDE.md (*"Do not bundle or download a model until … training-data provenance [has] been reviewed and recorded"*).
4. **Known training datasets / unresolved provenance** — `PROVENANCE_UNRESOLVED` (must cite)
   - Known: MobileFaceNet instances trained with the SFace loss ([SFace paper](https://arxiv.org/abs/2205.12010)); ONNX conversion by Chengrui Wang from [the original code base](https://github.com/zhongyy/SFace); Zoo accuracy table 0.9940 (fp32) / 0.9942 (block) / 0.9932 (quant).
   - Unresolved: **upstream issue [#313](https://github.com/opencv/opencv_zoo/issues/313) is OPEN as of 2026-09-11** — filed 2026-07-22, no maintainer reply visible, asking (a) whether the directory Apache-2.0 claim extends to the `2021dec` weight parameters for commercial prediction, (b) which corpus produced that parameter set (requester notes prior unlinked mentions of CASIA-WebFace, VGGFace2, MS1MV2), (c) whether source-data rules permit commercial deployment. No corpus is linked to this exact file by any maintainer statement. Kept in bake-off with this flag, per plan stop conditions.
5. **onnxruntime mobile-usability checker output** — `UNVERIFIED` (same required pre-Task-6 command as 1A, run against this artifact).
6. **Embedding dims / IO contract**
   - **128-dimensional** embedding (multiple corroborating sources); input is a **112×112 aligned BGR face crop** produced by `FaceRecognizerSF.alignCrop` (5-landmark warping; *"Supporting 5-landmark warping for now"* — Zoo README).
   - Post: L2-normalize, then cosine similarity (same-person threshold **0.363** in `sface.py`) or norm-L2 distance (threshold **1.128**); larger cosine passes, smaller L2 passes (`sface.py` `match()`, fetched verbatim 2026-09-11).
   - `UNVERIFIED` sub-item: the exact pixel scale/mean/std inside `FaceRecognizerSF.feature`/`alignCrop` (OpenCV-internal) was not established from primary sources in this spike — secondary reports mention `blobFromImage(swapRB=true)` on the BGR crop, but that is **not** recorded as fact. Task 6 must pin the preprocessing contract against the downloaded artifact and record it in `ModelManifest`; cross-runtime equivalence (spec gate 7) is untested.
7. **onnxruntime wheel** — same finding as 1A item 7 (shared runtime): `CLEAR (existence)`, local install `UNVERIFIED`.

### Pair 1 gate summary

| Gate (§13) | Verdict |
|---|---|
| Licenses allow commercial use + redistribution (text-explicit, locators on file) | CLEAR |
| Source / version / URL / retrieval date recorded | CLEAR (weight SHA-256 excepted) |
| Training-data provenance recorded | PROVENANCE_UNRESOLVED (SFace #313 open; YuNet corpus composition unaudited) |
| Inference under ORT on macOS | UNVERIFIED (no download, no run) |
| ORT-Mobile checker findings | UNVERIFIED (command staged for pre-Task-6) |
| Preprocessing / output contracts reproducible + versioned | PARTIAL (shapes, dims, thresholds on file; pixel normalization UNVERIFIED) |

Pair 1 **stays in the bake-off** with flags. No gate was weakened to keep it.

---

## Pair 2 — YuNet 2026may (dynamic-shape detector) + SFace 2021dec_int8bq (quantized embedder)

Rationale: exercises the dynamic-input detector path and the quantized embedder as a size/latency comparison point. License and provenance posture is identical to Pair 1 (same directories, same LICENSE files, same issue #313); only artifact identity and shape behavior differ, recorded below.

### 2A. Detector: `face_detection_yunet_2026may.onnx`

1. **Identity**: `models/face_detection_yunet/face_detection_yunet_2026may.onnx` on `main`; media URL `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx`; git blob SHA-1 `a9f99d9750ffadac632540d0360bd5a1e1873e36`, 229,738 bytes (contents API, 2026-09-11). Weight SHA-256 `UNVERIFIED` (no download). Default model of the current upstream `demo.py` (`--model` default `face_detection_yunet_2026may.onnx`, fetched 2026-09-11). Retrieval date 2026-09-11.
2. **Licenses**: same MIT locator and grant as 1A — `CLEAR`.
3. **Commercial / modification / redistribution**: same MIT text — `CLEAR`.
4. **Provenance**: re-export of `2023mar` with static H/W dims replaced by symbolic dims; same training lineage and same unresolved corpus composition as 1A — `PROVENANCE_UNRESOLVED`.
5. **Checker output**: `UNVERIFIED` — additionally watch dynamic-shape findings (symbolic `height`/`width`), which is the point of including this file.
6. **IO contract**: same N×15 output as 1A; **dynamic input shape** (any resolution without resizing) — but the README ties dynamic dims to the *OpenCV 5.x ONNX Runtime engine* (`OPENCV_FORCE_DNN_ENGINE=4`), while our approved runtime is standalone `onnxruntime`, not OpenCV DNN. Whether the symbolic dims behave under standalone ORT is `UNVERIFIED` and is a named pre-Task-6 check; if it fails, Pair 2's detector falls back to the 2023mar file (Pair 1) rather than weakening any gate.
7. **Wheel**: same as 1A — `CLEAR (existence)`.

### 2B. Embedder: `face_recognition_sface_2021dec_int8bq.onnx`

1. **Identity**: `models/face_recognition_sface/face_recognition_sface_2021dec_int8bq.onnx` on `main`; media URL `https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8bq.onnx`; git blob SHA-1 `c9acf218af0ae9d58b255b9f4eb5759c24505ca5`, 10,667,852 bytes (contents API, 2026-09-11). Block-quantized int8 (`block_quantize.py`, `block_size=64`); Zoo eval accuracy 0.9932 vs 0.9940 fp32. Weight SHA-256 `UNVERIFIED`. Retrieval date 2026-09-11.
2. **Licenses**: same Apache-2.0 locator as 1B — `CLEAR`.
3. **Commercial / modification / redistribution**: same grant text — `CLEAR`, same item-4 redistribution review caveat.
4. **Provenance**: same file lineage, same open issue #313 — `PROVENANCE_UNRESOLVED`.
5. **Checker output**: `UNVERIFIED` — additionally watch quantized-op support on the mobile checker.
6. **IO contract**: same 128-d / 112×112 / thresholds as 1B; numeric tolerance between fp32 and int8bq embeddings is `UNVERIFIED` (needs both artifacts + a run; Task 6/9 business, not assumed here).
7. **Wheel**: same as 1A — `CLEAR (existence)`.

### Pair 2 gate summary

Same verdicts as Pair 1, plus two pair-specific `UNVERIFIED` items: symbolic-dims behavior under standalone ORT (2A) and fp32↔int8 numeric tolerance (2B). **Stays in the bake-off** with flags.

---

## Fallback (comparison backend only, NOT a §13 bake-off candidate)

### dlib: `mmod_human_face_detector` (detector) + `dlib_face_recognition_resnet_model_v1` (embedder)

- **License** — `CLEAR` (text-explicit, locators on file):
  - Library: dlib states *"licensed under the Boost Software License"* and *"you can use dlib however you like, even in closed source commercial software"* (repo page, fetched 2026-09-11).
  - Models: `dlib-models` README (fetched verbatim 2026-09-11): *"anyone can do whatever they want with these model files as I've released them into the public domain"*; page footer labels the repo `CC0-1.0`. (Scope note: the 68-point landmark models carry a separate Imperial College commercial-use warning — **not** our files; our fallback uses the 5-point model + resnet model, which carry no such warning.)
- **Provenance** — documented but mixed: resnet trained from scratch on ~3M faces / 7,485 identities from **FaceScrub + VGG-Face + author web scrapes** (cleaned, LFW-overlap avoided); LFW 99.38% (with jitter) / 99.13%; metric loss, radius-0.6 balls, decision threshold 0.6. The *"large number of images I scraped from the internet"* portion has no per-image rights chain — recorded as an unresolvable source-data caveat, though the author's public-domain release is the operative license statement.
- **IO contract**: detector is a CNN MMOD (training collection documented: ImageNet/AFLW/Pascal VOC/VGG/WIDER/FaceScrub ex-FDDB); landmarks via `shape_predictor_5_face_landmarks` (7,198 author-labeled faces); embedder input is a **150×150 aligned `face_chip`**, output **128-d** descriptor, Euclidean distance, same-person threshold **0.6**.
- **Exclusion gate**: the artifacts are dlib `.dat` (non-ONNX) and the path is C++/`dlib` — it **fails the §13 hard gate "inference runs under ONNX Runtime"** as stated. An ONNX conversion is conceivable but was not investigated (out of scope; would itself need a fresh license/provenance pass on the converted artifact). **Excluded from the bake-off; retained only as a fallback comparison backend**, consistent with the 2026-09-09 note.
- Explicitly not candidates: DeepFace (wrapper MIT does not cover wrapped model licenses), CompreFace (server/AVX, not portable), Immich/InsightFace (project-specific permission, non-transferable) — per the 2026-09-09 evidence, re-affirmed 2026-09-11 without re-download.

---

## Pre-Task-6 completion list (all `UNVERIFIED` above, in dependency order)

1. Download the four Pair 1/2 artifacts; record weight **SHA-256** into `ModelManifest` fixtures verbatim (no retyping from memory); enable Task 6's hash-at-construction fail-closed check.
2. Run `python -m onnxruntime.tools.check_onnx_model_mobile_usability` per artifact; paste unsupported-ops / shape / fallback / accelerator output into this report's successor fields before Task 6.
3. Confirm `2026may` symbolic dims under **standalone** `onnxruntime` (not OpenCV DNN); on failure, drop 2A to `2023mar`.
4. Pin the SFace pixel scale/mean/std + channel order against the downloaded artifact; record fp32↔int8bq tolerance.
5. Re-check issue #313 status at Task 6 start; any maintainer provenance statement updates Pair 1/2 item 4.

## Exclusions and deviations

- No dependency installed, no model downloaded, no checker run — per dispatch scope; all affected fields say `UNVERIFIED`, none say pass.
- Git blob SHA-1 values are recorded as identity locators only and are **not** weight checksums; the report states this wherever they appear.
- The 2026-09-09 note's claim set was re-verified URL-by-URL; where a primary source was unreachable (OpenCV API docs returned HTTP 403 to the fetcher), the report uses upstream raw files (`demo.py`, `yunet.py`, `sface.py`, READMEs, LICENSEs) and says so — secondary search results informed wording for 128-d/112×112 only where corroborated by ≥2 sources, and remaining pixel-normalization detail is still `UNVERIFIED`.
- No Phase-1B mechanics, no real photos, no `project-docs-maintain` pass (that is Task 13's blocking business, not this spike's).

## Known issues

- SFace redistribution remains doubly gated: the Apache-2.0 text grants it, but open issue #313 leaves the training-data basis unanswered — the operator's selection report must surface both facts together.
- No release tags carry the Zoo ONNX files; artifact identity rests on branch-`main` paths + contents-API blob SHAs + (pending) weight SHA-256. Any `main` movement before Task 6 download must re-verify identity.
- `onnxruntime` 1.29.0 wheel existence for cp314/arm64 is confirmed via PyPI metadata only; build-from-source risk is not assessed.
