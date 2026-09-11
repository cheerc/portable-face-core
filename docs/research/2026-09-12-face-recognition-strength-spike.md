# Research spike — face-recognition strength vs SFace setup + 3.01.31 case

- Date: **2026-09-12** (all online evidence below fetched 2026-09-12; re-verify URLs before reliance — see §6).
- Source of truth: task `t-20260911162147982003-63720-5`, decision `d-20260911162201124576-5`, spec `docs/specs/2026-09-09-portable-face-core-design.md`, prior spikes `docs/research/2026-09-09-open-source-face-stack.md` and `docs/research/2026-09-10-model-candidate-gate.md`, rerun report `/tmp/phase-1a-bakeoff-rerun-13-probes.md` (repo-outside).
- Scope boundary (**analysis-only**): no product code written, no dependency installed, no model weight downloaded — per `research-task.md` satellite. This note is delivered via a small docs-only PR per explicit commander order (decision `d-20260911162201124576-5`); the satellite's no-PR default is noted as a deviation in §7, resolved in favor of the fresher explicit dispatch.
- Negative scope: no photos opened (filenames only); corpus manifest/embeddings/reports stay outside Git; 幼兒園相簿不讀不碰；frozen 0.90 detector 門不動（本報告不提任何調門建議）.

## 1. Is SFace / the current pipeline too weak?

Short answer: **on clean LFW protocol, SFace 2021dec fp32 (~99.4%) sits ~0.4pp below the ArcFace R100 SOTA class (~99.8%) — a small gap. The weakness actually observed in our rerun (8/13 usable, 3.01.31 misidentified) is dominated by occlusion + single-template gallery + unselected operating point, not by backbone capacity.** Switching embedders would not fix any of those three.

### 1.1 LFW-protocol accuracy table (all LFW unless noted)

| Model / artifact | Reported accuracy | Embedding dim | Source class |
|---|---|---|---|
| SFace 2021dec fp32 (our bake-off candidate) | **0.9940** | 128-d | OpenCV Zoo accuracy table (`tools/eval`) |
| SFace 2021dec block-quantized (Pair 2) | 0.9942 | 128-d | same table |
| SFace 2021dec int8-quantized | 0.9932 | 128-d | same table |
| SFace *loss* (paper, larger backbones — **not** the 2021dec artifact) | 99.82% LFW / 90.63% masked-LFW | — | [SFace paper](https://arxiv.org/abs/2205.12010) |
| MobileFaceNet + ArcFace loss (4.0 MB) | 99.55% LFW | — | MobileFaceNet literature |
| FaceNet (Google, triplet, 2015) | 99.63% LFW / 95.12% YouTube Faces | 128-d | FaceNet literature |
| InceptionResNet-V1, VGGFace2 weights (`facenet-pytorch` 20180402) | 0.9965 LFW | 512-d (default) | `facenet-pytorch` model table |
| InceptionResNet-V1, CASIA-WebFace weights (same arch) | 0.9905 LFW | 512-d (default) | same table — **training data moves accuracy more than arch here** |
| InsightFace `buffalo_l` (ResNet50@WebFace600K) | **99.80%** LFW | 512-d | InsightFace model zoo, release `v0.7` (`buffalo_l.zip`) |
| InsightFace `antelopev2` (ResNet100@Glint360K) | ~99.83% class | 512-d | InsightFace model zoo |
| ArcFace R100 (MS1MV2 / Glint360K, survey figures) | 99.80–99.85% LFW | 512-d | 2025–2026 surveys (LFW saturated ~99.8%) |

Key links (pinned verbatim from sources): [InsightFace repo](https://github.com/deepinsight/insightface) · [`buffalo_l.zip` at release v0.7](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) · [OpenCV Zoo](https://github.com/opencv/opencv_zoo) · [SFace reference implementation](https://github.com/zhongyy/SFace) · [Zoo issue #313 (SFace provenance, OPEN)](https://github.com/opencv/opencv_zoo/issues/313).

### 1.2 Hard-gate filter (§13) before any accuracy comparison

- **InsightFace models are "available for non-commercial research purposes only."** Our §13 hard gate requires commercial use + redistribution text-explicit. As stated, InsightFace **fails the license gate** — its 99.8% is not purchasable without a separate license/provenance pass and an operator decision. Accuracy alone never admits a candidate.
- SFace 2021dec stays admissible with its recorded caveats (Apache-2.0 directory claim + open issue #313 `PROVENANCE_UNRESOLVED`, per the 2026-09-10 gate report). Pair 2 int8bq costs only ~0.08pp on LFW (0.9940 → 0.9932) for ~3.6× smaller weights (38.7 MB → 10.7 MB).
- Any non-Zoo ArcFace-class ONNX export would need a fresh license/provenance/checker pass per the nine research gates — none has one on file.

**Conclusion for Q1:** SFace is a legitimate lightweight-class choice (~99.4%), not a broken one. The ~0.4pp SOTA gap is real but second-order next to the three factors in §2.

## 2. 3.01.31 case — separate attribution (model vs occlusion vs single-template vs threshold-policy)

Measured facts on file (forensics rescoring + rerun report, filenames only, no image opened):

| Fact | Value |
|---|---|
| Detector confidence | **0.9051** — marginal pass over frozen 0.90 |
| Display-anchor outcome | `review (0.4114, 0.0687)` (match≥0.363, margin≥0.1 anchor — **not selected**) |
| Top-1 at anchor | **enroll-02 (non-target) 0.4114** vs true enroll-23 0.3427 |
| Margin top-1 − true | 0.0687 (< 0.1 anchor margin) |
| Probe condition | sunglasses (upper-face occlusion), screenshot |

Attribution, separated:

- **(a) Model capacity — minor.** A 128-d MobileFaceNet-class embedder at ~99.4% LFW still resolves clean frontal pairs; nothing in the 0.4114/0.3427 pair implicates capacity limits. A 512-d ArcFace model would face the same occluded input.
- **(b) Occlusion — major.** Sunglasses = real-world upper-face occlusion. Published evidence: ArcFace-100 Top-1 drops **99.57% → 86.60%** with sunglasses (ROF dataset); MobileFaceNet-class drops **98.89% → 77.16%**; masked-LFW costs SOTA models **5–16pp** (MLFW benchmark); original ArcFace masked-LFW **94.75 vs 99.62** clean (MT-ArcFace paper); NIST-cited mask error rates 5–50%. The SFace-loss paper itself reports **99.82% → 90.63%** LFW→masked-LFW. 3.01.31's distortion is the expected direction and magnitude.
- **(c) Single-template gallery — major.** Gallery holds one photo per identity with no appearance variation (spec §7 one-shot). Literature: mean-aggregated 10-image templates beat single-image **1.8×** (match distance 0.7 → 0.4, Hofer et al., CelebA + ArcFace 512-d); single-image-per-class **93.50%** vs multi-image **98.20%** (Hossain et al., FaceNet 128-d study); **3 best images** (frontal + two profiles) reach 0.315 vs 0.291 for all images — i.e. most of the gain comes from the first few *semantically different* views. Our gallery has exactly one view; a sunglasses probe has no nearby anchor by construction.
- **(d) Threshold-policy — neutral, correctly undecided.** The (0.363, 0.1) anchor is display-only, never selected. Margin 0.0687 < 0.1 and top-1 being a non-target identity is precisely the shape `review` exists for — and the operator's product context (human confirmation menu as the final screen) is the designed sink for exactly this case. No policy failure; no tuning indicated (frozen gate stands).

**Conclusion for Q2:** occlusion × single-template explains the case; model capacity does not; threshold policy behaved as designed.

## 3. Recommendations (with costs + spec/plan amendment flags)

1. **KEEP SFace (2021dec fp32, Pair 1) through Phase-1B.** Reasons: already baked-off under the frozen one-shot protocol; portable (ORT, 128-d, 38.7 MB; int8bq 10.7 MB fallback at −0.08pp); license posture recorded with explicit caveats; switching gains ≤0.4pp on clean data and zero on occluded data. Cost of keep: none beyond the scheduled Phase-1B replay.
2. **Do NOT switch embedders now.** Costs of switch: fresh §13 license/provenance/checker pass for the new artifact; new template generation + re-embedding of all retained exemplars (old/new generations never compare); full bake-off rerun under the frozen protocol; InsightFace-class options additionally blocked on non-commercial licensing; mobile-checker + cross-runtime tolerance re-proof. Flag: requires an operator decision + plan amendment before any artifact download (CLAUDE.md hard boundary).
3. **Do NOT change enrollment to multi-photo.** Cost: amends spec §7 one-shot enrollment + Product Direction ("one student, one capture action, one accepted still photo") + Phase-1A/1B boundary contracts. The bounded multi-template accumulation the literature recommends **already exists** in the design as the Phase-1B confirmation-gated shadow-candidate flow (§9: cumulative but bounded bank, no permanent anchor, first-event-cannot-promote-itself, temporally-independent corroboration). Let occluded/angled observations enter through that guarded path instead of widening enrollment.
4. **If operator wants stronger occlusion robustness later**, the evidence-backed options in order are: (i) capture-condition guidance (sunglasses-off retry — cheapest, no spec change); (ii) Phase-1B replay measuring shadow-bank growth on varied observations (no spec change, already planned); (iii) a *second* bake-off candidate in the ArcFace class under the same frozen protocol (needs operator decision + license clearance + §13/§14 plan amendment). Never (iv) lowering the 0.90 detector gate or any threshold to fit the 13 probes (calibrating-to-data, prohibited).

### Spec / plan amendment flags touched by the above

- Rec. 2 (switch) → amends plan bake-off candidate list + `ModelManifest` generation + Phase-1B replay baseline. Needs operator decision first.
- Rec. 3-enrollment-variant (rejected) → would amend spec §7, §9 bank seeding, Product Direction registration line, CLI enrollment contract. **Not proposed.**
- Rec. 4(iii) → amends plan §13 candidate table only; spec protocol text unchanged (same frozen one-shot method).

## 4. Unverified scope (not established by this spike)

- Exact pixel scale/mean/std inside SFace `alignCrop`/`feature` (carried `UNVERIFIED` from the 2026-09-10 gate report; Task 6 business, needs artifact bytes).
- fp32↔int8bq numeric tolerance on our runtime (same carry).
- Zoo issue #313 status after 2026-09-11 (re-check at next download decision).
- Second-hand literature figures in §1–§2 whose locators are unpinned (Hofer/Hossain/MT-ArcFace/MLFW/ROF numbers surfaced via web search 2026-09-12; author/title recorded, exact URLs **not pinned** — re-verify from primary sources before citing them in any operator-facing selection claim).
- No new measurements were run; rerun/forensics numbers are cited from `/tmp` artifacts, not reproduced here.

## 5. Next decisions (for Lead / commander)

- D1: Accept rec. 1–4 (keep SFace; no switch; no enrollment change; occlusion handled via capture guidance + Phase-1B bank)?
- D2: If stronger robustness is still wanted, authorize rec. 4(iii) second-candidate bake-off (with license clearance) — or defer to post-Phase-1B evidence?
- D3: Proceed with work order 2 (verification-grade report format, `t-20260911162155004235-63720-6`) after this spike's PR opens?

## 6. Retrieval log (2026-09-12)

- Queries: `InsightFace ArcFace LFW 2025 2026 SOTA`; `SFace MobileFaceNet LFW 128-d`; `InsightFace buffalo_l ONNX release Glint360K`; `FaceNet VGGFace2 LFW embedding comparison`; `masked sunglasses occlusion ArcFace drop`; `multi-template averaging vs single template`.
- Pinned URLs (§1.1 link list + prior spikes' locators). Unpinned second-hand figures flagged in §4 — no URL fabricated for them.

## 7. Exclusions and deviations

- No product code, dependency, or weight touched (analysis boundary kept).
- Deviation note: `research-task.md` says research tasks create no branch/worktree/PR; the explicit commander work order (decision `d-20260911162201124576-5`: "docs/research/ note via small docs PR", branch `docs/face-recognition-strength-spike`) postdates and overrides it for delivery mechanics only. Analysis boundary itself was fully kept.
- Photos/manifest/embeddings/reports not in Git (this note cites only filenames + aggregate scores already delivered station-externally).
