# Model selection report — no-weights skeleton (deliverable, gates unweakened)

> 歷史證據快照：本文的 no-weights、unrun、pending 與數字描述原始小 gallery，不是最新派工入口。最新 M1–M6 交付與限制見 [PROJECT-STATE](PROJECT-STATE.md)；不得由本文重派已完成的 Pair2 或真圖 replay，也不得把舊水位當目前 production 預設。

- Date: **2026-09-12**. Status: **SKELETON** — deliverable without weights; upgrades to full version if weights are lawfully obtained (per d-20260911174907590232-15).
- Source of truth: skeleton d-20260911174907590232-15, evidence list d-20260911171231615360-11, selection plan v2 d-20260911165640512131-7, task t-20260911174759975945-63720-9.
- Scope boundary: **docs-only** — no weight download (STOP clause holds), frozen 0.90 detector gate untouched, photos never in repo, Pair 2 comparison + rename-after rerun deferred to weights dual-gate (not in this PR).
- Selection gate: **OPEN** (unchanged). Nothing in this report selects or rejects a model; §6 options go to the operator.

## 1. Pair 1 rerun evidence (YuNet 2023mar + SFace 2021dec fp32)

Locator: repo-outside rerun artifact `/tmp/phase-1a-bakeoff-rerun-13-probes.md` (71 lines; source `reports/phase-1a-bakeoff-rerun.md`, gitignored). Numbers carried over, method unchanged.

- Gallery: enroll-01, enroll-02, enroll-03, enroll-04, enroll-23 — enrollment refused: 0.
- Target probes usable: **8/13** (5 screenshots honestly refused at frozen 0.90: no single face). Non-target scored: 4/4.
- Provisional anchor 0.85/0.10 (**not selected**): matched 0/8, review 0/8, unknown 8/8, FA **0/4**.

### 1.1 Threshold-sweep operating table (match × margin grid)

- match>=0.00 margin>=0.00: matched 8/8, review 0/8, unknown 0/8, FA 4/4
- match>=0.05 margin>=0.50: matched 0/8, review 8/8, unknown 0/8, FA 4/4
- match>=0.15 margin>=0.45: matched 0/8, review 8/8, unknown 0/8, FA 4/4
- match>=0.25 margin>=0.40: matched 0/8, review 7/8, unknown 1/8, FA 3/4
- match>=0.35 margin>=0.35: matched 0/8, review 5/8, unknown 3/8, FA 3/4
- match>=0.45 margin>=0.30: matched 0/8, review 1/8, unknown 7/8, FA 3/4
- match>=0.55 margin>=0.25: matched 0/8, review 0/8, unknown 8/8, FA 1/4
- match>=0.65 margin>=0.20: matched 0/8, review 0/8, unknown 8/8, FA 1/4
- match>=0.75 margin>=0.15: matched 0/8, review 0/8, unknown 8/8, FA 1/4
- match>=0.85 margin>=0.10: matched 0/8, review 0/8, unknown 8/8, FA 0/4
- match>=0.95 margin>=0.05: matched 0/8, review 0/8, unknown 8/8, FA 0/4

N<30 throughout: counts only (`k/N`), never rates.

### 1.2 Per-probe detail (PR #16 self-contained format; renamed basenames)

Rename mapping (repo-external, 辨識組 only; 註冊組 untouched; photos never in repo):

| New (primary) | Old |
|---|---|
| enroll-23-probe-01..07.jpeg | 影像 2/3/4/5/6/7/.jpeg |
| enroll-23-probe-08.png | 截圖 2026-09-11 下午3.00.23(failed sample).png |
| enroll-23-probe-09.png | 截圖 2026-09-11 下午3.00.59(failed sample).png |
| enroll-23-probe-10.png | 截圖 2026-09-11 下午3.01.16(failed sample).png |
| enroll-23-probe-11.png | 截圖 2026-09-11 下午3.01.31.png |
| enroll-23-probe-12.png | 截圖 2026-09-11 下午3.01.59(failed sample).png |
| enroll-23-probe-13.png | 截圖 2026-09-11 下午3.02.09(failed sample).png |

Scores/margins from rerun evidence; `predicted` is **pending** except probe-11 (forensics-measured), per STOP clause — no fabrication:

- enroll-23-probe-01.jpeg → unknown (top1 0.3399, margin 0.0151) | expected enroll-23, predicted pending
- enroll-23-probe-02.jpeg → matched (top1 0.4577, margin 0.1385) | expected enroll-23, predicted pending
- enroll-23-probe-03.jpeg → unknown (top1 0.3376, margin 0.1037) | expected enroll-23, predicted pending
- enroll-23-probe-04.jpeg → review (top1 0.3852, margin 0.0756) | expected enroll-23, predicted pending
- enroll-23-probe-05.jpeg → review (top1 0.3665, margin 0.0161) | expected enroll-23, predicted pending
- enroll-23-probe-06.jpeg → unknown (top1 0.2434, margin 0.0224) | expected enroll-23, predicted pending
- enroll-23-probe-07.jpeg → review (top1 0.3769, margin 0.0687) | expected enroll-23, predicted pending
- enroll-23-probe-08.png → invalid_input (n/a, n/a) | expected enroll-23, predicted n/a
- enroll-23-probe-09.png → invalid_input (n/a, n/a) | expected enroll-23, predicted n/a
- enroll-23-probe-10.png → invalid_input (n/a, n/a) | expected enroll-23, predicted n/a
- enroll-23-probe-11.png → review (top1 0.4114, margin 0.0687) | expected enroll-23, predicted enroll-02 (measured)
- enroll-23-probe-12.png → invalid_input (n/a, n/a) | expected enroll-23, predicted n/a
- enroll-23-probe-13.png → invalid_input (n/a, n/a) | expected enroll-23, predicted n/a

### 1.3 Confusion + inventory

- enroll-01/02/03/04: target probes=0 (no confusion measurable — gallery runners-up exist only as non-target scorers).
- enroll-23: target probes=13, matched=1, review=4, unknown=3, invalid_input=5.
- Corpus inventory (10 conditions — age_change, hairstyle, eyewear, hats, pose, lighting, complex_background, masks, blur, occlusion): **all untested**.

## 2. LFW literature gap (cited_pending_primary)

Locator: spike PR #15 (`docs/research/2026-09-12-face-recognition-strength-spike.md`, main@06c0463).

- SFace 2021dec fp32 (Zoo accuracy table): **0.9940**; int8bq 0.9932 (−0.08pp, ~3.6× smaller).
- ArcFace R100 class (surveys; LFW saturated ~99.8%): **~99.80–99.85%**.
- Gap ≈ **0.4pp** on clean protocol — second-order next to occlusion × single-template (spike §2: sunglasses drops published at 5–16pp masked / 99.57→86.60% sunglasses for ArcFace-100).
- Literature figures flagged `cited_pending_primary` in the spike: re-verify from primaries before any operator-facing selection claim.

## 3. Synthetic comparison-latency baseline

Locator: d-20260911171347364900-13 (codex independent measurement).

- n=500: p95 **0.0195ms**, p99 0.0199ms — ~50× margin under the frozen 1.0ms budget.
- Scope of claim: capacity + latency only. **Not** FAR / ranking / margin / policy-trigger estimates (plan Task 11 verbatim).

## 4. License / provenance posture

Locator: S1 gate report (`docs/research/2026-09-10-model-candidate-gate.md`).

- Pair 1 + Pair 2: **provenance_unresolved** (SFace Zoo issue #313 OPEN; YuNet corpus composition unaudited). Stays flagged, no gate weakened.
- InsightFace-class options: **fail §13** ("non-commercial research purposes only") — 99.8% not purchasable without separate clearance + operator decision.
- Any non-Zoo ArcFace-class artifact needs a fresh nine-gate license/provenance/checker pass — none on file.

## 5. PhotoPrism section (bounded conditions, §1.x)

Bounded conditions (unchanged; no PR #15 amendment): PhotoPrism is architecture evidence only — not a dependency, no coupling, no model approval derived from it. Borrowed concepts: detector/embedding versioning independent; model change ⇒ re-embed all faces; normalized embeddings make cosine/Euclidean predictable; manual bad-match feedback changes policy/calibration, never retrains a foundation model. Locator: `docs/research/2026-09-09-open-source-face-stack.md` (§Reference Product).

## 6. Options (B void per plan v2)

- **A — 推薦（維持 provisional）**: keep SFace Pair 1 provisional; gate stays OPEN; Phase-1B chronological replay measures the selected point against the frozen baseline. Evidence: §1–§4. Cost: scheduled Phase-1B work only. Amendment: none.
- **B — 作廢** (plan v2: existing-photos-only; no supplementary probes).
- **C — 不推薦（缺 S1 重跑＋下載雙門）**: any switch lacks S1 license/provenance re-pass + weight-download dual-gate (S1 clear + operator decision), plus new template generation + full bake-off rerun + mobile-checker re-proof. Evidence: §4. Amendment flags: plan candidate table, ModelManifest generation, Phase-1B replay baseline — all require operator decision first.
- **D — 不改**: frozen 0.90 detector gate + display anchors unchanged; no threshold fitting to the 13 probes (calibrating-to-data prohibited).
- **E — 駁回**: reject lowering any gate/threshold to fit probe outcomes; 5 refused screenshots stay refused; margin 0.0687 stays `review`.
- **F — 納 1B**: open robustness questions (occluded/angled observations) enter through the confirmation-gated shadow-candidate flow (§9 design already provides bounded multi-template accumulation); capture-condition guidance (sunglasses-off retry) needs no spec change.

## 7. Honest gaps (pending, not concealed)

1. Pair 2 (2026may + int8bq) comparison **unrun** — no weights on disk.
2. Rename-after rerun (manifest-path confirmation) **pending weights**.
3. Per-probe `predicted` for 7 usable probes **pending** (§1.2) — only probe-11 measured.
4. Margin/confusion denominators (N<30) **undecidable** — no rates claimed.
5. LFW literature figures **cited_pending_primary** (§2).
6. Rerun/forensics numbers **carried over** from repo-outside artifacts, not reproduced here.

## 8. Exclusions and deviations

- No weight downloaded; no product code touched (docs-only diff).
- Frozen 0.90 gate untouched; no tuning proposed anywhere.
- Photos/manifest/embeddings/reports not in Git (this note cites basenames + aggregates only).
- 幼兒園相簿不讀不碰；Phase-1B mechanics untouched.
