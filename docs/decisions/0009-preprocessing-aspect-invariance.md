# 0009. Preprocessing preserves source aspect ratio

- Status: Accepted
- Date: 2026-09-15
- Amends: 0004 (ONNX-first portability) — adds an explicit invariant to the portable contract
- Relates: 0008 (Mac live identification research) — phase boundary amended, see below
- Implementation status: governance trigger (`contract_guard`, runtime manifest derivation) lands after PR #58; until then the 1→2 bump was applied by manual gallery rebuild only.

## Context

The Phase 2A live smoke ran end to end on real hardware but every identification was wrong: top candidates were not the participant, all results fell into the review band, and margins were 0.017–0.041. The failure signature was not a confident wrong match but a collapse — every gallery score compressed into a narrow low band.

Two spikes located the cause in preprocessing, not in the model, the thresholds, or the gallery:

- The detector resized any input onto a fixed 640×640 network input without preserving aspect ratio, then restored box and landmark coordinates through independent per-axis factors.
- The aligned crop was taken from that restored box and resized anisotropically to 112×112, compounding the distortion a second time.
- The enrollment corpus is square (22 of 23 images), so the gallery side passed through the same path with a distortion factor of exactly 1.0 — while a 720×1280 live frame carried a factor of 1.7778.

Measured effect, using the real scoring chain and controls:

| source | distortion | result |
|---|---|---|
| square enrollment, self-comparison | 1.000 | top-1 1.0000, margin 0.4482 |
| two 1440×1920 probes, as captured | 1.333 | top-1 0.3399 (margin 0.0151) and 0.4577 (margin 0.0639) |
| the same two probes, letterboxed | 1.000 | top-1 0.3988 and 0.6435; the second probe's margin rose from 0.0639 to 0.0753 |
| square image warped to 720×1280 | 1.778 | top-1 fell 1.0000 → 0.3178, margin 0.0219 |

Controls separated the cause: black bars alone cost nothing (0.9682 / 0.4496), pure warping alone collapsed the margin to 0.0004, and the same image under two preprocessings produced embeddings 0.42 apart in cosine distance. A square-cropped probe still scored 0.5656 with a wrong top-1, confirming that single-photo gallery coverage is a real but much smaller residual.

Every versioned stage of the portable contract was satisfied throughout. Versioning a stage does not make it correct.

## Decision

1. Preprocessing preserves source aspect ratio end to end. Fixed-input resize uses one uniform scale plus padding; coordinates are restored through that same scale and padding offset. Per-axis restoration is prohibited.
2. The aligned crop is produced by a similarity transform fitted to the five detected landmarks onto a canonical 112×112 template, replacing the raw-box crop. This restores compliance with the architecture already specified in the design document; it is a correction, not a new design.
3. `ALIGN_CONTRACT_VERSION` is bumped once for each of these corrections — 1 → 2 for the resize change, 2 → 3 for the alignment change. Per the existing governance clause, each bump creates a new template generation: retained exemplars are re-embedded, and identities whose stored geometry and margin cannot reproduce the new alignment become `re_enrollment_required`. Nothing is silently re-embedded under mismatched geometry.
4. Layer-A conformance gains non-square fixtures asserting geometric invariance. They stay face-free.
5. Evaluation artifacts measured under contract version 1 are labeled as such. They are not silently carried forward as descriptions of the shipped pipeline.

## Consequences

The research gallery is rebuilt under the new generation. Production carries no enrolled identities at this time, so the re-enrollment cost is currently zero — which is precisely why the change lands now rather than after real enrollment begins.

Selection evidence gathered under contract version 1 (the bake-off matrices and the real-image replay) was measured with a square gallery against a square non-target set — distortion 1.0 on both sides — while its probe set carried distortion 1.333. Whether those results need re-interpretation or re-measurement is a separate decision, deliberately not taken here.

Phase 2A is redefined as tooling correctness and preprocessing invariance plus the acceptance items the team can complete unaided, with no accuracy target. Research validity — comparison arms, session-level denominators, holdout split, and the unenrolled-participant session — becomes Phase 2B. The phase boundary amendment is recorded in the research design and the implementation plan; whether it also warrants its own ADR is left to the team lead.
