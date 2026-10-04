"""Isolated research CLI: live / replay / delete (Phase 2A §4 & §6 T7 + A).

Source of truth:
    - Phase 2A Implementation Plan §4 & §6 T7;
    - Task: t-20260914111211897569-76424-38 (T7);
    - Task: t-20260914144956431776-76424-50 (A: true camera wiring);
    - Governing decisions: d-20260914110757304910-5, d-20260914144615650283-7.

Hard boundaries:
    - This module never touches the production facecore CLI commands; it
      is a separate isolated entry point (``python -m facecore.research.cli``).
    - Store guard: the research store must resolve outside the repository
      and must not escape through symlinks (fail-closed StorePathError).
    - live requires explicit --record-consent AND --image-consent flags
      (active opt-in mapped from the UI checkboxes); missing either
      refuses to start and writes nothing committable.
    - ``--device fake`` runs the deterministic synthetic pump
      (controller → fake capture → real engine → real recorder) for CI
      and camera-free smoke. ``--device <id>`` opens the production
      OpenCV adapter and scores every frame through the true T2 pipeline
      (YuNet + SFace + frozen external gallery); needs a real camera and
      external --models/--corpus, not covered by CI.
    - replay/delete reuse the sealed T6/T5 paths; labels never enter
      replay. Exit codes: 0 ok, 2 usage/config, 4 store/key failure.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

import numpy as np

from facecore.contracts.crypto import StoreCorruptionError
from facecore.live.capture import CaptureSource, FakeCapture, OpenCVCapture
from facecore.live.contracts import (
    FrameObservation,
    FramePacket,
    ResearchProfile,
)
from facecore.live.desktop import DesktopSession
from facecore.live.session import SessionEngine
from facecore.research.analysis import analyze_batch, save_case_summaries
from facecore.research.diagnostics import SessionTrace
from facecore.research.experiment import EvaluationLabel
from facecore.research.records import ConsentRecord, FrameScore
from facecore.research.recorder import ResearchRecorder
from facecore.research.replay import (
    ArmOutcome,
    ReplayRefusal,
    evaluate_arms,
    replay_session,
)
from facecore.research.split import ContaminationRecord

# Frozen pair-1 model selection (matches production bakeoff wiring):
# YuNet 2023mar fixed-640 detector + SFace 2021dec fp32 embedder.
# License/provenance are NOT decided here: YuNet is MIT (2020 Shiqi Yu)
# and SFace is Apache-2.0 — recorded in
# docs/research/2026-09-10-model-candidate-gate.md (§1 items 2-4) and
# docs/plans/2026-09-10-phase-1a-implementation-plan.md (§Evidence).
# Both keep PROVENANCE_UNRESOLVED (SFace: upstream issue #313).
TRUE_PIPELINE_GENERATION = "gen-1"
TRUE_DETECTOR_FILENAME = "face_detection_yunet_2023mar.onnx"
TRUE_DETECTOR_SHA256 = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)
TRUE_EMBEDDER_FILENAME = "face_recognition_sface_2021dec.onnx"


class StorePathError(ValueError):
    """Research store path escapes the approved root (fail-closed)."""


class ModelSetupError(ValueError):
    """True model artifacts failed to load (G3 W7: told apart from gallery)."""


def resolve_store(
    store: Path,
    *,
    repo_root: Path | None = None,
    approved_root: Path | None = None,
) -> Path:
    """Resolve and guard the research store path.

    Refuses paths inside the repository and symlink escapes outside the
    approved root.
    """
    root = (
        repo_root if repo_root is not None else Path(__file__).resolve().parents[3]
    ).resolve()
    # Resolve fully (symlinks included) for the escape check.
    resolved = store.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        raise StorePathError(
            f"research store {resolved} must not live inside the repo {root}"
        )
    if approved_root is not None:
        approved = approved_root.resolve()
        try:
            resolved.relative_to(approved)
        except ValueError as exc:
            raise StorePathError(
                f"research store {resolved} escapes approved root {approved}"
            ) from exc
    return resolved


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _fake_scorer(
    model_generation: str, gallery_digest: str
) -> Callable[[FramePacket], FrameObservation]:
    def _score(packet: FramePacket) -> FrameObservation:
        # Deterministic synthetic observation: single-face quality pass
        # with fixed mid scores (no ground truth, no learning).
        return FrameObservation(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            processed_ns=packet.captured_ns + 1_000_000,
            quality_pass=True,
            quality_reasons=(),
            face_count=1,
            face_box=(4.0, 4.0, 8.0, 8.0),
            identity_scores={"person-01": 0.50, "person-02": 0.30},
            quality_rank=0.5,
            model_generation=model_generation,
            gallery_digest=gallery_digest,
        )

    return _score


class PresenceDetectedError(ValueError):
    """A face was detected in no-participant checkpoint mode (fail-closed)."""


def presence_scorer(
    detector: Any,
    model_generation: str,
    gallery_digest: str,
    *,
    presence_mode: str = "collection",
) -> Callable[[FramePacket], FrameObservation]:
    """Hybrid scorer: true detector presence + synthetic identity.

    face_count/face_box/quality_pass/quality_reasons come from running the
    true detector on the true pixels; identity_scores stay synthetic fixed
    values (identity NEVER goes true for convenience).

    presence_mode splits the stop semantics (never a global rule):
    - "checkpoint": any detected face (>= 1) raises PresenceDetectedError
      (fail-closed abort; the pump layer turns it into exit 4 + stderr +
      an uncommitted bundle — the exception alone is NOT the surfacing
      mechanism, because the controller can only re-raise it when a
      frame_transform is present).
    - "collection" (default): no presence stop; faced frames score
      normally so future collection callers never inherit checkpoint
      semantics by omission. The default is deliberately the non-stopping
      side: checkpoint callers must opt in explicitly.
    """
    if presence_mode not in ("checkpoint", "collection"):
        raise ValueError(f"unknown presence_mode {presence_mode!r}")

    def _score(packet: FramePacket) -> FrameObservation:
        from facecore.pipeline.decode import DecodedImage  # noqa: PLC0415
        from facecore.pipeline.detect import enforce_single_face  # noqa: PLC0415

        height, width, _ = packet.rgb.shape
        decoded = DecodedImage(
            width=width,
            height=height,
            color_order="RGB",
            pixels=packet.rgb.tobytes(),
        )
        detected = detector.detect(decoded)
        status, reason, face = enforce_single_face(detected)
        face_count = len(detected)
        if presence_mode == "checkpoint" and face_count >= 1:
            raise PresenceDetectedError(
                f"presence_face_detected: {face_count} face(s) in "
                f"no-participant checkpoint frame {packet.sequence}"
            )
        if status == "ok" and face is not None:
            quality_pass = True
            quality_reasons: tuple[str, ...] = ()
            face_box = face.box
        else:
            quality_pass = False
            quality_reasons = (reason,) if reason else ()
            face_box = None
        return FrameObservation(
            sequence=packet.sequence,
            captured_ns=packet.captured_ns,
            processed_ns=packet.captured_ns + 1_000_000,
            quality_pass=quality_pass,
            quality_reasons=quality_reasons,
            face_count=face_count,
            face_box=face_box,
            identity_scores={"person-01": 0.50, "person-02": 0.30},
            quality_rank=0.5,
            model_generation=model_generation,
            gallery_digest=gallery_digest,
        )

    return _score


def _load_profile(profile_path: Path) -> ResearchProfile:
    try:
        payload = json.loads(profile_path.read_text())
    except (ValueError, OSError) as exc:
        raise ValueError(f"profile unreadable: {profile_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"profile must be a JSON object: {profile_path}")
    return ResearchProfile.from_dict(payload)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def frame_score_of(observation: FrameObservation) -> FrameScore:
    """Reduce one scored observation to its best-match ledger entry."""
    scores = observation.identity_scores
    if not scores:
        return FrameScore(
            sequence=observation.sequence,
            top_identity=None,
            top_score=None,
            margin=None,
        )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_identity, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else None
    return FrameScore(
        sequence=observation.sequence,
        top_identity=top_identity,
        top_score=float(top_score),
        margin=float(top_score - runner_up) if runner_up is not None else None,
    )


# G3 W4: image + record retention for the local test app (spec §3:
# G3 images and records are kept 30 days so the operator can analyze
# after testing). Replaces the hardcoded 30d/7d TTLs below.
G3_RETENTION_DAYS = 30

G3_RESULTS_CSV_COLUMNS = (
    "round_id",
    "session_id",
    "started_utc",
    "result",
    "shown_identity",
    "top1_identity",
    "top1_score",
    "top2_identity",
    "top2_score",
    "margin",
    "elapsed_ms",
    "frames_sampled",
    "label_kind",
    "label_identity",
    "profile_version",
    "model_generation",
    "gallery_digest",
)


def g3_round_best_scores(
    observations: tuple[FrameObservation, ...],
) -> tuple[
    tuple[str | None, float | None], tuple[str | None, float | None], float | None
]:
    """Best-frame top1/top2/margin for one G3 round (csv display only)."""
    ranked_frames = sorted(
        (o for o in observations if o.identity_scores),
        key=lambda o: (o.quality_rank, -o.sequence),
        reverse=True,
    )
    if not ranked_frames:
        return (None, None), (None, None), None
    ordered = sorted(
        ranked_frames[0].identity_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top1 = (ordered[0][0], float(ordered[0][1]))
    top2: tuple[str | None, float | None] = (None, None)
    if len(ordered) > 1:
        top2 = (ordered[1][0], float(ordered[1][1]))
    margin = (
        float(top1[1] - top2[1])
        if top1[1] is not None and top2[1] is not None
        else None
    )
    return top1, top2, margin


def g3_round_row(round_: Any) -> dict[str, object]:
    """Reduce one labeled round to its results.csv row (spec §3 fields)."""
    from facecore.live.qt_window import RoundComplete as _RC

    assert isinstance(round_, _RC), f"expected RoundComplete, got {type(round_)}"
    terminal = round_.terminal
    (top1_ident, top1_score), (top2_ident, top2_score), margin = g3_round_best_scores(
        round_.observations
    )
    shown = terminal.matched_identity
    return {
        "round_id": round_.attempt_id or round_.session_id,
        "session_id": round_.session_id,
        "started_utc": round_.started_utc,
        "result": terminal.status.value,
        "shown_identity": shown or "",
        "top1_identity": top1_ident or "",
        "top1_score": "" if top1_score is None else f"{top1_score:.4f}",
        "top2_identity": top2_ident or "",
        "top2_score": "" if top2_score is None else f"{top2_score:.4f}",
        "margin": "" if margin is None else f"{margin:.4f}",
        "elapsed_ms": f"{terminal.elapsed_ms:.1f}",
        "frames_sampled": str(terminal.frames_sampled),
        "label_kind": round_.label_kind,
        "label_identity": round_.label_identity or "",
        "profile_version": round_.profile_version,
        "model_generation": terminal.model_generation,
        "gallery_digest": terminal.gallery_digest,
    }


def append_g3_results_csv(results_csv: Path, round_: Any) -> None:
    """Append one G3 round row (images/embeddings never enter the csv)."""
    import csv as _csv

    row = g3_round_row(round_)
    write_header = not results_csv.is_file()
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "a", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=G3_RESULTS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in G3_RESULTS_CSV_COLUMNS})


# D3b (decision d-20260929174858816709-11 item 2): the non-recording mode's
# plaintext demo result file. It lives in the same store directory as
# results.csv but is a DIFFERENT file with its OWN header, so the frozen
# 17-column research ledger and its 29 existing rounds are never touched.
#
# Plaintext, therefore privacy-relevant: every column below is a scalar
# summary already present in results.csv or the session envelope. No frame
# pixels, no embeddings, no gallery photos, no face boxes.
G3_DEMO_RESULTS_CSV_NAME = "demo-results.csv"
# D7-A #141: the demo log is named per App EXECUTION, not per session and
# not per round — see `demo_results_csv_path`. `G3_DEMO_RESULTS_CSV_NAME`
# stays as the un-suffixed legacy name: it is what existing files are
# called, what the close-out envelope reports for them, and what the
# header guard refuses to append to.
G3_DEMO_RESULTS_CSV_STEM = "demo-results"

G3_DEMO_RESULTS_CSV_COLUMNS = (
    # mode + provenance: a demo row is distinguishable from research evidence
    "mode",
    "app_version",
    "profile_version",
    "model_generation",
    "gallery_digest",
    "profile_digest",
    # round identity and timing
    "round_id",
    "session_id",
    "started_utc",
    "labeled_at_utc",
    "elapsed_ms",
    # D1 reason codes: the whole point of the D1 split is that a thin-evidence
    # round is not the same event as "not enrolled", and this file is where
    # that distinction survives without decryption.
    "result",
    "reason_codes",
    # top1/top2 similarity and margin, formatted exactly like results.csv
    "top1_identity",
    "top1_score",
    "top2_identity",
    "top2_score",
    "margin",
    # effective / required frames
    "frames_sampled",
    "frames_usable",
    "required_support",
    # operator verdict (label_terminal semantics, unchanged)
    "label_kind",
    "label_identity",
    # D7-A W3. Everything below is data a live round ALREADY computes; none
    # of it is a new measurement. See plan v8 §3.3 and §4.
    #
    # Time actually spent gathering recognition evidence. This is the
    # `TimingMarks.recognition_duration_ms` property (contracts.py:302),
    # NOT `open_duration_ms` / `open_to_first_frame_ms`: controller.py:235
    # deliberately never fills the open segment on the demo path, so those
    # two would be permanently blank here — plan v8 §8.
    "recognition_duration_ms",
    # Frames the quality gate rejected (session.py:293/:317). With
    # frames_sampled/frames_usable this gives the usable rate directly.
    # `frames_dropped` is deliberately NOT recorded: no `+= 1` exists, so
    # it would be a column that is always 0 — plan v8 §8.
    "frames_rejected",
    # ⚠️ SCOPE: this column answers 「why the support window was cleared」,
    #    which is NOT all of D4 §11 item 18b. A round with no usable
    #    frame at all is the extreme case this column cannot explain: the
    #    support window never opened, so nothing was ever cleared and
    #    there is no clear reason to record. 18b's real answer for those
    #    is in the gap between `frames_sampled` and `frames_usable`,
    #    which needs a per-gate breakdown of what the quality gate
    #    rejected — that is W2's scope, not this column.
    #    ⚠️ NO PER-RUN NUMBERS HERE, deliberately. An earlier version
    #    cited per-round counts from one operator batch, along with a
    #    total frame count. That batch was deleted and the file refilled,
    #    so those figures described data that no longer exists — and any
    #    replacement would go stale the next time the operator runs the
    #    App, because the log is appended per execution. The reasoning
    #    above holds for every batch; counts never did. For what the
    #    operator has accumulated, read the runbook section named below;
    #    it is maintained against the live log.
    #    `docs/w0a-diagnostic-run-runbook.md` §「你目前累積了多少資料」
    # D4 §11 item 18b: why the support window was emptied, and how often.
    # `reason:count` pairs joined by `;`, empty when the window was never
    # disturbed. `reset_reason` is the key rather than `event_type`
    # because four sites share `event_type="rejected"` while their reasons
    # split into 「restart the App」 and 「keep sampling」 — the distinction
    # 18b is after does not exist on `event_type`. Counted only when the
    # window actually shrank, so this column never reports a clear that
    # did not happen.
    "support_clear_reasons",
    # This frame was NOT accumulated because it arrived inside
    # `min_support_interval_ms` of the previous support frame.
    # ⚠️ It is NOT a clearing event: the support window is left exactly as
    # it was (`support_before == support_after`). W4 must not add this to
    # `support_clear_reasons` or present the two as competing answers to
    # 「support 為何沒到 3」 — conflating them is the 18b error itself.
    "score_reset_count",
    "interval_skip_count",
    # ⚠️ HOW THE THREE MAY BE COMBINED — this lives here because a reader
    #    of the CSV cannot see any of it. W4's summary contract has to
    #    state it outright:
    #      · `interval_skip_count` and `support_clear_reasons` must NOT
    #        be added together — a skipped frame left the window intact.
    #      · `score_reset_count` and `support_clear_reasons` must NOT be
    #        added together, and neither is a subset of the other in
    #        EITHER direction — the two directions fail in OPPOSITE ways,
    #        so neither counter alone can stand in for the reasons:
    #        ① THE COUNTER OVER-COUNTS. Some resets clear nothing: the
    #           window was already empty, so `support_after` equals
    #           `support_before` and no clear reason is recorded. Summing
    #           it in double-counts a single clear, or invents one.
    #        ② THE COUNTER IS BLIND TO MOST CLEARS. It reads
    #           `event_counts()["score_reset"]`, and exactly ONE site emits
    #           that event — session.py:534, the score/margin gate — so it
    #           moves for `score_below_threshold` and
    #           `margin_below_threshold` ONLY. Every other clearing path
    #           empties the window without moving the counter:
    #           `continuity_jump_detected`, `quality_rejected: …`,
    #           `no_face_detected`, `empty_identity_scores`,
    #           `none_runner_up`, `input_multiple_faces`. Summing it in
    #           then invents clears that never happened.
    #        Measured against `profiles/g3-v1.json`, not assumed: driving
    #        all eight paths leaves six of them invisible to the counter.
    #        Report `support_clear_reasons`, with `reason_codes`'
    #        `support_N_of_3` alongside it — the counter alone answers
    #        neither direction.
    # W0 ground truth. Empty means "not recorded", never a guess: an
    # invented 'target' would silently corrupt the cross-identity counts.
    # ⚠️ CONDITIONALLY RETAINED (D7-A W3 rework 2, R4). These are
    # RECEIVING fields, not measuring ones: the empty string honestly
    # reports that the round has no ground truth, which is the operator's
    # actual state today — that is information, not noise. (Contrast
    # `score_p50`, deleted because it measured a biased subset and invited
    # a wrong inference.) No input path exists — nothing in the CLI
    # writes them — so as the code stands they are ALWAYS empty, and no
    # amount of rerunning rounds changes that. What would change it is a
    # decision, not a missing edit: whether the W0-b run (cross-identity
    # plus non-target rotation, which is operator's open question and is
    # NOT scheduled) needs an input path added. If W0-b's scheduling
    # finds one is needed, adding it is that run's work item, not this
    # file's. Do not read the empty cells as "target" or as a guess.
    # REVIEW CONDITION (reviewers' wording, not a deadline): if W0-b is
    # not scheduled, delete these columns. W0-b's scheduling is the
    # operator's open decision — it needs the operator present to drive
    # a real device, which no implementation task can authorise. No time
    # limit is stated here on purpose: a deadline would put the columns'
    # existence on a timer this codebase has no authority to set.
    "probe_kind",
    "presenting_identity",
    # Threshold snapshot. Plan v8 §4 admits only parameters proven to be
    # read on the live control flow. `sample_interval_ms` and `max_frames`
    # are excluded on purpose — the first is a false knob (hardcoded
    # 200 ms at controller.py:57), the second is unreachable in the
    # launcher's --continuous path.
    "match_threshold",
    "review_threshold",
    "margin_threshold",
    "min_support_interval_ms",
    "timeout_ms",
    # D7-A W1: gallery startup visibility, already computed by
    # `GalleryLoadReport` and already shown in the Qt UI — this only
    # writes the same three facts into every round of the demo log, so a
    # CSV reader no longer has to trust the UI for them.
    #
    # APPENDED, never inserted: the existing 23 columns keep their
    # positions because the operator's spreadsheet formulas depend on
    # `margin` (col 18) and `label_kind` (col 22).
    #
    # `gallery_rejected` is "filename（reason）" per failure joined by ";"
    # — filename AND reason, not counts, because a count alone reproduces
    # the original problem: you cannot tell WHICH photos were dropped.
    # Empty string means every photo loaded.
    "expected_count",
    "loaded_count",
    "gallery_rejected",
)


class DemoLogHeaderMismatch(OSError):
    """#141: the demo log's header is not this build's header.

    An `OSError` subclass on purpose, not a new failure kind: both window
    write paths already wrap `append_g3_demo_results_csv` in
    `except OSError`, which sets 「紀錄寫入失敗」 and drops the round from
    `completed_rounds`. A refusal must land on that same fail-closed path —
    a round we refuse to record must not count as a recorded round.

    The guard exists because the header is written once, on first write
    (`write_header = not demo_csv.is_file()`), so a file created by an
    older build keeps its old header forever and every later row lands in
    the wrong cells. `demo_results_csv_path` removes that situation by
    construction (a new App run opens a new file); this is the backstop
    for when it does not happen.
    """


def _second_precision_stamp(started_utc: str) -> str:
    """An ISO instant as `YYYY-MM-DDTHHMMSS_±HHMM`, dropped to seconds.

    Filesystem-safe because an ISO timestamp contains `:`.

    Parsed, then reformatted, rather than sliced: `2026-10-03T15:30:00+00:00`
    and `2026-10-03T15:30:00.123456+00:00` describe the same second and must
    produce the same name — otherwise a caller that happens to omit the
    microseconds gets a filename the operator has never seen.

    A `Z` suffix is accepted because that is how UTC instants are commonly
    written, and `Z` is legal in a filename but reads as noise next to the
    rest of the stamp.

    Unparseable input is sanitized rather than raised on: a filename is
    not worth crashing a camera run over, and the microseconds-free
    fallback still gives the operator a per-session file.
    """
    text = started_utc.strip()
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text.replace(":", "").replace("+", "_").replace(".", "")
    moment = moment.replace(microsecond=0)
    return moment.isoformat().replace(":", "").replace("+", "_")


def demo_results_csv_path(store_root: Path, started_utc: str) -> Path:
    """The demo log path for ONE App execution (#141 丙-新).

    Named for when the App started, so a new App run opens a new file and
    never appends to an older one. That is the decision's point: the
    operator's version and the file's header move together, so an existing
    log never needs its header upgraded in place.

    `started_utc` is the App's start time, NOT `session_id` — in this
    codebase `session_id` is per-ROUND (`round_session_id` in `cmd_live`),
    and naming files after it would split one App run into dozens of files.

    Second precision, and it is PARSED rather than sliced. The earlier
    version stripped `:`/`+`/`.` out of the string, which left the
    microseconds in the name: `datetime.now().isoformat()` almost always
    carries six of them, so a real App produced
    `demo-results-2026-10-03T090533492749_0000.csv` while every example in
    the operator's runbook showed six digits. A reader who compared his
    Finder window against the manual had no way to reconcile the two.

    Slicing the string would have been the wrong fix twice over: a
    caller passing second precision has no decimal point to cut at, so
    the cut lands somewhere else entirely, and the timezone offset would
    be chopped along with it. Parsing makes both inputs land on the same
    name, which is the property that actually matters here.
    """
    stamp = _second_precision_stamp(started_utc)
    return store_root / f"{G3_DEMO_RESULTS_CSV_STEM}-{stamp}.csv"


def _demo_log_header_mismatch_error(
    demo_csv: Path, actual: tuple[str, ...]
) -> DemoLogHeaderMismatch:
    """Build the refusal the operator can act on without asking anyone.

    The runbook's step 0 tells the operator to identify their log by the
    LAST column's heading, not by counting columns — so the message names
    the last column of both headers and both counts. It also names the
    per-App file, because that is the actual next step: the old log is not
    broken, it is from another App run.
    """
    expected = G3_DEMO_RESULTS_CSV_COLUMNS
    return DemoLogHeaderMismatch(
        f"demo log 的欄位標題與這版 App 不符，已拒絕寫入（零寫入）：{demo_csv}\n"
        f"  你的檔案：{len(actual)} 欄，最後一欄是 "
        f"{actual[-1] if actual else '(空檔)'}。\n"
        f"  這版 App：{len(expected)} 欄，最後一欄是 {expected[-1]}。\n"
        f"  這是舊一次 App 執行留下的檔案（每個欄位標題都不同時，"
        f"照舊標題讀會讀錯欄）。\n"
        f"  下一步：不用改這個檔案 —— 下次啟動 App 會自動開新檔 "
        f"{G3_DEMO_RESULTS_CSV_STEM}-<啟動時間>.csv，用那個新檔即可。"
    )


def _read_demo_log_header(demo_csv: Path) -> tuple[str, ...] | None:
    """The file's ACTUAL first row, or None when there is no file yet.

    Read from disk on every append rather than remembered: the whole point
    is to catch a header written by a different build, which by definition
    this process never saw being written.
    """
    if not demo_csv.is_file():
        return None
    import csv as _csv

    with demo_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in _csv.reader(handle):
            return tuple(row)
    return None


def _app_version() -> str:
    """Installed package version, recorded in every demo row."""
    try:
        from importlib.metadata import version as _pkg_version  # noqa: PLC0415

        return _pkg_version("facecore")
    except Exception:
        return "unknown"


def g3_demo_round_row(
    round_: Any,
    *,
    required_support: int,
    labeled_at_utc: str,
    app_version: str | None = None,
    event_counts: Mapping[str, int] | None = None,
    support_clears: Mapping[str, int] | None = None,
    profile: ResearchProfile | None = None,
    probe_kind: str = "",
    presenting_identity: str = "",
    gallery_load_report: Any | None = None,
) -> dict[str, object]:
    """Reduce one labeled round to its demo row (D3b decision -11 item 2).

    Reuses g3_round_row for the shared top1/top2/margin reduction so the
    demo file cannot drift from the research ledger's semantics. Only the
    D1 reason codes, the frame counts, and the mode/version provenance are
    added here.

    D7-A W3 adds the fields a live round already computed but never wrote
    out. Three of them are optional arguments on purpose:

    - `event_counts` — the caller owns the engine that produced the round,
      so it is the only place the per-round event tally can be read. When
      omitted the counts are 0, which is a true reading for a round that
      emitted nothing; the column is always present either way, so a
      reader never has to guess whether 0 means "none happened" or
      "nobody looked".
    - `profile` — supplies the §4 threshold snapshot. Recorded only for
      parameters proven to be read on the live control flow.
    - `support_clears` — the per-`reset_reason` buckets for rounds whose
      support window actually shrank. This, not `event_counts`, is the read
      that answers D4 §11 18b; see `SessionEngine.support_clear_reasons`.
      Serialised as `reason:count` joined by `;`, and empty when the window
      was never disturbed.
    - `probe_kind` / `presenting_identity` — the W0 runbook's ground
      truth. Default to empty, never to a guess.
    - `gallery_load_report` — D7-A W1. The startup `GalleryLoadReport`
      the window already holds, forwarded so the row records which
      gallery the round ran against. Absent (None) renders the three
      columns empty rather than zero: zero would claim the gallery held
      zero people, which is a measurement, while empty says no report
      reached this row.
    """
    from facecore.live.qt_window import RoundComplete as _RC

    assert isinstance(round_, _RC), f"expected RoundComplete, got {type(round_)}"
    terminal = round_.terminal
    base = g3_round_row(round_)
    counts = event_counts or {}
    clears = support_clears or {}
    # D7-A W3 rework 4 (B): aggregate by FAMILY before serialising.
    # Truncating each item inside the comprehension collided but never
    # merged, so a round that was refused for two different quality
    # reasons produced two `quality_rejected:1` cells instead of one
    # `quality_rejected:2` — the F1 long tail returning under a new name.
    # Sums are preserved either way; the presentation was wrong.
    clears_by_family: Counter[str] = Counter()
    for _reason, _n in clears.items():
        clears_by_family[_reason.split(":", 1)[0]] += _n

    # `recognition_duration_ms` is a derived property, so it is None when
    # the round never anchored. Render None as empty rather than 0: a zero
    # would claim "the round took no time to recognise", which is a
    # measurement; empty says the marks were never set.
    marks = terminal.timing_marks
    recognition_ms = (
        marks.recognition_duration_ms if marks is not None else None
    )

    def _thr(name: str) -> str:
        return "" if profile is None else str(getattr(profile, name))

    # D7-A W1: gallery visibility. Read defensively rather than asserting
    # the exact type: the window holds whatever the gallery carried, and a
    # caller with a report-shaped object should get a row, not a crash
    # during CSV writing (the window turns any write error into 「紀錄
    # 寫入失敗」, which would hide a healthy round).
    if gallery_load_report is None:
        expected_str = loaded_str = rejected_str = ""
    else:
        expected_str = str(getattr(gallery_load_report, "expected_count", ""))
        loaded_str = str(getattr(gallery_load_report, "loaded_count", ""))
        # filename AND reason, semicolon-joined. Empty means every photo
        # loaded — NOT "0" and NOT "none", which would read as a value.
        rejected_str = ";".join(
            f"{getattr(f, 'filename', '')}（{getattr(f, 'reason', '')}）"
            for f in getattr(gallery_load_report, "failures", ())
        )

    return {
        "mode": "demo-no-recording",
        "app_version": app_version or _app_version(),
        "profile_version": base["profile_version"],
        "model_generation": base["model_generation"],
        "gallery_digest": base["gallery_digest"],
        "profile_digest": terminal.profile_digest,
        "round_id": base["round_id"],
        "session_id": base["session_id"],
        "started_utc": base["started_utc"],
        "labeled_at_utc": labeled_at_utc,
        "elapsed_ms": base["elapsed_ms"],
        "result": base["result"],
        "reason_codes": "|".join(terminal.reason_codes),
        "top1_identity": base["top1_identity"],
        "top1_score": base["top1_score"],
        "top2_identity": base["top2_identity"],
        "top2_score": base["top2_score"],
        "margin": base["margin"],
        "frames_sampled": base["frames_sampled"],
        "frames_usable": str(terminal.frames_usable),
        "label_kind": base["label_kind"],
        "label_identity": base["label_identity"],
        "recognition_duration_ms": (
            "" if recognition_ms is None else f"{recognition_ms}"
        ),
        "frames_rejected": str(terminal.frames_rejected),
        # `reason:count` pairs joined by `;`.
        #
        # D7-A W3 rework 3 (F1): `quality_rejected` is truncated to its
        # prefix. The seven quality gates at `pipeline/quality.py:36-55`
        # each append independently, so the full reset_reason has an
        # unbounded 2^7 key space — 127 non-empty subsets — and a W4
        # `groupby(reason)` would produce a near-singleton tail. The
        # question the log has to answer is 「which FAMILY of cause」,
        # and 「was the quality gate the blocker」 is boolean, not a
        # 7-way combination. The per-gate codes are not lost: they stay
        # in `SessionEngine.support_clear_reasons()` and the trace
        # channel. They are NOT in this CSV — `reason_codes` carries the
        # family too (session.py:414 stores only 「quality_rejected」 in
        # `_rejection_reasons`), so the per-gate detail is unreachable
        # from the demo log alone and must be read in-process or from a
        # recording.
        #
        # Aggregate first (above), then serialise — never truncate inside
        # the comprehension, which is what produced duplicate keys.
        #
        # Split on the FIRST colon: that is the family boundary. Today
        # exactly one producer (`q_reason` at `session.py:407`) yields a
        # reason containing a colon, and that is a coincidence of the
        # current call sites, NOT a contract: a future reason carrying a
        # colon outside this family would have its tail dropped with
        # nothing in the CSV to show it. The key is documented as a
        # family prefix precisely so that shows up as a decision.
        "support_clear_reasons": ";".join(
            f"{family}:{n}" for family, n in sorted(clears_by_family.items())
        ),
        "score_reset_count": str(counts.get("score_reset", 0)),
        "interval_skip_count": str(counts.get("interval_skip", 0)),
        "probe_kind": probe_kind,
        "presenting_identity": presenting_identity,
        "match_threshold": _thr("match_threshold"),
        "review_threshold": _thr("review_threshold"),
        "margin_threshold": _thr("margin_threshold"),
        "required_support": str(required_support),
        "min_support_interval_ms": _thr("min_support_interval_ms"),
        "timeout_ms": _thr("timeout_ms"),
        "expected_count": expected_str,
        "loaded_count": loaded_str,
        "gallery_rejected": rejected_str,
    }


def append_g3_demo_results_csv(
    demo_csv: Path,
    round_: Any,
    *,
    required_support: int,
    labeled_at_utc: str,
    event_counts: Mapping[str, int] | None = None,
    support_clears: Mapping[str, int] | None = None,
    profile: ResearchProfile | None = None,
    probe_kind: str = "",
    presenting_identity: str = "",
    gallery_load_report: Any | None = None,
) -> None:
    """Append one non-recording round to the plaintext demo result file.

    The header is written once, exactly like append_g3_results_csv, and the
    row is derived from g3_round_row so the two ledgers agree on scores.

    D7-A W3: the caller forwards the per-round engine tallies and the
    round's profile so the row can carry data the live round already
    computed. See `g3_demo_round_row` for why each is optional.

    D7-A W1: `gallery_load_report` rides along so every row records which
    gallery it ran against. The append semantics are unchanged — writing a
    row still never touches the header.

    D7-A #141: a file whose header is not this build's is REFUSED with
    zero writes. This IS a new branch on the write path, and it is what
    issue #140 asked for: the header used to be written once, on first
    write, so a file created by an older build kept its old header forever
    and every later row landed in the wrong cells. The comparison is the
    whole header tuple, not its length — a renamed or reordered column
    keeps the count and would still be caught.
    """
    import csv as _csv

    actual_header = _read_demo_log_header(demo_csv)
    if actual_header is not None and actual_header != G3_DEMO_RESULTS_CSV_COLUMNS:
        raise _demo_log_header_mismatch_error(demo_csv, actual_header)

    row = g3_demo_round_row(
        round_,
        required_support=required_support,
        labeled_at_utc=labeled_at_utc,
        event_counts=event_counts,
        support_clears=support_clears,
        profile=profile,
        probe_kind=probe_kind,
        presenting_identity=presenting_identity,
        gallery_load_report=gallery_load_report,
    )
    write_header = not demo_csv.is_file()
    demo_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(demo_csv, "a", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=G3_DEMO_RESULTS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in G3_DEMO_RESULTS_CSV_COLUMNS})


class _NullRecorder:
    """D3b demo-mode stand-in for the encrypted ResearchRecorder.

    Decision d-20260929193128013174-12 item 3 forbids constructing a
    ResearchRecorder in the non-recording mode, because ``__init__``
    mkdirs the store root and ``begin_attempt`` opens the attempt ledger.
    This object performs no I/O at all: it opens no bundle, writes no
    encrypted store, and appends no attempt row.

    It exists so the ~20 bookkeeping call sites in ``cmd_live`` (abort,
    finish_attempt, purge-style tails) do not each need their own
    ``if recording`` branch — the DEMO path simply has no ledger to
    update. The biometric channel (``append_frame``) and the spatial
    channel (``record_crop_mapping``) are NOT routed here: those are
    guarded explicitly at their call sites, because "no recorder" must
    never read as "silently stage the frame anyway".

    The no-op set is an explicit ALLOW-LIST, not a catch-all. A blanket
    ``__getattr__`` would answer every ``getattr(recorder, "x", None)``
    capability probe in the codebase — including
    ``controller._append_live_trace``'s
    ``getattr(self._trace_recorder, "append_trace", None)`` — so a
    renamed or misspelled recorder method would resolve to a no-op and
    lose its write without any error. The allow-list makes that failure
    loud instead: an unexpected method raises, exactly as a real
    recorder missing that method would.
    """

    _NO_OPS = frozenset(
        {
            "abort",
            "begin",
            "commit",
            "finish_attempt",
            "begin_attempt",
        }
    )

    def __getattr__(self, name: str) -> Any:
        if name not in self._NO_OPS:
            raise AttributeError(
                f"_NullRecorder has no {name!r}: demo mode must never reach a "
                "recorder channel that writes (this is the guard that keeps "
                "frames, embeddings and the attempt ledger out of the "
                "non-recording path)"
            )

        def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        return _noop


NULL_RECORDER = _NullRecorder()


def _g3_round_op_status(status_value: str) -> str:
    """Map a round terminal status onto the attempt ledger vocabulary."""
    if status_value == "matched":
        return "completed"
    if status_value in (
        "accepted",
        "open_error",
        "setup_error",
        "cancelled",
        "timeout",
        "completed",
        "error",
    ):
        return status_value
    return "completed"


def commit_g3_rounds(
    recorder: ResearchRecorder,
    results_csv: Path,
    rounds: list[Any],
) -> tuple[int, int]:
    """Commit each labeled G3 round: bundle + attempt + csv row.

    Returns (committed, failed). A failed round is aborted and its
    attempt closed as error; other rounds still commit. Raises nothing.
    """
    committed = 0
    failed = 0
    for round_ in rounds:
        terminal = round_.terminal
        try:
            recorder.commit(
                terminal,
                frame_scores=tuple(frame_score_of(o) for o in round_.observations),
            )
        except (KeyError, ValueError) as exc:
            print(f"research live: round commit failed: {exc}", file=sys.stderr)
            try:
                recorder.abort(terminal.session_id, reason="round_commit_failed")
            except Exception:
                pass
            if round_.attempt_id is not None:
                try:
                    recorder.finish_attempt(
                        round_.attempt_id,
                        result=None,
                        operational_status="error",
                        error_code="round_commit_failed",
                    )
                except Exception:
                    pass
            failed += 1
            continue
        if round_.attempt_id is not None:
            try:
                recorder.finish_attempt(
                    round_.attempt_id,
                    result=terminal,
                    operational_status=_g3_round_op_status(terminal.status.value),
                    error_code=None,
                )
            except Exception as exc:
                print(
                    f"research live: round finish_attempt failed: {exc}",
                    file=sys.stderr,
                )
                failed += 1
                continue
        try:
            append_g3_results_csv(results_csv, round_)
        except OSError as exc:
            print(f"research live: results.csv append failed: {exc}", file=sys.stderr)
            failed += 1
            continue
        committed += 1
    return committed, failed


def _build_true_context(
    models: Path,
    corpus: Path | None,
    profile: ResearchProfile,
    *,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
    gallery_dir: Path | None = None,
) -> Any:
    """Build the frozen true scoring context from external artifacts.

    G3 W6: ``gallery_dir`` (enrollment folder, identity = filename stem)
    takes precedence over the ``corpus`` manifest when given.
    """
    from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
        ScoringContext,
        build_gallery_from_folder,
        build_research_gallery,
    )

    if detector_factory is None:
        from facecore.pipeline.yunet import (  # noqa: PLC0415 (device-gated)
            YuNetDetector,
        )

        def _default_detector(models_dir: Path) -> Any:
            return YuNetDetector(
                models_dir / TRUE_DETECTOR_FILENAME,
                TRUE_DETECTOR_SHA256,
            )

        detector_factory = _default_detector
    if embedder_factory is None:
        from facecore.pipeline.embed import (  # noqa: PLC0415 (device-gated)
            Embedder,
        )
        from facecore.contracts.manifest import (  # noqa: PLC0415
            ModelManifest,
        )

        def _default_embedder(models_dir: Path) -> Any:
            manifest = ModelManifest.sface_2021dec_fp32()
            return Embedder(manifest, models_dir / TRUE_EMBEDDER_FILENAME)

        embedder_factory = _default_embedder
    detector: Any
    embedder: Any
    try:
        detector = detector_factory(models)
        embedder = embedder_factory(models)
    except Exception as exc:
        # G3 W7 (W6 minor finding): model failures are told apart from
        # gallery failures so the Chinese startup reason names the
        # right cause.
        raise ModelSetupError(f"模型檔載入失敗：{exc}") from exc
    if gallery_dir is not None:
        gallery = build_gallery_from_folder(
            gallery_dir,
            detector=detector,
            embedder=embedder,
            generation=TRUE_PIPELINE_GENERATION,
        )
    else:
        assert corpus is not None
        gallery = build_research_gallery(
            corpus,
            repo_root=models,
            detector=detector,
            embedder=embedder,
            generation=TRUE_PIPELINE_GENERATION,
        )
    policy = profile_to_policy(profile)
    return ScoringContext(
        gallery=gallery,
        model_version=gallery.model_version,
        policy=policy,
        detector=detector,
        embedder=embedder,
    )


def _build_true_detector_only(
    models: Path,
    *,
    detector_factory: Callable[[Path], Any] | None = None,
) -> Any:
    """Build ONLY the true detector (checkpoint path; no gallery).

    Checkpoint presence scoring uses the detector alone (face_count /
    face_box / quality from true pixels; identity stays synthetic), so
    building an embedder + gallery — which demands faceless-rejecting
    enrollment faces — would force a human-face corpus for a gallery
    the branch never reads (G3 boundary conflict). The YuNet default
    (with SHA pin) and any factory product flow through the same
    isinstance guard at the call site; nothing here relaxes it.
    """
    if detector_factory is None:
        from facecore.pipeline.yunet import (  # noqa: PLC0415 (device-gated)
            YuNetDetector,
        )

        def _default_detector(models_dir: Path) -> Any:
            return YuNetDetector(
                models_dir / TRUE_DETECTOR_FILENAME,
                TRUE_DETECTOR_SHA256,
            )

        detector_factory = _default_detector
    return detector_factory(models)


def profile_to_policy(profile: ResearchProfile) -> Any:
    """Map a research profile onto a frozen-v1 policy with its thresholds."""
    from facecore.contracts.policy import (  # noqa: PLC0415 (device-gated)
        PolicyProfile,
    )

    return PolicyProfile.frozen_v1().with_thresholds(
        match_threshold=profile.match_threshold,
        review_threshold=profile.review_threshold,
        margin_threshold=profile.margin_threshold,
    )


def _default_experiment_manifest(experiment_id: str) -> Any:
    """Build the minimal frozen manifest for a CLI-driven attempt (E3)."""
    from facecore.research.experiment import ExperimentManifest

    return ExperimentManifest.from_dict(
        {
            "identity": {"experiment_id": experiment_id},
            "software": {},
            "gallery": {},
            "policy": {},
            "capture": {},
            "privacy": {},
            "study": {},
            "analysis": {},
        }
    )


def cmd_live(
    *,
    profile_path: Path,
    store: Path,
    key_dir: Path,
    device: str,
    session_id: str,
    record_consent: bool,
    image_consent: bool,
    fixed_seconds: bool = False,
    ui: str = "fake",
    qt_offscreen: bool = False,
    models: Path | None = None,
    corpus: Path | None = None,
    capture_factory: Callable[[str], CaptureSource] | None = None,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
    experiment_id: str = "exp-cli-e3",
    attempt_id: str | None = None,
    presence_mode: str = "collection",
    continuous: bool = False,
    config: Path | None = None,
    gallery_dir: Path | None = None,
    mode: str = "record",
) -> int:
    """Run one bounded research session (fake pump or real camera).

    continuous (G3 W2): when True with --ui qt, the Qt window runs the
    spec §2 ready → running → result → ready loop over one
    shared camera handle instead of a single round. G3 W3: every round
    gets its own attempt and the operator verdict key persists into
    that round's label sidecar. G3 W8: the key press commits the round
    at once (bundle + attempt + csv row); the close-out tail only
    aborts the never-labeled remainder.

    mode (D3b, decision d-20260929193651396921-13): ``"record"`` is the
    research executor and stays byte-identical to b9e3bb8 — encrypted
    staging, attempt ledger, and the frozen 17-column results.csv.
    ``"demo"`` is the non-recording path the double-clicked G3 App uses:
    no ResearchRecorder, no encrypted store, no attempt ledger, and one
    plaintext demo result file instead.
    """
    try:
        profile = _load_profile(profile_path)
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research live: {exc}", file=sys.stderr)
        return 2
    # D3b: the mode flag decides whether consent is required or forbidden.
    # Both directions refuse loudly — silently ignoring a consent flag in
    # demo mode would let an operator believe recording is on when it is not.
    if mode not in {"record", "demo"}:
        print(f"research live: unsupported mode {mode!r}", file=sys.stderr)
        return 2
    recording = mode == "record"
    if recording and not (record_consent and image_consent):
        print(
            "research live: explicit --record-consent and --image-consent "
            "are both required in record mode",
            file=sys.stderr,
        )
        return 2
    if not recording and (record_consent or image_consent):
        print(
            "research live: --record-consent/--image-consent are refused in "
            "demo mode: demo mode records no encrypted frames or embeddings, "
            "so consent flags would wrongly imply recording. Use "
            "--mode record to record research evidence.",
            file=sys.stderr,
        )
        return 2
    if ui not in {"fake", "qt"}:
        print(f"research live: unsupported UI {ui!r}", file=sys.stderr)
        return 2
    if qt_offscreen and ui != "qt":
        print("research live: --qt-offscreen requires --ui qt", file=sys.stderr)
        return 2
    if qt_offscreen and device != "fake":
        print(
            "research live: --qt-offscreen only supports --device fake",
            file=sys.stderr,
        )
        return 2
    if continuous and ui != "qt":
        print(
            "research live: --continuous requires --ui qt",
            file=sys.stderr,
        )
        return 2
    # G3 W6: local config supplies defaults; explicit flags win.
    if config is not None:
        from facecore.research.g3_config import ensure_config_dir, load_g3_config

        try:
            local_config = load_g3_config(config)
        except ValueError as exc:
            print(f"research live: {exc}", file=sys.stderr)
            return 2
        # The package owns the local dir (spec §3); idempotent mkdir.
        ensure_config_dir(config)
        if models is None:
            models = local_config.models_dir
        if gallery_dir is None and corpus is None:
            gallery_dir = local_config.enrollment_dir
        # G3 W8: config store_dir/key_dir take effect (explicit
        # --store/--key-dir from main already override the config, so
        # reaching this point with a config means its paths win).
        try:
            store_root = resolve_store(local_config.store_dir)
        except (ValueError, StorePathError) as exc:
            print(f"research live: {exc}", file=sys.stderr)
            return 2
        key_dir = local_config.key_dir

    from datetime import timedelta

    from facecore.research.experiment import AttemptRecord

    now = _now_utc()
    # G3 W4: record + image retention follows G3_RETENTION_DAYS (spec §3,
    # 30 days), replacing the former hardcoded 30d/7d TTLs.
    consent = ConsentRecord(
        session_id=session_id,
        participant_id="cli-operator",
        record_consent=True,
        image_consent=True,
        consented_at_utc=now.isoformat(),
        record_expires_at_utc=(now + timedelta(days=G3_RETENTION_DAYS)).isoformat(),
        image_expires_at_utc=(now + timedelta(days=G3_RETENTION_DAYS)).isoformat(),
    )

    # E3 wiring (1): attempt pre-placement BEFORE camera open / model setup.
    # The durable encrypted write is the accepted-Start boundary (spec §5);
    # any later open/setup failure must still leave exactly one attempt.
    # The attempt id is namespaced away from the session id: recorder.begin
    # creates rk_{session_id} for image staging, so reusing the raw session
    # id as attempt id would collide DEKs and destroy the attempt on abort.
    resolved_attempt_id = attempt_id or f"att-{session_id}"
    attempt_manifest = _default_experiment_manifest(experiment_id)
    attempt = AttemptRecord(
        experiment_id=experiment_id,
        attempt_id=resolved_attempt_id,
        participant_id="cli-operator",
        visit_id="visit-cli-001",
        condition_id="cond-cli-live",
        attempt_index=1,
        retry_of=None,
        consent_ref=session_id,
        requested_at_utc=now.isoformat(),
        accepted_at_utc=now.isoformat(),
        started_at_utc=None,
        ended_at_utc=None,
        operational_status="accepted",
        error_code=None,
        bundle_ref=None,
    )
    # D3b: demo mode never constructs a ResearchRecorder (its __init__ would
    # mkdir the store root and its begin_attempt would open the ledger).
    # The plaintext demo file lives in the same store directory and creates
    # its own parent; nothing else is written.
    recorder: ResearchRecorder | Any
    if recording:
        recorder = ResearchRecorder(
            store_root=store_root, key_dir=key_dir, clock=_now_utc
        )
        try:
            recorder.begin_attempt(attempt_manifest, attempt, consent)
        except (PermissionError, ValueError) as exc:
            print(f"research live: attempt refused: {exc}", file=sys.stderr)
            return 4
    else:
        recorder = NULL_RECORDER

    def _finish_attempt_error(error_code: str) -> None:
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="open_error",
                error_code=error_code,
            )
        except Exception:
            pass

    window_label = "early-stop"
    source: CaptureSource
    context: Any = None
    # E3 wiring (2): live trace + diagnostic/event sinks on the true path.
    trace_diags: list[Any] = []
    # D7-A W3 rework 2 (R3): NO CONSUMER. `trace_events` is appended to by
    # `_event_sink` and read nowhere in the repo — it exists so the trace
    # channel has the same shape as `trace_diags`, and is retained as the
    # wiring point W5 will read when the demo/research comparison lands.
    # Do not read a value from it today: there is none. Per-round counts
    # reach the demo log through `SessionEngine.event_counts()` /
    # `support_clear_reasons()` instead, which do not need this buffer.
    trace_events: list[Any] = []
    # G3 W5: the live desktops (first round + continuous rounds) that
    # currently own the trace writer. The scorer closures below are
    # defined before the desktops exist; late binding routes each
    # emitted diag to the live desktop's pending store.
    diag_desktops: list[DesktopSession] = []

    def _diagnostic_sink(diag: Any) -> None:
        trace_diags.append(diag)
        for live_desktop in diag_desktops:
            live_desktop.note_diagnostics(diag)

    def _event_sink(event: Any) -> None:
        trace_events.append(event)

    if device == "fake":
        model_generation = "cli-fake-gen-1"
        gallery_digest = "cli-fake-gallery"
        engine = SessionEngine(
            profile,
            gallery_digest,
            model_generation,
            event_sink=_event_sink,
        )
        if capture_factory is not None:
            source = capture_factory(device)
        else:
            frames = [
                FramePacket(
                    sequence=seq,
                    captured_ns=seq * 200_000_000,
                    rgb=np.ascontiguousarray(
                        np.full((16, 16, 3), 120 + (seq % 40), dtype=np.uint8)
                    ),
                )
                for seq in range(1, 8)
            ]
            source = FakeCapture(frames=frames)
        scorer = _fake_scorer(model_generation, gallery_digest)
        is_true_path = False
    else:
        # True camera path (Task A): external models/gallery required,
        # fail-clear otherwise. Fake path above is untouched.
        if models is None or (corpus is None and gallery_dir is None):
            print(
                "research live: --device <id> requires --models <dir> and "
                "--corpus <manifest> (or --gallery-dir <folder>); use "
                "--device fake for camera-free operation",
                file=sys.stderr,
            )
            _finish_attempt_error("setup_error:missing_models_corpus")
            return 2
        if capture_factory is not None:
            source = capture_factory(device)
        else:
            source = OpenCVCapture()
        # G3 W7 picker mode (Qt window owns a camera dropdown): the
        # operator has not picked yet, so the startup check is
        # enumeration-only (at least one camera, never a refusal when
        # one exists). The picked index opens on selection; a wrong
        # pick fails in the window (worst case the app closes).
        picker_mode = ui == "qt" and capture_factory is None
        if picker_mode:
            from facecore.live.camera_picker import list_cameras

            try:
                if not list_cameras():
                    print("research live: 找不到相機", file=sys.stderr)
            except Exception as exc:
                print(f"research live: 相機列舉失敗：{exc}", file=sys.stderr)
        else:
            # Fix (a+c): open the requested device now (not fallback 0) and
            # probe the first frame BEFORE heavy model loading; a
            # dry/disconnected device fails clear here instead of committing
            # a silent 0-frame bundle.
            try:
                source.open(device)
            except Exception as exc:
                print(
                    f"research live: cannot open device {device!r}: {exc}",
                    file=sys.stderr,
                )
                _finish_attempt_error("open_error:camera_open_failed")
                return 2
            probe = source.read()
            source.close()
            if probe is None:
                print(
                    f"research live: device {device!r} opened but delivered "
                    "no frames; refusing to start",
                    file=sys.stderr,
                )
                _finish_attempt_error("open_error:no_frames")
                return 2
        try:
            if presence_mode == "checkpoint":
                # Checkpoint builds the detector ONLY (no embedder, no
                # gallery): presence scoring never reads them, and demanding
                # faceless-rejecting enrollment faces for an unread gallery
                # would force a human-face corpus (G3 boundary). Collection
                # below is byte-unchanged (full gallery path).
                checkpoint_detector = _build_true_detector_only(
                    models,
                    detector_factory=detector_factory,
                )
                context = None
            else:
                checkpoint_detector = None
                try:
                    context = _build_true_context(
                        models,
                        corpus,
                        profile,
                        detector_factory=detector_factory,
                        embedder_factory=embedder_factory,
                        gallery_dir=gallery_dir,
                    )
                except ModelSetupError as exc:
                    # G3 W7: model failures name the model cause in
                    # Chinese (split from gallery failures per the W6
                    # minor finding).
                    print(f"research live: {exc}", file=sys.stderr)
                    _finish_attempt_error("setup_error:model_setup_failed")
                    return 2
                except (ValueError, FileNotFoundError) as exc:
                    # G3 W6 spec §7-2: gallery startup failures name the
                    # cause in Chinese (e.g. which enrollment photo).
                    print(f"research live: 註冊組建立失敗：{exc}", file=sys.stderr)
                    _finish_attempt_error("setup_error:gallery_build_failed")
                    return 2
                except Exception as exc:
                    print(
                        f"research live: true pipeline setup failed: {exc}",
                        file=sys.stderr,
                    )
                    _finish_attempt_error("setup_error:model_setup_failed")
                    return 2
        except Exception as exc:
            print(f"research live: true pipeline setup failed: {exc}", file=sys.stderr)
            _finish_attempt_error("setup_error:model_setup_failed")
            return 2
        if presence_mode == "checkpoint":
            # Detector-only path has no gallery; the checkpoint branch
            # below sets synthetic generation/digest. Keep the names bound
            # so the shared tail below type-checks; they are overwritten
            # before any use.
            assert checkpoint_detector is not None
            model_generation = ""
            gallery_digest = ""
        else:
            assert context is not None
            model_generation = context.gallery.generation
            gallery_digest = context.gallery.digest
        # D7-A W3: the sinks are defined ABOVE the `device == "fake"`
        # branch, not inside this one.
        #
        # Why they had to move: W3 attaches `event_sink` to the per-round
        # engine built in `next_session_factory`, and that closure sits in
        # the `ui == "qt"` branch — which does not intersect this one. With
        # the definition below, a reference to `_event_sink` from inside
        # `next_session_factory` is a free-variable lookup that never binds.
        # So this is the structural adjustment the new requirement needed,
        # not a repair of a pre-existing bug: the per-round engine simply
        # had no sink to attach to.
        engine = SessionEngine(
            profile,
            gallery_digest,
            model_generation,
            event_sink=_event_sink,
        )

        from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
            score_frame,
        )

        if presence_mode == "checkpoint":
            # No-participant checkpoint: hybrid scorer (true detector
            # presence + synthetic identity). The detector is the true
            # YuNet built detector-only above (no embedder, no gallery —
            # no enrollment faces demanded, G3 NOT opened); identity never
            # goes true. The corpus manifest is accepted but never read
            # for faces on this path.
            from facecore.pipeline.yunet import (  # noqa: PLC0415
                YuNetDetector,
            )

            if not isinstance(checkpoint_detector, YuNetDetector):
                print(
                    "research live: checkpoint presence mode requires "
                    "the true YuNet detector",
                    file=sys.stderr,
                )
                _finish_attempt_error("setup_error:presence_detector")
                return 2
            model_generation = "checkpoint-presence-gen-1"
            gallery_digest = "checkpoint-presence-gallery"
            # D7-A W3: same sink as every other engine, so a checkpoint
            # round's events are not the one path that silently discards
            # them. This branch re-assigns `engine` (it did so before W3
            # too); the sink rides along rather than being left behind.
            engine = SessionEngine(
                profile,
                gallery_digest,
                model_generation,
                event_sink=_event_sink,
            )
            _hybrid = presence_scorer(
                checkpoint_detector,
                model_generation,
                gallery_digest,
                presence_mode="checkpoint",
            )

            def scorer(packet: FramePacket) -> FrameObservation:
                return _hybrid(packet)
        else:
            assert context is not None  # set in the collection fork above

            def scorer(packet: FramePacket) -> FrameObservation:
                return score_frame(packet, context, diagnostic_sink=_diagnostic_sink)

        window_label = "early-stop"
        is_true_path = True

    # G3 R1 PR-A change 5 (lead ruling m-20260924090052055374-357):
    # the continuous Qt loop never starts the initial desktop (Ready;
    # replaces it while idle), so its session must not occupy the
    # recorder's single-active slot — the first round's begin would
    # otherwise coexist and every append_frame refuses with KeyError.
    # Only the session staging begin is skipped (the attempt ledger
    # begin_attempt above still runs, so the tail's finish_attempt for
    # the initial attempt id updates a real row). The tail's abort for
    # the never-begun session id is a no-op (abort pops missing ids)
    # and every tail finish_attempt sits inside try/except, so skipping
    # is safe and leaves single-shot/headless paths unchanged.
    if recording and not (continuous and ui == "qt"):
        try:
            recorder.begin(session_id, consent)
        except (PermissionError, ValueError) as exc:
            print(f"research live: recorder refused: {exc}", file=sys.stderr)
            return 4
    # t-3: true path stages each sampled frame encrypted as it is scored;
    # fake path keeps envelope-only behavior. Qt also receives frames for its
    # preview/crop mapping.
    #
    # E7-B Appendix A.2-A.3: the center-square mapping is applied to the
    # scorer input (capture adapter), while mirror stays preview-only.
    # The default transform is identity so the headless fake path is
    # untouched; Qt/offscreen synthetic smoke wires the square crop.
    from facecore.live.qt_window import crop_packet as _crop_packet

    staged_errors: list[str] = []
    qt_window: Any = None

    def _square_capture_transform(packet: FramePacket) -> FramePacket:
        cropped_packet, mapping = _crop_packet(packet)
        # D3b: the crop mapping is per-round spatial state that only the
        # encrypted research bundle needs. Demo mode has no bundle, so the
        # write is skipped rather than routed through a null ledger.
        if recording:
            try:
                recorder.record_crop_mapping(
                    resolved_attempt_id, mapping.to_dict()
                )
            except ValueError as exc:
                # Geometry mismatch across frames: same-frame evidence would be
                # unreconstructible, so the scorer input is refused fail-closed.
                raise ValueError(
                    f"capture geometry changed mid-session: {exc}"
                ) from exc
            except Exception as exc:
                staged_errors.append(f"crop:{type(exc).__name__}")
        return cropped_packet

    def _stage_frame(packet: FramePacket) -> None:
        # D3b: append_frame is the ONE biometric channel — raw frame pixels.
        # Gating it on "is recording" rather than on the recorder's identity
        # is what makes demo mode provably write no frames.
        if is_true_path and recording:
            try:
                recorder.append_frame(packet)
            except Exception as exc:
                staged_errors.append(f"{packet.sequence}:{type(exc).__name__}")
        if qt_window is not None:
            try:
                # A.7: the preview renders the original full frame; the
                # scorer-side transform owns the mapping write, so the sink
                # must not re-crop (that double-crop diverges the mapping).
                qt_window.render_full_frame(packet.rgb)
            except Exception as exc:
                staged_errors.append(f"crop:{type(exc).__name__}")

    # D3b: the demo result file, resolved once so the initial desktop and
    # every round the next_session factory builds share the same target.
    #
    # D7-A #141: named for when THIS APP EXECUTION started, not for
    # `session_id` (which is per-round here). Every round of this run lands
    # in one file; the next App start opens a new one. That is what keeps an
    # existing log's header from needing an in-place upgrade — the header
    # guard in `append_g3_demo_results_csv` is the backstop, not the
    # everyday path. The stamp is `now`, this App's start instant.
    demo_csv_path = demo_results_csv_path(store_root, now.isoformat())

    desktop = DesktopSession(
        engine=engine,
        source=source,
        scorer=scorer,
        session_id=session_id,
        frame_sink=_stage_frame if (is_true_path or ui == "qt") else None,
        frame_transform=_square_capture_transform if ui == "qt" else None,
        fixed_seconds=fixed_seconds,
        # D4-F1 (issue #123): gate the trace channel on `recording`, not on
        # the recorder's IDENTITY. `is_true_path` answers "is this the real
        # camera path", which is still true in demo mode — so it handed
        # NULL_RECORDER to the trace writer, the controller's None-guard
        # let it through, and `_NullRecorder.__getattr__` raised on the
        # append_trace probe. Demo mode now receives no trace writer at
        # all, which is also what `_stage_frame` and `record_crop_mapping`
        # below already do. `trace_attempt_id` moves with it: the guard is
        # an `or`, so leaving the id behind would re-open the same hole.
        trace_recorder=recorder if recording else None,
        trace_attempt_id=resolved_attempt_id if recording else None,
        # D3b: the plaintext demo sink for the initial desktop. The rounds
        # built by next_session_factory below get the same value.
        demo_label_sink=demo_csv_path if not recording else None,
        # G3 W2: the continuous loop keeps one camera handle across
        # rounds; the terminal path must not release it mid-loop.
        release_source_on_terminal=False if continuous else True,
    )
    if is_true_path:
        # G3 W5: route scorer-emitted true diagnostics to the live
        # desktop's pending store for the encrypted trace writer.
        diag_desktops.append(desktop)
    import time as _time

    # Fix (b): the session clock anchors at the live monotonic clock on the
    # true path (fake path keeps its synthetic zero-origin stamps, so its
    # start stays 0 and its envelope stays consistent).
    start_ns = _time.monotonic_ns() if is_true_path else 0
    # S2: the Qt countdown reads the session clock, not wall time. The Qt
    # smoke path keeps the synthetic zero-origin stamps, so the countdown
    # clock tracks synthetic session time: session start plus the elapsed
    # synthetic capture span consumed so far. The window updates it after
    # each processing tick (see _qt_advance_ns below).
    _qt_elapsed_ns = 0
    _qt_last_consumed_ns = start_ns

    def _qt_clock_ns() -> int:
        return start_ns + _qt_elapsed_ns

    def _qt_advance_ns() -> None:
        # G3 R1 PR-A change 2: the countdown clock follows the live
        # round's consumed span, not the never-started initial desktop.
        # qt_window is late-bound (None until built below); before that
        # the initial desktop is the only source.
        nonlocal _qt_elapsed_ns, _qt_last_consumed_ns
        live_desktop = desktop
        window = qt_window
        if window is not None:
            try:
                live_desktop = window.desktop
            except Exception:
                pass
        consumed = live_desktop.controller_consumed_ns
        if consumed is not None and consumed > _qt_last_consumed_ns:
            _qt_elapsed_ns += consumed - _qt_last_consumed_ns
            _qt_last_consumed_ns = consumed

    qt_clock_ns: Callable[[], int] = _qt_clock_ns
    qt_app: Any = None
    if ui == "qt":
        try:
            from PySide6.QtWidgets import QApplication
            from facecore.live.qt_window import QtResearchWindow
        except ImportError as exc:
            print(
                f"research live: Qt UI unavailable: {exc}; install research-ui",
                file=sys.stderr,
            )
            _finish_attempt_error("setup_error:qt_dependency_missing")
            return 2
        if qt_offscreen:
            # The Qt smoke path is synthetic and never touches a camera device.
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        qt_app = QApplication.instance() or QApplication([])
        # G3 W6: picker options for the Qt dropdown. Listing only: no
        # auto-select, no uid/shape assertion (that path is untouched).
        # An empty list means no camera (the window shows 找不到相機).
        camera_options: list[tuple[int, str]] = []
        if device != "fake":
            try:
                from facecore.live.camera_picker import list_cameras

                camera_options = [
                    (option.index, option.label) for option in list_cameras()
                ]
            except Exception as exc:
                print(
                    f"research live: 相機列舉失敗：{exc}",
                    file=sys.stderr,
                )
                camera_options = []
        next_session_factory = None
        if continuous:
            from datetime import timedelta as _td

            from facecore.research.experiment import AttemptRecord as _Attempt

            round_counter = 0

            def next_session_factory() -> tuple[
                DesktopSession, ConsentRecord, str | None
            ]:
                nonlocal round_counter
                round_counter += 1
                round_session_id = f"{session_id}-r{round_counter}"
                round_attempt_id = f"{resolved_attempt_id}-r{round_counter}"
                round_now = _now_utc()
                round_consent = ConsentRecord(
                    session_id=round_session_id,
                    participant_id="cli-operator",
                    record_consent=True,
                    image_consent=True,
                    consented_at_utc=round_now.isoformat(),
                    record_expires_at_utc=(
                        round_now + _td(days=G3_RETENTION_DAYS)
                    ).isoformat(),
                    image_expires_at_utc=(
                        round_now + _td(days=G3_RETENTION_DAYS)
                    ).isoformat(),
                )
                # G3 W3: each round gets its own attempt so the operator
                # verdict key persists into that round's label sidecar.
                # G3 W4: each round also opens its own session staging so
                # the CLI tail can commit the encrypted bundle per round.
                # A begin failure fails closed: the round never starts and
                # Ready shows the error.
                if recording:
                    recorder.begin_attempt(
                        attempt_manifest,
                        _Attempt(
                            experiment_id=experiment_id,
                            attempt_id=round_attempt_id,
                            participant_id="cli-operator",
                            visit_id="visit-cli-001",
                            condition_id="cond-cli-live",
                            attempt_index=round_counter + 1,
                            retry_of=None,
                            consent_ref=round_session_id,
                            requested_at_utc=round_now.isoformat(),
                            accepted_at_utc=round_now.isoformat(),
                            started_at_utc=None,
                            ended_at_utc=None,
                            operational_status="accepted",
                            error_code=None,
                            bundle_ref=None,
                        ),
                        round_consent,
                    )
                    recorder.begin(round_session_id, round_consent)

                # G3 R1 PR-A change 4: the round owns its frame path —
                # staging binds the round session, the square-crop
                # mapping binds the round attempt. Mirrors the initial
                # desktop closures above, scoped to this round's ids.
                def _round_capture_transform(
                    packet: FramePacket,
                    _attempt_id: str = round_attempt_id,
                ) -> FramePacket:
                    cropped_packet, mapping = _crop_packet(packet)
                    # D3b: demo mode has no bundle to bind the mapping to.
                    if not recording:
                        return cropped_packet
                    try:
                        recorder.record_crop_mapping(_attempt_id, mapping.to_dict())
                    except ValueError as exc:
                        raise ValueError(
                            f"capture geometry changed mid-session: {exc}"
                        ) from exc
                    except Exception as exc:
                        staged_errors.append(f"crop:{type(exc).__name__}")
                    return cropped_packet

                def _round_stage_frame(packet: FramePacket) -> None:
                    # D3b: raw pixels go to the encrypted store only while
                    # recording. The preview below is a separate callback and
                    # is NOT gated here — previewing never stages a frame.
                    if is_true_path and recording:
                        try:
                            recorder.append_frame(packet)
                        except Exception as exc:
                            staged_errors.append(
                                f"{packet.sequence}:{type(exc).__name__}"
                            )
                    # qt_window is late-bound (None until the window is
                    # built below); the closure reads it at call time.
                    if qt_window is not None:
                        try:
                            qt_window.render_full_frame(packet.rgb)
                        except Exception as exc:
                            staged_errors.append(f"crop:{type(exc).__name__}")

                round_desktop = DesktopSession(
                    # D7-A W3: this per-round engine is built inline, so it
                    # is the one place the live loop's rounds were getting an
                    # engine with no `event_sink` — the first round's engine
                    # (built above) has had one all along. Counting no longer
                    # depends on the sink (SessionEngine tallies every event
                    # before the sink guard), but wiring it keeps the two
                    # rounds' event behaviour identical instead of leaving a
                    # difference that only the first round can see.
                    engine=SessionEngine(
                        profile,
                        gallery_digest,
                        model_generation,
                        event_sink=_event_sink,
                    ),
                    source=source,
                    scorer=scorer,
                    session_id=round_session_id,
                    frame_sink=_round_stage_frame
                    if (is_true_path or ui == "qt")
                    else None,
                    frame_transform=_round_capture_transform if ui == "qt" else None,
                    fixed_seconds=fixed_seconds,
                    release_source_on_terminal=False,
                    # D3b: the encrypted label sidecar only exists while
                    # recording. Demo mode labels into the plaintext demo file.
                    label_recorder=recorder if recording else None,
                    label_attempt_id=round_attempt_id if recording else None,
                    # D3b: every round needs the demo sink, not just the
                    # initial desktop, or the operator verdict is refused
                    # with 「標註未綁定」 from the second round onward.
                    demo_label_sink=None if recording else demo_csv_path,
                    # D4-F1: same gate as the initial desktop above, and
                    # for the same reason — the round's trace writer must
                    # follow `recording`, not the recorder's identity.
                    trace_recorder=recorder if recording else None,
                    trace_attempt_id=round_attempt_id if recording else None,
                )
                if is_true_path:
                    # G3 W5: the new round owns the trace writer now.
                    diag_desktops.clear()
                    diag_desktops.append(round_desktop)
                return (
                    round_desktop,
                    round_consent,
                    round_attempt_id,
                )

        qt_window = QtResearchWindow(
            desktop,
            consent=consent,
            # D3b: the window distinguishes record from demo by the
            # recorder's IDENTITY, not by a separate flag. Passing
            # NULL_RECORDER would make demo mode look like "a recorder
            # exists but does nothing" and would route verdicts into
            # results.csv, so demo mode passes None here.
            recorder=recorder if recording else None,
            attempt_id=resolved_attempt_id if recording else None,
            device_id=device,
            offscreen=qt_offscreen,
            clock_ns=qt_clock_ns,
            clock_advance=_qt_advance_ns,
            next_session=next_session_factory,
            camera_options=camera_options if device != "fake" else None,
            # G3 W8: commit on label needs the csv target up front. D3b: the
            # research ledger target exists ONLY while recording, so demo
            # mode can never reach append_g3_results_csv.
            results_csv=store_root / "results.csv"
            if next_session_factory is not None and recording
            else None,
            # D3b: the plaintext demo result file, same store directory.
            demo_results_csv=demo_csv_path
            if next_session_factory is not None and not recording
            else None,
            # D2: only a real Qt event loop gets worker-driven inference.
            # A real GUI would otherwise block on inference, encrypted
            # staging and camera reads inside one timer callback. The
            # offscreen driver below and the headless path keep the
            # synchronous tick, whose determinism and step budget the
            # existing tests pin.
            background_inference=not qt_offscreen,
            gallery=getattr(context, "gallery", None) if context is not None else None,
            enrollment_dir=gallery_dir,
        )
        if device != "fake" and not camera_options:
            # G3 W6 spec §7-2: no camera at all → Chinese reason, stay
            # put (no crash, no silent continue).
            qt_window.show_startup_error("找不到相機")
    try:
        if qt_window is None:
            desktop.on_start(consent, now_ns=start_ns, device_id=device)
        elif continuous:
            if device != "fake" and not camera_options:
                pass
            else:
                # G3 R1: the loop opens in Ready (lens shut); rounds
                # start only on operator Start.
                qt_window.enter_ready()
        else:
            qt_window.start_clicked()
    except (PermissionError, ValueError, RuntimeError) as exc:
        print(f"research live: start refused: {exc}", file=sys.stderr)
        recorder.abort(session_id, reason="start_refused")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="start_refused",
            )
        except Exception:
            pass
        return 2
    # Pump frames into the recorder's staging area as they are sampled.
    # (Desktop owns inference; recorder owns encrypted staging.)
    try:
        if qt_window is None:
            while desktop.state == "running":
                desktop.run_until_terminal(max_steps=50)
            terminal = desktop.terminal
        elif qt_offscreen:
            qt_window.process_until_terminal(max_steps=200)
            terminal = desktop.terminal
        else:
            qt_window.show()
            assert qt_app is not None
            qt_app.exec()
            terminal = desktop.terminal
    except (ValueError, RuntimeError) as exc:
        # Capture-geometry drift (e.g. frame-dimension change) refuses the
        # scorer input fail-closed mid-session. Terminal should be None
        # here; close safely and refuse the commit.
        terminal = None
        capture_failure: Exception | None = exc
    else:
        capture_failure = None
    # T2: an operator Delete in the Qt window already removed the session
    # bundle plus linked attempts. Never commit afterwards: report success
    # only once deletion is complete.
    if qt_window is not None:
        if desktop.deleted:
            _emit(
                {
                    "session_id": session_id,
                    "status": "deleted",
                    "window": window_label,
                    "elapsed_ms": 0.0,
                    "reason_codes": ["operator_deleted"],
                    "generation": model_generation,
                    "gallery_digest": gallery_digest,
                }
            )
            return 0
        if desktop.delete_failed:
            print("research live: operator deletion failed", file=sys.stderr)
            desktop.close()
            recorder.abort(session_id, reason="delete_failed")
            try:
                recorder.finish_attempt(
                    resolved_attempt_id,
                    result=None,
                    operational_status="error",
                    error_code="delete_failed",
                )
            except Exception:
                pass
            return 4
    if capture_failure is not None:
        if isinstance(capture_failure, PresenceDetectedError):
            print(
                f"research live: presence stop: {capture_failure}",
                file=sys.stderr,
            )
            desktop.close()
            recorder.abort(session_id, reason="presence_stop")
            try:
                recorder.finish_attempt(
                    resolved_attempt_id,
                    result=None,
                    operational_status="error",
                    error_code="presence_stop",
                )
            except Exception:
                pass
            return 4
        print(f"research live: capture failed: {capture_failure}", file=sys.stderr)
        desktop.close()
        recorder.abort(session_id, reason="capture_failed")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="capture_failed",
            )
        except Exception:
            pass
        return 4
    # Presence-guard canonical-path fix (issue #82): the controller can
    # only re-raise scorer exceptions when a frame_transform is present
    # (ui == "qt"); on the runner/runbook canonical path (no transform)
    # it surfaces them as an error terminal with a scorer_failure reason
    # code instead. Detect the presence stop HERE by terminal reason code
    # — never by exception propagation (Qt event-loop re-raise is
    # unverified) and never by a hand-copied stderr literal. The marker
    # is built from the exception class name so emitter and detector
    # cannot drift apart. The faced bundle is NOT committed: a stopped
    # run is not a research product (R4 fail-closed spirit).
    _presence_marker = f"scorer_failure: {PresenceDetectedError.__name__}"
    if terminal is not None and any(
        _presence_marker in code for code in terminal.reason_codes
    ):
        print(
            "research live: presence stop: "
            f"{_presence_marker} in no-participant checkpoint session "
            f"{session_id}",
            file=sys.stderr,
        )
        desktop.close()
        recorder.abort(session_id, reason="presence_stop")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="presence_stop",
            )
        except Exception:
            pass
        return 4
    # G3 W4 continuous close-out (answers the W2 reviewer Note): the
    # initial desktop never started (enter_ready leaves it idle while
    # idle), so `terminal` above is always None here. Exit code is
    # defined by labeled rounds: >= 1 committed round → rc0, otherwise
    # rc4. Unlabeled rounds (result shown but no key press) are aborted,
    # never committed: a record without an operator verdict is not G3
    # test data (spec §3 requires the label column).
    if continuous and qt_window is not None:
        rounds = list(qt_window.completed_rounds)
        committed_ids = {round_.session_id for round_ in rounds}
        if staged_errors:
            print(
                "research live: staging failed: "
                + ";".join(sorted(set(staged_errors))),
                file=sys.stderr,
            )
            for round_ in rounds:
                if round_.attempt_id is not None:
                    try:
                        recorder.finish_attempt(
                            round_.attempt_id,
                            result=None,
                            operational_status="error",
                            error_code="staging_failed",
                        )
                    except Exception:
                        pass
            try:
                recorder.abort(session_id, reason="staging_failed")
            except Exception:
                pass
            desktop.close()
            qt_window.close()
            return 4
        # G3 W8: rounds committed on label already; the tail only aborts
        # the never-labeled remainder (current round + initial session).
        try:
            current_session_id = qt_window.desktop.session_id
        except Exception:
            current_session_id = None
        try:
            current_session_id = qt_window.desktop.session_id
        except Exception:
            current_session_id = None
        for pending_id, pending_attempt in (
            (session_id, resolved_attempt_id),
            (current_session_id, qt_window.attempt_id),
        ):
            if pending_id is None or pending_id in committed_ids:
                continue
            try:
                recorder.abort(pending_id, reason="unlabeled_close")
            except Exception:
                pass
            if pending_attempt is not None:
                try:
                    recorder.finish_attempt(
                        pending_attempt,
                        result=None,
                        operational_status="cancelled",
                        error_code=None,
                    )
                except Exception:
                    pass
        desktop.close()
        qt_window.close()
        committed = len(rounds)
        if committed > 0:
            # D3b: name the file this mode actually wrote. Emitting
            # results.csv in demo mode would point an operator (or a
            # script) at a file the run never touched.
            _emit(
                {
                    "session_id": session_id,
                    "status": "continuous_complete",
                    "window": window_label,
                    "rounds_committed": committed,
                    "mode": mode,
                    (
                        "results_csv"
                        if recording
                        else "demo_results_csv"
                    ): str(
                        store_root
                        / (
                            "results.csv"
                            if recording
                            else demo_csv_path.name
                        )
                    ),
                    "generation": model_generation,
                    "gallery_digest": gallery_digest,
                }
            )
            return 0
        print(
            "research live: continuous close-out without a labeled round",
            file=sys.stderr,
        )
        return 4
    if terminal is None:
        print("research live: no terminal reached", file=sys.stderr)
        desktop.close()
        recorder.abort(session_id, reason="no_terminal")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="no_terminal",
            )
        except Exception:
            pass
        return 4
    # R4 (fail-closed): any capture/staging failure refuses the commit.
    # Staging errors carry geometry/staging integrity evidence, so a
    # successful bundle must never mask them.
    if staged_errors:
        print(
            "research live: staging failed: " + ";".join(sorted(set(staged_errors))),
            file=sys.stderr,
        )
        desktop.close()
        recorder.abort(session_id, reason="staging_failed")
        try:
            recorder.finish_attempt(
                resolved_attempt_id,
                result=None,
                operational_status="error",
                error_code="staging_failed",
            )
        except Exception:
            pass
        return 4
    if fixed_seconds:
        # Fixed-window comparison mode: inference terminal is locked, the
        # collector ran to the original deadline, and the window label
        # reflects collector evidence (not a fabricated 5s claim).
        window_label = (
            "fixed-window-complete"
            if desktop.collection_complete
            else "fixed-window-incomplete"
        )
    # t-3: per-frame best-match ledger from scored observations; the
    # terminal matched_identity is still written only on matched.
    frame_scores = tuple(frame_score_of(obs) for obs in desktop.observations)
    collection_window = None
    if fixed_seconds:
        from facecore.research.records import CollectionWindow

        deadline_ns = start_ns + int(profile.timeout_ms * 1_000_000)
        collection_window = CollectionWindow(
            session_id=session_id,
            collection_start_ns=start_ns,
            collection_deadline_ns=deadline_ns,
            collection_end_ns=None,
            collection_stop_reason=desktop.collection_stop_reason,
            collection_complete=desktop.collection_complete,
            frames_sampled=len(desktop.observations),
        )
    try:
        recorder.commit(
            terminal,
            frame_scores=frame_scores,
            collection_window=collection_window,
        )
    except (KeyError, ValueError) as exc:
        print(f"research live: commit failed: {exc}", file=sys.stderr)
        desktop.close()
        return 4
    # E3: close the attempt ledger with the operational outcome and link
    # the committed session bundle for trace recovery.
    try:
        op_status = (
            "completed" if terminal.status.value == "matched" else terminal.status.value
        )
        if op_status not in (
            "accepted",
            "open_error",
            "setup_error",
            "cancelled",
            "timeout",
            "completed",
            "error",
        ):
            op_status = "completed"
        recorder.finish_attempt(
            resolved_attempt_id,
            result=terminal,
            operational_status=op_status,
            error_code=None,
        )
    except Exception as exc:
        print(f"research live: finish_attempt failed: {exc}", file=sys.stderr)
        desktop.close()
        return 4
    # R3: the fixed-window collector may end running after B locks. Only a
    # terminal session accepts an evaluator label; an incomplete collector
    # closes without labeling instead of raising.
    if qt_window is None:
        if desktop.state == "terminal":
            desktop.label_terminal(None)
        desktop.close()
    else:
        qt_window.close()
    _emit(
        {
            "session_id": session_id,
            "status": terminal.status.value,
            "window": window_label,
            "elapsed_ms": terminal.elapsed_ms,
            "reason_codes": list(terminal.reason_codes),
            "generation": model_generation,
            "gallery_digest": gallery_digest,
        }
    )
    return 0


def cmd_replay(
    *,
    store: Path,
    key_dir: Path,
    session_id: str,
    profile_path: Path,
    models: Path | None = None,
    corpus: Path | None = None,
    detector_factory: Callable[[Path], Any] | None = None,
    embedder_factory: Callable[[Path], Any] | None = None,
) -> int:
    """Replay one committed bundle deterministically (labels never enter).

    t-3: with --models/--corpus the true gallery is rebuilt and true
    bundles replay without generation_mismatch; without them the legacy
    fake generation is used (true bundles then refuse, fail-closed).
    """
    scorer: Callable[[FramePacket], FrameObservation]
    try:
        profile = _load_profile(profile_path)
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research replay: {exc}", file=sys.stderr)
        return 2
    if models is not None or corpus is not None:
        if models is None or corpus is None:
            print(
                "research replay: --models and --corpus must be given together",
                file=sys.stderr,
            )
            return 2
        try:
            context = _build_true_context(
                models,
                corpus,
                profile,
                detector_factory=detector_factory,
                embedder_factory=embedder_factory,
            )
        except Exception as exc:
            print(f"research replay: gallery rebuild failed: {exc}", file=sys.stderr)
            return 2
        from facecore.live.frame_pipeline import (  # noqa: PLC0415 (device-gated)
            score_frame,
        )

        def _scorer(packet: FramePacket) -> FrameObservation:
            return score_frame(packet, context)

        scorer = _scorer
        model_generation = context.gallery.generation
        gallery_digest = context.gallery.digest
    else:
        scorer = _fake_scorer("cli-fake-gen-1", "cli-fake-gallery")
        model_generation = "cli-fake-gen-1"
        gallery_digest = "cli-fake-gallery"
    try:
        replayed = replay_session(
            session_id,
            store_root=store_root,
            key_dir=key_dir,
            clock=_now_utc,
            profile=profile,
            scorer=scorer,
            model_generation=model_generation,
            gallery_digest=gallery_digest,
        )
    except ReplayRefusal as exc:
        _emit(
            {
                "session_id": session_id,
                "status": "error",
                "kind": exc.kind,
                "detail": exc.detail,
            }
        )
        return 4
    _emit(
        {
            "session_id": session_id,
            "status": replayed.result.status.value,
            "window": replayed.window,
            "frames_replayed": replayed.frames_replayed,
            "elapsed_ms": replayed.result.elapsed_ms,
        }
    )
    return 0


def cmd_delete(*, store: Path, key_dir: Path, session_id: str) -> int:
    """Tombstone-first delete of one session bundle."""
    try:
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research delete: {exc}", file=sys.stderr)
        return 2
    recorder = ResearchRecorder(store_root=store_root, key_dir=key_dir, clock=_now_utc)
    try:
        ok = recorder.delete(session_id)
    except StoreCorruptionError as exc:
        print(f"research delete: {exc}", file=sys.stderr)
        _emit({"session_id": session_id, "deleted": False, "error": str(exc)})
        return 4
    _emit({"session_id": session_id, "deleted": ok})
    return 0 if ok else 4


def cmd_analyze(
    *,
    store: Path,
    key_dir: Path,
    experiment_id: str,
    mode: str = "development",
    profile_path: Path | None = None,
) -> int:
    """Analyze a batch of attempts using evaluate_arms and analyze_batch (F5).

    Production path:
    - Lists durable attempts for experiment_id from ResearchRecorder.
    - End-to-end calls evaluate_arms(trace, profile, window=window) for every
      attempt with a diagnostic trace.
    - Loads latest evaluation labels for ground truth.
    - Executes analyze_batch to compute denominators, arm rates, and triggers.
    - Encrypts and persists detailed case summaries to AEAD store under _cases.
    - Emits aggregate summary and run code to stdout (zero sensitive leaks).
    """
    if mode not in ("development", "holdout"):
        print(
            f"research analyze: mode {mode!r} is rejected; "
            "only 'development' and 'holdout' are supported",
            file=sys.stderr,
        )
        return 2

    try:
        store_root = resolve_store(store)
    except (ValueError, StorePathError) as exc:
        print(f"research analyze: {exc}", file=sys.stderr)
        return 2

    if not store_root.is_dir():
        print(
            f"research analyze: store {store_root} does not exist",
            file=sys.stderr,
        )
        return 4

    if profile_path is None:
        print(
            "research analyze: --profile is required and must match stored provenance",
            file=sys.stderr,
        )
        return 2

    try:
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = ResearchProfile.from_dict(profile_data)
    except Exception as exc:
        print(
            f"research analyze: invalid profile at {profile_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    profile_digest = profile.profile_digest()

    recorder = ResearchRecorder(store_root=store_root, key_dir=key_dir, clock=_now_utc)

    freeze = recorder.get_freeze(experiment_id)
    release = recorder.get_release(experiment_id)

    if mode == "holdout":
        if freeze is None:
            print(
                f"research analyze: experiment {experiment_id!r} lacks "
                "candidate freeze; holdout mode requires authorized freeze and release",
                file=sys.stderr,
            )
            return 4
        if release is None:
            print(
                f"research analyze: holdout for experiment {experiment_id!r} "
                "is sealed; authorized release required before analysis",
                file=sys.stderr,
            )
            return 4
        if profile_digest != freeze.profile_digest:
            recorder.record_contamination(
                ContaminationRecord(
                    contamination_id=f"cnt_{uuid4().hex[:12]}",
                    experiment_id=experiment_id,
                    reason="hash_mismatch",
                    details={
                        "expected_profile_digest": freeze.profile_digest,
                        "got_profile_digest": profile_digest,
                    },
                    detected_at_utc=_now_utc().isoformat(),
                )
            )
            print(
                f"research analyze: profile digest {profile_digest} does not match "
                f"frozen candidate profile digest {freeze.profile_digest}",
                file=sys.stderr,
            )
            return 4

    try:
        attempts = recorder.list_attempts(experiment_id=experiment_id)
    except Exception as exc:
        print(f"research analyze: failed to list attempts: {exc}", file=sys.stderr)
        return 4

    if mode == "development":
        attempts = [
            a
            for a in attempts
            if a.split == "development"
            and (freeze is None or a.visit_id not in freeze.planned_visit_ids)
        ]
    elif mode == "holdout":
        assert freeze is not None
        attempts = [
            a
            for a in attempts
            if a.split == "holdout" or a.visit_id in freeze.planned_visit_ids
        ]

    if not attempts:
        print(
            f"research analyze: no eligible {mode} attempts found for "
            f"experiment {experiment_id!r}",
            file=sys.stderr,
        )
        return 2

    # Verify profile against provenance (record/trace)
    for attempt in attempts:
        if attempt.bundle_ref:
            try:
                s_rec = recorder.read_record(attempt.bundle_ref)
                if (
                    s_rec.result.profile_digest
                    and s_rec.result.profile_digest != profile_digest
                ):
                    print(
                        f"research analyze: profile digest {profile_digest} "
                        "does not match stored record digest "
                        f"{s_rec.result.profile_digest}",
                        file=sys.stderr,
                    )
                    return 4
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: failed to verify record provenance: {exc}",
                    file=sys.stderr,
                )
                return 4

        # Check attempt metadata (covers bundle-less attempts!)
        _, _, meta = recorder._find_attempt_and_path(attempt.attempt_id)
        stored_prof_digest = meta.get("profile_digest")
        if stored_prof_digest and stored_prof_digest != profile_digest:
            print(
                f"research analyze: profile digest {profile_digest} does not match "
                f"attempt manifest policy profile digest {stored_prof_digest}",
                file=sys.stderr,
            )
            return 4
        elif not stored_prof_digest and not attempt.bundle_ref:
            trace_dir = recorder._trace_dir(attempt.attempt_id)
            if trace_dir.is_dir() and any(trace_dir.glob("frame_*.enc")):
                print(
                    f"research analyze: attempt {attempt.attempt_id} lacks verifiable "
                    "profile provenance; refusing unverified analysis",
                    file=sys.stderr,
                )
                return 4

    outcomes: list[ArmOutcome] = []
    traces: dict[str, SessionTrace] = {}

    for attempt in attempts:
        trace = None
        trace_dir = recorder._trace_dir(attempt.attempt_id)
        if trace_dir.is_dir() and any(trace_dir.glob("frame_*.enc")):
            try:
                trace = recorder.read_trace(attempt.attempt_id)
                traces[attempt.attempt_id] = trace
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: integrity violation in trace for attempt "
                    f"{attempt.attempt_id}: {exc}",
                    file=sys.stderr,
                )
                return 4

        window = None
        if attempt.bundle_ref:
            try:
                s_rec = recorder.read_record(attempt.bundle_ref)
                window = s_rec.collection_window
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: integrity violation in record for attempt "
                    f"{attempt.attempt_id}: {exc}",
                    file=sys.stderr,
                )
                return 4

        if trace is not None:
            # F5: Production path calls evaluate_arms end-to-end
            arm_a, arm_b = evaluate_arms(trace, profile, window=window)
            outcomes.extend([arm_a, arm_b])

    labels: list[EvaluationLabel] = []
    for attempt in attempts:
        lbl_dir = recorder._label_dir(attempt.attempt_id)
        if lbl_dir.is_dir() and any(lbl_dir.glob("rev_*.enc")):
            try:
                lbl = recorder.read_label(attempt.attempt_id)
                labels.append(lbl)
            except KeyError:
                pass
            except Exception as exc:
                print(
                    f"research analyze: integrity violation in label for attempt "
                    f"{attempt.attempt_id}: {exc}",
                    file=sys.stderr,
                )
                return 4

    try:
        analysis = analyze_batch(
            attempts,
            outcomes,
            labels,
            traces=traces,
            profile=profile,
            mode=mode,
            freeze=freeze,
            release=release,
        )
    except Exception as exc:
        print(f"research analyze: analyze_batch failed: {exc}", file=sys.stderr)
        return 4

    try:
        save_case_summaries(recorder, experiment_id, analysis.cases)
    except Exception as exc:
        print(
            f"research analyze: failed to save encrypted cases: {exc}",
            file=sys.stderr,
        )
        return 4

    print(analysis.render_summary())
    print(f"Run code: {experiment_id}-{mode}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Isolated research entry point (never the production CLI)."""
    parser = argparse.ArgumentParser(prog="facecore.research.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live")
    live.add_argument("--profile", required=True, type=Path)
    # G3 W8: --store falls back to the config store_dir when --config
    # is given (explicit --store still wins; neither → usage error).
    live.add_argument("--store", required=False, type=Path, default=None)
    live.add_argument("--key-dir", required=False, type=Path, default=None)
    live.add_argument("--device", required=True)
    live.add_argument("--session", required=True)
    live.add_argument("--record-consent", action="store_true")
    live.add_argument("--image-consent", action="store_true")
    live.add_argument("--fixed-seconds", action="store_true")
    live.add_argument(
        "--ui",
        choices=["fake", "qt"],
        default="fake",
        help="Research UI backend; fake is the headless default",
    )
    live.add_argument(
        "--qt-offscreen",
        action="store_true",
        help="Use offscreen Qt for synthetic smoke tests (requires --ui qt)",
    )
    live.add_argument("--corpus", required=False, type=Path, default=None)
    live.add_argument("--models", required=False, type=Path, default=None)
    live.add_argument(
        "--gallery-dir",
        required=False,
        type=Path,
        default=None,
        help=(
            "G3 W6: enrollment folder (identity = filename stem); "
            "takes precedence over --corpus manifest when both are given"
        ),
    )
    live.add_argument(
        "--config",
        required=False,
        type=Path,
        default=None,
        help=(
            "G3 W6: local JSON config (enrollment_dir/models_dir/"
            "store_dir/key_dir); explicit flags override it"
        ),
    )
    live.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "G3 R1: Qt ready → running → result → ready loop over "
            "one camera handle (requires --ui qt)"
        ),
    )
    live.add_argument(
        "--presence-mode",
        choices=["collection", "checkpoint"],
        default="collection",
        help=(
            "Presence guard mode: 'checkpoint' aborts (exit 4) when any "
            "face is detected in a no-participant run and requires the "
            "true YuNet detector; default 'collection' never stops "
            "(all existing callers unchanged)"
        ),
    )
    live.add_argument(
        "--mode",
        choices=["record", "demo"],
        default="record",
        help=(
            "D3b: 'record' (default) is the research executor — encrypted "
            "staging, attempt ledger, frozen 17-column results.csv, and it "
            "requires --record-consent --image-consent. 'demo' records "
            "nothing encrypted: no frames, no embeddings, no attempt "
            "ledger; the operator verdict goes to a plaintext demo result "
            "file instead, and passing consent flags in demo mode is a "
            "usage error rather than a silent no-op"
        ),
    )

    replay = sub.add_parser("replay")
    replay.add_argument("--store", required=True, type=Path)
    replay.add_argument("--key-dir", required=False, type=Path, default=None)
    replay.add_argument("--session", required=True)
    replay.add_argument("--profile", required=True, type=Path)
    replay.add_argument("--corpus", required=False, type=Path, default=None)
    replay.add_argument("--models", required=False, type=Path, default=None)

    delete = sub.add_parser("delete")
    delete.add_argument("--store", required=True, type=Path)
    delete.add_argument("--key-dir", required=False, type=Path, default=None)
    delete.add_argument("--session", required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--store", required=True, type=Path)
    analyze.add_argument("--key-dir", required=False, type=Path, default=None)
    analyze.add_argument("--experiment", required=True)
    analyze.add_argument(
        "--mode",
        choices=["development", "holdout"],
        default="development",
        help=(
            "Analysis mode ('development' or 'holdout' guarded by "
            "candidate freeze/release)"
        ),
    )
    analyze.add_argument(
        "--profile",
        required=True,
        type=Path,
        help="Frozen research profile JSON (must match stored provenance)",
    )

    args = parser.parse_args(argv)
    # G3 W8: config supplies store/key defaults; explicit flags win.
    if (
        getattr(args, "command", None) == "live"
        and getattr(args, "config", None) is not None
    ):
        from facecore.research.g3_config import load_g3_config as _load_cfg_main

        try:
            _cfg_main = _load_cfg_main(args.config)
        except ValueError as exc:
            print(f"research live: {exc}", file=sys.stderr)
            return 2
        if getattr(args, "store", None) is None:
            args.store = _cfg_main.store_dir
        if getattr(args, "key_dir", None) is None:
            args.key_dir = _cfg_main.key_dir
    if (
        getattr(args, "command", None) == "live"
        and getattr(args, "store", None) is None
    ):
        print(
            "research live: --store is required (or --config with store_dir)",
            file=sys.stderr,
        )
        return 2
    default_key_dir = (
        Path(args.store) / ".." / "research_keys"
        if getattr(args, "store", None) is not None
        else None
    )
    key_dir = (
        Path(args.key_dir)
        if getattr(args, "key_dir", None) is not None
        else (default_key_dir or Path("research_keys"))
    )
    if args.command == "live":
        return cmd_live(
            profile_path=args.profile,
            store=args.store,
            key_dir=key_dir,
            device=args.device,
            session_id=args.session,
            record_consent=args.record_consent,
            image_consent=args.image_consent,
            fixed_seconds=args.fixed_seconds,
            ui=args.ui,
            qt_offscreen=args.qt_offscreen,
            models=args.models,
            corpus=args.corpus,
            gallery_dir=args.gallery_dir,
            config=args.config,
            presence_mode=args.presence_mode,
            continuous=args.continuous,
            mode=args.mode,
        )
    if args.command == "replay":
        return cmd_replay(
            store=args.store,
            key_dir=key_dir,
            session_id=args.session,
            profile_path=args.profile,
            models=args.models,
            corpus=args.corpus,
        )
    if args.command == "delete":
        return cmd_delete(store=args.store, key_dir=key_dir, session_id=args.session)
    if args.command == "analyze":
        return cmd_analyze(
            store=args.store,
            key_dir=key_dir,
            experiment_id=args.experiment,
            mode=args.mode,
            profile_path=args.profile,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
