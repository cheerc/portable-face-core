"""CLI surface: init / evaluate (Task 8) / bakeoff (Task 9)."""

import argparse
import json
import sys
from pathlib import Path

from facecore import SCHEMA_VERSION
from facecore.errors import FaceCoreError
from facecore.eval.bakeoff import TEN_CONDITIONS_NOTE, run_candidate
from facecore.eval.corpus import CorpusFile, load_manifest
from facecore.eval.report import write_report
from facecore.eval.session import EvaluationSession
from facecore.eval.sweep import format_cell, sweep_thresholds


def _result_payload(status: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "identity": None,
        "decision": {
            "score": None,
            "runner_up_score": None,
            "threshold": None,
            "margin": None,
            "reason_codes": [],
        },
        "quality": {"status": "accepted", "reason_codes": []},
        "model_version": "unevaluated",
        "template_revision": 1,
        "candidate_created": False,
    }


def cmd_init() -> int:
    payload = {"schema_version": SCHEMA_VERSION, "status": "ready"}
    print("facecore init: session ready (in-memory, no biometric state)")
    print(json.dumps(payload))
    return 0


def cmd_evaluate(enrollment: Path, probe: Path) -> int:
    try:
        enrollment_bytes = enrollment.read_bytes()
        probe_bytes = probe.read_bytes()
    except OSError as exc:
        print(f"facecore evaluate: unreadable input: {exc}", file=sys.stderr)
        print(json.dumps(_result_payload("invalid_input")))
        return 2
    session = EvaluationSession()
    try:
        outcome = session.enroll(enrollment_bytes, "enrolled-1")
    except FaceCoreError as exc:
        print(json.dumps(_result_payload("invalid_input")))
        return exc.exit_code
    if outcome != "enrolled":
        print("facecore evaluate: enrollment invalid_input (no identity created)")
        print(json.dumps(_result_payload("invalid_input")))
        return 0
    try:
        probe_outcome = session.enroll(probe_bytes, "probe-1")
    except FaceCoreError as exc:
        print(json.dumps(_result_payload("invalid_input")))
        return exc.exit_code
    _ = probe_outcome
    print("facecore evaluate: probe accepted (no model scored in Phase 1A Task 8)")
    print(json.dumps(_result_payload("unknown")))
    return 0


SFACE_FP32_SHA = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
YUNET_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MATCH_GRID = [round(v, 2) for v in [x * 0.05 for x in range(0, 21)]]
MARGIN_GRID = [round(v, 2) for v in [x * 0.05 for x in range(0, 11)]]
REVIEW_THRESHOLD = 0.5


ProbeRow = tuple[str, str, float | None, float | None, str | None, str | None]


def render_per_probe_detail(
    rows: list[tuple[str, str, float | None, float | None]] | list[ProbeRow],
) -> list[str]:
    """Per-probe lines, self-contained (verification-grade, work order 2).

    6-tuple rows render `basename → status (top1 S, margin M) | expected E,
    predicted P` so a reader judges each pairing without rerunning. 4-tuple
    rows keep the legacy `basename → status (score, margin)` shape (PR #14).
    Basename only — absolute paths never enter the report (redaction guard
    red line). Identity names here are manifest-derived gallery ids
    (expected/predicted), never real names.
    """
    lines: list[str] = []
    for row in rows:
        if len(row) == 6:
            path, status, top1, runner_up, expected, predicted = row
            name = Path(path).name
            if top1 is None:
                detail = "n/a, n/a"
            else:
                base = runner_up if runner_up is not None else 0.0
                detail = f"top1 {top1:.4f}, margin {top1 - base:.4f}"
            exp = expected if expected is not None else "n/a"
            pred = predicted if predicted is not None else "n/a"
            lines.append(
                f"{name} → {status} ({detail}) | expected {exp}, predicted {pred}"
            )
            continue
        path, status, score, margin = row
        name = Path(path).name
        score_s = f"{score:.4f}" if score is not None else "n/a"
        margin_s = f"{margin:.4f}" if margin is not None else "n/a"
        lines.append(f"{name} → {status} ({score_s}, {margin_s})")
    return lines


def cmd_bakeoff(corpus: Path, models: Path, out: Path) -> int:
    from facecore.contracts.manifest import ModelManifest
    from facecore.contracts.policy import PolicyProfile
    from facecore.pipeline.embed import Embedder
    from facecore.pipeline.yunet import YuNetDetector

    repo_root = Path(__file__).resolve().parents[2]
    try:
        loaded = load_manifest(corpus, repo_root=repo_root)
    except FaceCoreError as exc:
        print(json.dumps(_result_payload("invalid_input")))
        return exc.exit_code
    enrollments = [f for f in loaded.files if f.role == "enrollment"]
    by_identity: dict[str, list[CorpusFile]] = {}
    for entry in enrollments:
        by_identity.setdefault(entry.identity, []).append(entry)
    for identity, entries in by_identity.items():
        if len(entries) != 1:
            print(
                f"bakeoff: {identity} has {len(entries)} enrollment photos, need 1",
                file=sys.stderr,
            )
            return 5
    detector = YuNetDetector(models / "face_detection_yunet_2023mar.onnx", YUNET_SHA)
    manifest = ModelManifest.sface_2021dec_fp32()
    embedder = Embedder(manifest, models / "face_recognition_sface_2021dec.onnx")
    policy = PolicyProfile.frozen_v1()
    session = EvaluationSession(detector=detector, embedder=embedder, policy=policy)
    enrollment_refused = 0
    for identity, entries in sorted(by_identity.items()):
        data = Path(entries[0].path).read_bytes()
        if session.enroll(data, identity) != "enrolled":
            enrollment_refused += 1
    probe_refused = 0
    gallery = session.gallery()
    model_version = session.model_version()
    target_files = [f for f in loaded.files if f.role == "target_probe"]
    nontarget_files = [f for f in loaded.files if f.role == "non_target_probe"]
    import numpy as np

    from facecore.pipeline.decode import decode_image

    def probe_vector(path: str) -> np.ndarray | None:
        raw = Path(path).read_bytes()
        try:
            decoded = decode_image(raw)
        except FaceCoreError:
            return None
        faces = detector.detect(decoded)
        if len(faces) != 1:
            return None
        from facecore.pipeline.align import align_crop

        crop = align_crop(decoded.pixels, decoded.width, decoded.height, faces[0])
        vector, _ = embedder.embed(crop)
        return vector

    # Display anchor point for per-identity/confusion sections ONLY (not a
    # selected operating point — the sweep table is the evidence). Uses the
    # upstream SFace wrapper cosine default 0.363 with margin 0.1.
    ANCHOR_MATCH, ANCHOR_MARGIN = 0.363, 0.1

    def band_of(score: float, margin: float | None) -> str:
        if score >= ANCHOR_MATCH and (margin is None or margin >= ANCHOR_MARGIN):
            return "matched"
        if score >= ANCHOR_MATCH or score >= REVIEW_THRESHOLD:
            return "review"
        return "unknown"

    target_scores: list[float] = []
    target_margins: list[float | None] = []
    target_vectors: list[np.ndarray | None] = [
        probe_vector(f.path) for f in target_files
    ]
    usable_targets = [v for v in target_vectors if v is not None]
    probe_refused += len(target_vectors) - len(usable_targets)
    nontarget_scores: list[float] = []
    for entry in nontarget_files:
        vector = probe_vector(entry.path)
        if vector is None:
            probe_refused += 1
            continue
        outcome = run_candidate(gallery, [vector], session._repository, model_version)[
            0
        ]
        nontarget_scores.append(
            outcome.top_score if outcome.top_score is not None else -1.0
        )
    per_identity: dict[str, list[str]] = {}
    per_probe: list[ProbeRow] = []
    for entry, vector in zip(target_files, target_vectors, strict=True):
        if vector is None:
            per_identity.setdefault(entry.identity, []).append("invalid_input")
            per_probe.append(
                (entry.path, "invalid_input", None, None, entry.identity, None)
            )
            continue
        outcome = run_candidate(gallery, [vector], session._repository, model_version)[
            0
        ]
        score = outcome.top_score if outcome.top_score is not None else -1.0
        target_scores.append(score)
        target_margins.append(outcome.margin)
        status = band_of(score, outcome.margin)
        per_identity.setdefault(entry.identity, []).append(status)
        per_probe.append(
            (
                entry.path,
                status,
                outcome.top_score,
                outcome.runner_up_score,
                entry.identity,
                outcome.top_identity,
            )
        )
    rows = sweep_thresholds(
        target_scores=target_scores,
        target_margins=target_margins,
        nontarget_scores=nontarget_scores,
        match_grid=MATCH_GRID,
        margin_grid=MARGIN_GRID,
        review_threshold=REVIEW_THRESHOLD,
    )
    lines = [
        "# Phase-1A bake-off operating table",
        "",
        f"candidate: {manifest.model_id} (provenance_unresolved)",
        f"gallery: {sorted(gallery)} | enrollment refused: {enrollment_refused} "
        f"| probe refused (no single face): {probe_refused}",
        f"target probes usable: {len(usable_targets)}/{len(target_files)}",
        f"non-target probes scored: {len(nontarget_scores)}/{len(nontarget_files)}",
        "",
        TEN_CONDITIONS_NOTE,
        "",
        "## Corpus inventory",
        "",
    ]
    for inv in loaded.inventory:
        lines.append(f"- {inv.condition}: {inv.samples if inv.samples else 'untested'}")
    lines += ["", "## Operating table (match x margin grid excerpts)", ""]
    for sweep in rows[::21]:
        lines.append(
            f"- match>={sweep.match_threshold:.2f} "
            f"margin>={sweep.margin_threshold:.2f}: "
            f"matched {format_cell(sweep.matched, sweep.target_denom)}, "
            f"review {format_cell(sweep.review, sweep.target_denom)}, "
            f"unknown {format_cell(sweep.unknown, sweep.target_denom)}, "
            f"FA {format_cell(sweep.false_accepts, sweep.nontarget_denom)}"
        )
    lines += [
        "",
        "## Per-probe detail "
        f"(display anchor match>={ANCHOR_MATCH} margin>={ANCHOR_MARGIN}; not selected)",
        "",
    ]
    lines += render_per_probe_detail(per_probe)
    lines += [
        "",
        "## Per-identity target outcomes "
        f"(display anchor match>={ANCHOR_MATCH} margin>={ANCHOR_MARGIN}; not selected)",
        "",
    ]
    for identity in sorted(per_identity):
        lines.append(f"- {identity}: {', '.join(per_identity[identity])}")
    lines += ["", "## Confusion rows (one per enrolled identity, aggregated)", ""]
    for identity in sorted(
        {f.identity for f in loaded.files if f.role == "enrollment"}
    ):
        statuses = per_identity.get(identity, [])
        lines.append(
            f"- {identity}: target probes={len(statuses)} "
            f"matched={statuses.count('matched')} "
            f"review={statuses.count('review')} "
            f"unknown={statuses.count('unknown')} "
            f"invalid_input={statuses.count('invalid_input')}"
        )
    body = "\n".join(lines) + "\n"
    write_report(out, body)
    print(f"bakeoff: wrote {out} ({len(rows)} sweep rows)")
    print(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "status": "ok", "sweep_rows": len(rows)}
        )
    )
    return 0


def cmd_conformance() -> int:
    from facecore.conformance.check import check_conformance_cli

    return check_conformance_cli()


def cmd_confirm_learning(verdict: str, score: float) -> int:
    """Handle `--confirm-learning`: validate verdict, never persist here.

    Task 3 boundary: the CLI admits only the closed verdict taxonomy and
    holds the observation in memory. Candidate persistence happens only
    through `CandidatePipeline.evaluate_observation` with an explicit
    `correct` confirmation; every other verdict exits 0 with zero disk
    mutation (spec §9 line 179).
    """
    from facecore.contracts.confirmation import ConfirmationVerdict

    try:
        parsed = ConfirmationVerdict(verdict)
    except ValueError:
        print(f"facecore confirm-learning: unknown verdict {verdict!r}")
        return 2
    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": parsed.value,
        "score": score,
        "candidate_created": False,
    }
    print(json.dumps(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facecore")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--enrollment", required=True, type=Path)
    ev.add_argument("--probe", required=True, type=Path)
    bo = sub.add_parser("bakeoff")
    bo.add_argument("--corpus", required=True, type=Path)
    bo.add_argument("--models", required=True, type=Path)
    bo.add_argument("--out", required=True, type=Path)
    sub.add_parser("conformance")
    cl = sub.add_parser("confirm-learning")
    cl.add_argument("--verdict", required=True)
    cl.add_argument("--score", required=True, type=float)
    args = parser.parse_args(argv)
    if args.command == "init":
        return cmd_init()
    if args.command == "evaluate":
        # cmd_evaluate maps FaceCoreError (incl. InputDecodeError) itself;
        # no dead except here (Task 5.5 cleanup).
        return cmd_evaluate(args.enrollment, args.probe)
    if args.command == "bakeoff":
        return cmd_bakeoff(args.corpus, args.models, args.out)
    if args.command == "conformance":
        return cmd_conformance()
    if args.command == "confirm-learning":
        return cmd_confirm_learning(args.verdict, args.score)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
