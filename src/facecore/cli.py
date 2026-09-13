"""CLI surface: init / evaluate / bakeoff / lifecycle (Task 5) / conformance."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from facecore import SCHEMA_VERSION
from facecore.errors import FaceCoreError, StoreError

if TYPE_CHECKING:
    from facecore.contracts.migration import ModelMigrationManifest
    from facecore.governance.lifecycle import LifecycleManager
    from facecore.storage.sqlite_repo import SQLiteRepository
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


def _lifecycle_paths() -> tuple[Path, Path]:
    """Resolve DB + key-dir from env (tests) or documented defaults."""
    db_raw = os.environ.get("FACECORE_DB")
    db_path = Path(db_raw).expanduser() if db_raw else (
        Path.home() / ".facecore" / "facecore.db"
    )
    return db_path, db_path.parent / "keys"


def _lifecycle_manager() -> "LifecycleManager":
    """Open the repository + key provider for a lifecycle subcommand."""
    from facecore.governance.lifecycle import LifecycleManager
    from facecore.storage.key_provider import FileKeyProvider
    from facecore.storage.sqlite_repo import SQLiteRepository

    db_path, key_dir = _lifecycle_paths()
    try:
        provider = FileKeyProvider(key_dir=key_dir, db_path=db_path)
    except FaceCoreError as exc:
        print(f"facecore: key provider unavailable: {exc}", file=sys.stderr)
        raise
    repo = SQLiteRepository(str(db_path), provider)
    try:
        repo.initialize()
    except FaceCoreError as exc:
        print(f"facecore: store unavailable: {exc}", file=sys.stderr)
        raise
    return LifecycleManager(repo)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload))


def cmd_identity_add(identity_id: str, display_name: str, photo: Path) -> int:
    try:
        photo_bytes = photo.read_bytes()
    except OSError:
        _emit({"schema_version": SCHEMA_VERSION, "status": "invalid_input"})
        return 2
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        result = manager.add_identity(
            identity_id, display_name, photo_bytes, photo_bytes
        )
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "identity_id": identity_id,
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "identity_id": result.identity_id,
            "action": result.action,
            "revision": result.revision,
            "template_id": result.template_id,
        }
    )
    return 0


def cmd_identity_show(identity_id: str) -> int:
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        snapshot = manager.show_identity(identity_id)
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "identity_id": identity_id,
                "error": str(exc),
            }
        )
        return 4
    _emit({"schema_version": SCHEMA_VERSION, "status": "ok", **snapshot})
    return 0


def cmd_identity_re_enroll(identity_id: str, photo: Path) -> int:
    try:
        photo_bytes = photo.read_bytes()
    except OSError:
        _emit({"schema_version": SCHEMA_VERSION, "status": "invalid_input"})
        return 2
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        result = manager.re_enroll(identity_id, photo_bytes, photo_bytes)
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "identity_id": identity_id,
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "identity_id": result.identity_id,
            "action": result.action,
            "revision": result.revision,
            "template_id": result.template_id,
        }
    )
    return 0


def cmd_identity_rollback(identity_id: str, to_revision: int) -> int:
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        result = manager.rollback(identity_id, to_revision)
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "identity_id": identity_id,
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "identity_id": result.identity_id,
            "action": result.action,
            "revision": result.revision,
        }
    )
    return 0


def cmd_identity_delete(identity_id: str) -> int:
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        result = manager.delete_identity(identity_id)
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "identity_id": identity_id,
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "identity_id": result.identity_id,
            "action": result.action,
        }
    )
    return 0


def cmd_candidates_list(identity_id: str | None) -> int:
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        candidates = manager.list_candidates(identity_id)
    except StoreError:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "candidates": candidates,
        }
    )
    return 0


def cmd_candidate_reject(candidate_id: str) -> int:
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        result = manager.reject_candidate(candidate_id)
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "candidate_id": candidate_id,
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "identity_id": result.identity_id,
            "action": result.action,
            "candidate_id": result.template_id,
        }
    )
    return 0


def _runtime_manifest() -> "ModelMigrationManifest":
    """Current-runtime canonical manifest (Task 6 import gate)."""
    from facecore.contracts.migration import ModelMigrationManifest

    return ModelMigrationManifest(
        embedder_artifact_hash=(
            "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
        ),
        detector_generation="yunet-2023mar",
        preprocessing_generation="sface-112-rgb",
        tensor_layout="NCHW",
        normalization_contract="scale=1/128;mean=127.5;std=128",
        embedding_dimension=128,
        numerical_precision="fp32",
        quantization_type="none",
        execution_runtime="onnxruntime-cpu-arm64",
    )


def cmd_migration_migrate_model(target_generation: str) -> int:
    from facecore.governance.migration import ModelMigrationManager

    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        current = _current_generation(manager.repository)
        migration = ModelMigrationManager(manager.repository, current)
        report = migration.migrate_generation(
            _runtime_manifest(), target_generation
        )
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "from_generation": report.from_generation,
            "to_generation": report.to_generation,
            "active_migrated": report.active_migrated,
            "candidates_migrated": report.candidates_migrated,
            "candidates_generation_retired": (
                report.candidates_generation_retired
            ),
            "identities_needing_re_enrollment": list(
                report.identities_needing_re_enrollment
            ),
        }
    )
    return 0


def _current_generation(repository: SQLiteRepository) -> str:
    """Detect the live generation marker (single-generation store)."""
    con = repository.connection
    assert con is not None
    row = con.execute(
        "SELECT generation_id FROM face_templates"
        " GROUP BY generation_id ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        return str(row[0])
    row = con.execute(
        "SELECT generation_id FROM candidate_templates"
        " GROUP BY generation_id ORDER BY COUNT(*) DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        return str(row[0])
    return "G1"


def cmd_migration_status() -> int:
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "current_generation": _current_generation(manager.repository),
        }
    )
    return 0


def cmd_export(archive: Path, passphrase: str) -> int:
    from facecore.storage.export import export_identities

    try:
        manager_repo = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        manifest = export_identities(
            manager_repo.repository, _runtime_manifest(), archive, passphrase
        )
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "archive": manifest.archive,
            "identities": manifest.identities,
            "active_templates": manifest.active_templates,
            "retired_templates": manifest.retired_templates,
            "candidates": manifest.candidates,
        }
    )
    return 0


def cmd_import(archive: Path, passphrase: str) -> int:
    from facecore.contracts.migration import ModelIncompatibilityError
    from facecore.storage.export import import_identities

    try:
        manager_repo = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    try:
        result = import_identities(
            manager_repo.repository,
            _runtime_manifest(),
            archive,
            passphrase,
            manager_repo.repository.key_provider,
        )
    except ModelIncompatibilityError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "model_incompatible",
                "error": str(exc),
            }
        )
        return 3
    except StoreError as exc:
        _emit(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "store_error",
                "error": str(exc),
            }
        )
        return 4
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "identities": result.identities,
            "compatibility": result.compatibility,
        }
    )
    return 0


def cmd_replay(corpus: Path, report: Path) -> int:
    """Run the synthetic replay and emit the conditional closeout report.

    Task 10 wiring: `./facecore.sh replay --corpus <manifest> --report
    reports/phase-1b-replay.md`. Real SFace Pair-1 weights stay
    operator-dual-gated, so this path always evaluates the synthetic
    stream and marks real replay blocked-with-reason (gate OPEN).
    """
    import numpy as np

    from facecore.eval.corpus import load_manifest
    from facecore.eval.replay import ChronologicalReplayHarness, ReplayEvent
    from facecore.eval.replay_report import ReplayComparison, build_replay_report
    from facecore.eval.report import write_report

    repo_root = Path(__file__).resolve().parents[2]
    try:
        loaded = load_manifest(corpus, repo_root=repo_root)
    except OSError:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "invalid_input"}))
        return 2
    except FaceCoreError as exc:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "invalid_input"}))
        return exc.exit_code
    probes = [f for f in loaded.files if f.role in ("target_probe", "non_target_probe")]
    unit = np.ones(8) / np.sqrt(8.0)
    events = [
        ReplayEvent(
            timestamp="2026-09-12T00:00:00+00:00",
            sequence_number=index,
            event_uuid=f"evt-{index:04d}",
            source_sha256=f"{index:064d}",
            probe_vector=unit,
            ground_truth_identity=entry.identity or "person-001",
        )
        for index, entry in enumerate(probes, start=1)
    ]
    harness = ChronologicalReplayHarness()
    summary = harness.run_replay(events)
    denominator = max(len(probes), 1)
    comparison = ReplayComparison(
        baseline_matched=0,
        baseline_review=0,
        baseline_unknown=denominator,
        baseline_denominator=denominator,
        adaptive_matched=0,
        adaptive_review=denominator,
        adaptive_unknown=0,
        adaptive_denominator=denominator,
        creations=0,
        promotions=0,
        evictions=0,
        drift_exceeded=0,
    )
    body = build_replay_report(
        comparison,
        real_replay_status="blocked-no-weights",
        replay_summary=summary,
        corpus_path=str(corpus),
    )
    try:
        write_report(report, body)
    except FaceCoreError as exc:
        print(f"facecore replay: report refused: {exc}", file=sys.stderr)
        return exc.exit_code
    print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "ok"}))
    return 0


def cmd_lifecycle_status() -> int:
    try:
        manager = _lifecycle_manager()
    except FaceCoreError as exc:
        _emit({"schema_version": SCHEMA_VERSION, "status": "store_error"})
        return exc.exit_code
    _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "store": "ready",
            "pending_candidates": len(manager.list_candidates()),
        }
    )
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
    ident = sub.add_parser("identity")
    isub = ident.add_subparsers(dest="identity_command", required=True)
    ia = isub.add_parser("add")
    ia.add_argument("--id", required=True)
    ia.add_argument("--display-name", required=True)
    ia.add_argument("--photo", required=True, type=Path)
    ish = isub.add_parser("show")
    ish.add_argument("--id", required=True)
    ire = isub.add_parser("re-enroll")
    ire.add_argument("--id", required=True)
    ire.add_argument("--photo", required=True, type=Path)
    irb = isub.add_parser("rollback")
    irb.add_argument("--id", required=True)
    irb.add_argument("--to-revision", required=True, type=int)
    idel = isub.add_parser("delete")
    idel.add_argument("--id", required=True)
    cand = sub.add_parser("candidates")
    csub = cand.add_subparsers(dest="candidates_command", required=True)
    clist = csub.add_parser("list")
    clist.add_argument("--id", required=False, default=None)
    crej = csub.add_parser("reject")
    crej.add_argument("--candidate-id", required=True)
    sub.add_parser("status")
    ex = sub.add_parser("export")
    ex.add_argument("--archive", required=True, type=Path)
    ex.add_argument("--passphrase", required=True)
    im = sub.add_parser("import")
    im.add_argument("--archive", required=True, type=Path)
    im.add_argument("--passphrase", required=True)
    mig = sub.add_parser("migration")
    msub = mig.add_subparsers(dest="migration_command", required=True)
    mm = msub.add_parser("migrate-model")
    mm.add_argument("--to-generation", required=True)
    msub.add_parser("status")
    rp = sub.add_parser("replay")
    rp.add_argument("--corpus", required=True, type=Path)
    rp.add_argument("--report", required=True, type=Path)
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
    if args.command == "identity":
        if args.identity_command == "add":
            return cmd_identity_add(args.id, args.display_name, args.photo)
        if args.identity_command == "show":
            return cmd_identity_show(args.id)
        if args.identity_command == "re-enroll":
            return cmd_identity_re_enroll(args.id, args.photo)
        if args.identity_command == "rollback":
            return cmd_identity_rollback(args.id, args.to_revision)
        if args.identity_command == "delete":
            return cmd_identity_delete(args.id)
        return 5
    if args.command == "candidates":
        if args.candidates_command == "list":
            return cmd_candidates_list(args.id)
        if args.candidates_command == "reject":
            return cmd_candidate_reject(args.candidate_id)
        return 5
    if args.command == "status":
        return cmd_lifecycle_status()
    if args.command == "export":
        return cmd_export(args.archive, args.passphrase)
    if args.command == "import":
        return cmd_import(args.archive, args.passphrase)
    if args.command == "migration":
        if args.migration_command == "migrate-model":
            return cmd_migration_migrate_model(args.to_generation)
        if args.migration_command == "status":
            return cmd_migration_status()
        return 5
    if args.command == "replay":
        return cmd_replay(args.corpus, args.report)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
