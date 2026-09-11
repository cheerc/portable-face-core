"""CLI surface: init / evaluate (Task 8). Exit codes per spec section 11."""

import argparse
import json
import sys
from pathlib import Path

from facecore import SCHEMA_VERSION
from facecore.errors import FaceCoreError
from facecore.eval.session import EvaluationSession


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="facecore")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--enrollment", required=True, type=Path)
    ev.add_argument("--probe", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "init":
        return cmd_init()
    if args.command == "evaluate":
        # cmd_evaluate maps FaceCoreError (incl. InputDecodeError) itself;
        # no dead except here (Task 5.5 cleanup).
        return cmd_evaluate(args.enrollment, args.probe)
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
