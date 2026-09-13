#!/usr/bin/env python3
"""Full 500-identity scale run for the CI `scale` job (carry-forward 3).

Synthetic stream only (no biometric data): 500 identities, 2500 vectors,
fetch+decrypt p95 vs 15.0 ms budget, comparison p95 vs 3.0 ms budget.
Prints the capacity report section always; exits 1 on budget overrun —
overruns are recorded, never fitted (no threshold tuning to pass).

Usage: python scripts/run_500_scale.py [--identities 500]
"""

import argparse
import sys
import tempfile

from facecore.eval.benchmark_1b import (
    BACKUP_CEILING,
    COMPARISON_BUDGET_MS,
    FETCH_BUDGET_MS,
    render_capacity_section,
    run_capacity_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identities", type=int, default=500)
    parser.add_argument("--vectors-per-identity", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="scale500-") as tmpdir:
        result = run_capacity_benchmark(
            identities=args.identities,
            vectors_per_identity=args.vectors_per_identity,
            db_path=f"{tmpdir}/scale.db",
            repeats=args.repeats,
            seed=args.seed,
        )
    print(render_capacity_section(result))
    overruns: list[str] = []
    comparison = float(result["comparison_p95_ms"])
    fetch = float(result["sqlite_fetch_p95_ms"])
    if comparison > COMPARISON_BUDGET_MS:
        overruns.append(
            f"comparison p95 {comparison:.3f} ms > budget "
            f"{COMPARISON_BUDGET_MS:.1f} ms"
        )
    if fetch > FETCH_BUDGET_MS:
        overruns.append(
            f"fetch+decrypt p95 {fetch:.3f} ms > budget {FETCH_BUDGET_MS:.1f} ms"
        )
    if overruns:
        print("BUDGET OVERRUN (recorded, not fitted):", file=sys.stderr)
        for line in overruns:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(
        f"within budgets "
        f"(comparison {COMPARISON_BUDGET_MS:.1f} ms, "
        f"fetch {FETCH_BUDGET_MS:.1f} ms, "
        f"backup ceiling {BACKUP_CEILING})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
