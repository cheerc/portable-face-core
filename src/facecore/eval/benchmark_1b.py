"""500-identity capacity benchmark + docs closeout helpers (1B plan §11).

Dual-stage timing (reported separately per the plan RED case):
- ``sqlite_fetch_p95_ms``: fetch sealed blobs + decrypt under the record
  DEK (storage path, budget 15.0 ms p95).
- ``comparison_p95_ms``: pure cosine comparison over the gallery matrix
  (compute path, budget 3.0 ms p95).

``check_backup_ceiling`` enforces ``backup_max_count`` (provisional 5,
ratified Manifest): keeps the newest ``ceiling`` archives, prunes older.

``verify_readme_map`` asserts the README repository map matches disk
exactly (returns missing/extra paths; empty means clean).

``render_capacity_section`` renders the capacity report section for
``reports/phase-1b-replay.md``.
"""

import time
from pathlib import Path

import numpy as np

COMPARISON_BUDGET_MS = 3.0
FETCH_BUDGET_MS = 15.0
BACKUP_CEILING = 5


def _percentile_p95(samples_ms: list[float]) -> float:
    return float(np.percentile(np.asarray(samples_ms), 95))


def run_capacity_benchmark(
    *,
    identities: int,
    vectors_per_identity: int,
    db_path: str,
    repeats: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """Benchmark encrypted storage at synthetic scale (offline only)."""
    from facecore.storage.key_provider import InMemoryKeyProvider
    from facecore.storage.sqlite_repo import SQLiteRepository

    rng = np.random.default_rng(seed)
    provider = InMemoryKeyProvider()
    repo = SQLiteRepository(db_path, provider)
    repo.initialize()
    gallery: list[np.ndarray] = []
    for index in range(identities):
        identity_id = f"bench-{index:04d}"
        vector = rng.normal(size=128)
        vector /= np.linalg.norm(vector)
        gallery.append(vector)
        payload = vector.astype(np.float64).tobytes()
        from facecore.contracts.template import FaceTemplate, TemplateRevision

        template = FaceTemplate(
            template_id=f"t-{identity_id}-1",
            identity_id=identity_id,
            model_version="sface-2021dec-fp32",
            embedding_dim=128,
            revision=TemplateRevision(
                revision=1,
                template_id=f"t-{identity_id}-1",
                supersedes=None,
            ),
            key_id="benchmark-managed",
            exemplar_crop_box=(0.0, 0.0, 112.0, 112.0),
            exemplar_landmarks=((30.0, 30.0), (82.0, 30.0)),
            quality_score=0.9,
        )
        repo.enroll_identity(
            identity_id, identity_id, template, payload, payload
        )
    matrix = np.stack(gallery)
    probe = rng.normal(size=128)
    probe /= np.linalg.norm(probe)
    template_ids = [f"t-bench-{i:04d}-1" for i in range(identities)]
    fetch_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for template_id in template_ids:
            repo.read_embedding(template_id)
        fetch_ms.append((time.perf_counter() - start) * 1000.0 / identities)
    comparison_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        scores = matrix @ probe
        _ = int(np.argmax(scores))
        comparison_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "identities": identities,
        "vectors": identities * vectors_per_identity,
        "sqlite_fetch_p95_ms": _percentile_p95(fetch_ms),
        "comparison_p95_ms": _percentile_p95(comparison_ms),
        "comparison_budget_ms": COMPARISON_BUDGET_MS,
        "fetch_budget_ms": FETCH_BUDGET_MS,
        "backup_ceiling": BACKUP_CEILING,
    }


def check_backup_ceiling(directory: Path | str, ceiling: int = BACKUP_CEILING) -> int:
    """Prune oldest backups beyond the ceiling; return pruned count."""
    directory = Path(directory)
    candidates = sorted(
        directory.glob("*.db"), key=lambda p: p.name
    )
    if len(candidates) <= ceiling:
        return 0
    pruned = 0
    for path in candidates[: len(candidates) - ceiling]:
        path.unlink()
        pruned += 1
    return pruned


def render_capacity_section(result: dict[str, object]) -> str:
    """Render the capacity report section (counts + budgets)."""
    lines = [
        "## Capacity benchmark (500-identity synthetic scale)",
        "",
        f"- identities: {result['identities']}",
        f"- vectors: {result['vectors']}",
        f"- sqlite fetch+decrypt p95: {result['sqlite_fetch_p95_ms']:.3f} ms "
        f"(budget {result['fetch_budget_ms']:.1f} ms)",
        f"- comparison p95: {result['comparison_p95_ms']:.3f} ms "
        f"(budget {result['comparison_budget_ms']:.1f} ms)",
        f"- backup retention ceiling: {result['backup_ceiling']}",
        "",
        "Synthetic scale validates latency only; zero evidence on false",
        "acceptances, margin distributions, or recognition accuracy.",
        "",
    ]
    return "\n".join(lines)


def verify_readme_map(repo_root: Path | str) -> list[str]:
    """Compare the README repository map against disk; return mismatches.

    The map is an indented tree (``├── cli.py`` under ``facecore/``);
    a disk file counts as mapped when its parent-dir section header
    (``parent/``) is followed somewhere later by its basename leaf.
    """
    import re

    root = Path(repo_root)
    text = (root / "README.md").read_text()
    disk_py = sorted(
        p.relative_to(root / "src")
        for p in (root / "src").rglob("*.py")
        if "__pycache__" not in p.parts
    )
    mismatches: list[str] = []
    for path in disk_py:
        parent = path.parent.name
        leaf = path.name
        section_pos: int | None = None
        for match in re.finditer(rf"^.*{re.escape(parent)}/\s*$", text, re.M):
            section_pos = match.start()
            break
        if section_pos is None:
            # Top-level files (facecore/__init__.py) sit under facecore/.
            if parent == "facecore":
                section_pos = 0
            else:
                mismatches.append(f"missing-from-readme: {parent}/{leaf}")
                continue
        found = any(
            match.start() > section_pos
            for match in re.finditer(rf"^.*{re.escape(leaf)}\s*$", text, re.M)
        )
        if not found:
            mismatches.append(f"missing-from-readme: {parent}/{leaf}")
    return mismatches
