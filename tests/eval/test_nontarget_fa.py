"""M1 RED/GREEN: real non-target 30 FA evaluation + sweep inside bakeoff.

SSOT: /tmp/face-accept/fa_real.log @ gate 0.8 (YuNet/SFace + real photos,
repo-external, read-only; operator-provided corpus
/Users/cheerc/Downloads/face_sample/non-target/nontarget-01..30.jpg).
Tests reconstruct the SSOT score/margin pairs with SYNTHETIC vectors only —
no biometric data enters the repo.

FA rule (F1): match >= 0.45 AND margin >= 0.1 -> 4/30 true FA
(09->person-11, 20->person-18, 21->person-22, 25->person-12).
High-risk thin-margin pin: 12->person-05 @0.8199/margin 0.0789.
Sweep (F2): 0.30-0.50 -> 3/13 | 4/30; 0.55 -> 3/13 | 3/30;
0.60 -> 2/13 | 1/30 (target denom is the 13-probe target arm).
"""

import math

import numpy as np
import pytest

from facecore.eval.nontarget_fa import (
    FA_MATCH,
    MARGIN_FIREWALL,
    REAL_NONTARGET_COUNT,
    RealFaRow,
    RealFaSweepRow,
    load_real_nontarget_vectors,
    real_fa_rows,
    real_fa_sweep_table,
    render_real_fa_section,
    render_real_fa_skipped,
    stable_target_scores,
)

_ALPHA = 0.3
_BETA = math.sqrt(1.0 - _ALPHA**2)

# (filename, top1, top1_score, margin) — fa_real.log F1, verbatim (4 THICK).
REAL_SSOT = [
    ("nontarget-09.jpg", "person-11", 0.7346, 0.1439),
    ("nontarget-20.jpg", "person-18", 0.5486, 0.1094),
    ("nontarget-21.jpg", "person-22", 0.5744, 0.1127),
    ("nontarget-25.jpg", "person-12", 0.5728, 0.1461),
]
RISK_SSOT = ("nontarget-12.jpg", "person-05", 0.8199, 0.0789)

# (match_threshold, target_hits/13, real_fa/30) — fa_real.log F2.
SWEEP_SSOT = [
    (0.30, 3, 4),
    (0.35, 3, 4),
    (0.40, 3, 4),
    (0.45, 3, 4),
    (0.50, 3, 4),
    (0.55, 3, 3),
    (0.60, 2, 1),
]


def _members() -> list[str]:
    return [f"person-{i:02d}" for i in range(1, 24)]


def _harness(identities: list[str], dim: int):
    from facecore.contracts.template import FaceTemplate, TemplateRevision
    from facecore.repository.memory import InMemoryRepository

    repo = InMemoryRepository()
    for identity in identities:
        repo.create_identity(
            identity,
            identity,
            FaceTemplate(
                template_id=f"t-{identity}",
                identity_id=identity,
                model_version="m",
                embedding_dim=dim,
                revision=TemplateRevision(
                    revision=1, template_id=f"t-{identity}", supersedes=None
                ),
            ),
        )
    return repo


_RESIDUAL_DIMS = 8


def _synthetic_gallery(
    members: list[str], top1: str, runner_up: str, s1: float, s2: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    others = [m for m in members]
    base = len(others) + 1
    dim = base + _RESIDUAL_DIMS
    gallery: dict[str, np.ndarray] = {}
    for pos, ident in enumerate(others):
        vec = np.zeros(dim)
        if ident == top1:
            vec[0] = 1.0
        elif ident == runner_up:
            vec[0] = _ALPHA
            vec[1] = _BETA
        else:
            vec[pos + 1] = 1.0
        gallery[ident] = vec
    b = (s2 - s1 * _ALPHA) / _BETA
    rest = 1.0 - s1**2 - b**2
    assert rest > 0.0, f"infeasible SSOT pair s1={s1} s2={s2}"
    probe = np.zeros(dim)
    probe[0] = s1
    probe[1] = b
    probe[base:] = math.sqrt(rest / _RESIDUAL_DIMS)
    return gallery, probe


def _rows_for(ssot_rows) -> list[RealFaRow]:
    """Score each SSOT pair against its own synthetic gallery (cf. §6-2)."""
    from facecore.eval.nontarget_fa import real_fa_rows

    members = _members()
    rows: list[RealFaRow] = []
    for name, top1, s1, margin in ssot_rows:
        runner_up = next(m for m in members if m != top1)
        gallery, probe = _synthetic_gallery(members, top1, runner_up, s1, s1 - margin)
        repo = _harness(members, len(probe))
        rows.extend(real_fa_rows(gallery, [probe], repo, "m", names=[name]))
    return rows


def test_constants_pin_protocol() -> None:
    assert FA_MATCH == 0.45
    assert MARGIN_FIREWALL == 0.1
    assert REAL_NONTARGET_COUNT == 30


def test_real_fa_rows_reproduce_ssot() -> None:
    rows = _rows_for([*REAL_SSOT, RISK_SSOT])
    assert len(rows) == 5
    pairs = [*REAL_SSOT, RISK_SSOT]
    for row, (name, top1, s1, margin) in zip(rows, pairs, strict=True):
        assert row.probe_name == name
        assert row.top_identity == top1, name
        assert round(float(row.top_score), 4) == pytest.approx(s1, abs=1e-4), name
        assert round(float(row.margin), 4) == pytest.approx(margin, abs=1e-4), name


def test_real_fa_rule_counts_4_of_5_at_anchor() -> None:
    """Acceptance: at FA_MATCH/MARGIN_FIREWALL exactly the 4 THICK admit."""
    rows = _rows_for([*REAL_SSOT, RISK_SSOT])
    admitted = [r for r in rows if r.is_false_accept]
    assert [r.probe_name for r in admitted] == [
        "nontarget-09.jpg",
        "nontarget-20.jpg",
        "nontarget-21.jpg",
        "nontarget-25.jpg",
    ]
    risk = next(r for r in rows if r.probe_name == "nontarget-12.jpg")
    assert not risk.is_false_accept  # thin margin 0.0789 < 0.1 firewall


def test_regression_real_fa_pins() -> None:
    """Named pins: 4 true FA + 1 thin-margin high-risk (tolerance noted)."""
    rows = {r.probe_name: r for r in _rows_for([*REAL_SSOT, RISK_SSOT])}
    assert rows["nontarget-09.jpg"].top_identity == "person-11"
    assert round(float(rows["nontarget-09.jpg"].top_score), 2) == 0.73
    assert round(float(rows["nontarget-09.jpg"].margin), 2) == 0.14
    assert rows["nontarget-20.jpg"].top_identity == "person-18"
    assert round(float(rows["nontarget-20.jpg"].top_score), 2) == 0.55
    assert rows["nontarget-21.jpg"].top_identity == "person-22"
    assert round(float(rows["nontarget-21.jpg"].top_score), 2) == 0.57
    assert rows["nontarget-25.jpg"].top_identity == "person-12"
    assert round(float(rows["nontarget-25.jpg"].top_score), 2) == 0.57
    assert rows["nontarget-12.jpg"].top_identity == "person-05"
    assert round(float(rows["nontarget-12.jpg"].top_score), 2) == 0.82
    assert round(float(rows["nontarget-12.jpg"].margin), 2) == 0.08


def test_real_fa_sweep_reproduces_ssot() -> None:
    from facecore.eval.fa_matrix import TargetProbeScore

    rows = _rows_for([*REAL_SSOT, RISK_SSOT])
    # 25 silent non-FA probes filling N=30 exactly (score 0.3, thin margin).
    members = _members()
    extra: list[RealFaRow] = []
    for i in range(25):
        runner_up = next(m for m in members if m != "person-01")
        gal, probe = _synthetic_gallery(members, "person-01", runner_up, 0.30, 0.29)
        repo_e = _harness(members, len(probe))
        extra.extend(
            real_fa_rows(gal, [probe], repo_e, "m", names=[f"quiet-{i:02d}.jpg"])
        )
    full = rows + extra
    assert len(full) == 30
    # Target arm: 3 stable hits + 10 misses (SSOT 3/13 @ 0.30-0.55,
    # 2/13 @ 0.60 — third hit scores between 0.55 and 0.60).
    target = [
        TargetProbeScore(
            probe_index=i,
            person23_score=0.65 if i < 2 else (0.57 if i == 2 else 0.20),
            top_is_person23=i < 3,
            margin=0.15 if i < 3 else None,
        )
        for i in range(13)
    ]
    sweep = real_fa_sweep_table(
        target=target,
        real_rows=full,
        match_grid=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    )
    assert len(sweep) == len(SWEEP_SSOT)
    for row, (mt, hits, fa) in zip(sweep, SWEEP_SSOT, strict=True):
        assert row.match_threshold == pytest.approx(mt)
        assert row.target_hits == hits, mt
        assert row.real_fa == fa, mt
        assert row.target_denom == 13
        assert row.real_denom == 30


def test_missing_corpus_dir_skips_fail_clear() -> None:
    from pathlib import Path

    outcome = load_real_nontarget_vectors(Path("/nonexistent/nontarget-xyz"))
    assert outcome.skipped is True
    assert outcome.reason != ""
    assert outcome.files == []


def test_real_fa_rows_require_names() -> None:
    members = _members()
    name, top1, s1, margin = REAL_SSOT[0]
    runner_up = next(m for m in members if m != top1)
    gallery, probe = _synthetic_gallery(members, top1, runner_up, s1, s1 - margin)
    repo = _harness(members, len(probe))
    with pytest.raises(ValueError):
        real_fa_rows(gallery, [probe], repo, "m", names=["only-one", "extra"])


def test_render_real_fa_section_exact_denominators() -> None:
    section = render_real_fa_section(
        rows=[
            RealFaRow(
                probe_name="nontarget-09.jpg",
                top_identity="person-11",
                top_score=0.7346,
                margin=0.1439,
                is_false_accept=True,
            )
        ],
        sweep_rows=[
            RealFaSweepRow(
                match_threshold=0.30,
                target_hits=3,
                target_denom=13,
                real_fa=4,
                real_denom=30,
            )
        ],
        usable=30,
        total=30,
    )
    assert "N=30" in section
    assert "| nontarget-09.jpg | person-11 | 0.7346 | 0.1439 |" in section
    # Exact denominators written out; no bare rate without counts.
    assert "4/30" in section
    assert "3/13" in section


def test_stable_target_scores_from_probe_outcomes() -> None:
    """CLI wiring: target-arm scores derived from scored probe outcomes."""
    from facecore.eval.nontarget_fa import ProbeTargetScore

    assert stable_target_scores([]) == []
    scored = [
        ProbeTargetScore(
            person23_score=0.65, top_is_person23=True, margin=0.15
        ),
        ProbeTargetScore(
            person23_score=0.20, top_is_person23=False, margin=None
        ),
    ]
    target = stable_target_scores(scored)
    assert len(target) == 2
    assert target[0].top_is_person23 is True
    assert target[0].person23_score == 0.65
    assert target[1].margin is None


def test_render_real_fa_skipped_names_corpus() -> None:
    section = render_real_fa_skipped("corpus dir missing: /x (skipped)")
    assert "skipped" in section
    assert "/x" in section
    assert "N=30" in section


def test_skip_reason_carries_no_absolute_path() -> None:
    """Report redaction guard: skip reasons must stay path-free (Task 10)."""
    import re

    from pathlib import Path

    outcome = load_real_nontarget_vectors(Path("/nonexistent/nontarget-xyz"))
    assert outcome.skipped is True
    assert not re.search(r"(?:/Users/|/tmp/|/private/)\\S+", outcome.reason)
    assert not re.search(
        r"(?:/Users/|/tmp/|/private/)\\S+",
        render_real_fa_skipped(outcome.reason),
    )


def test_real_fa_grid_margin_010_matches_r2_sweep() -> None:
    """M5 regression guard: grid row at margin 0.10 == R2 sweep, all 7."""
    from facecore.eval.fa_matrix import TargetProbeScore
    from facecore.eval.nontarget_fa import (
        REAL_MARGIN_GRID,
        REAL_MATCH_GRID,
        real_fa_grid_table,
        real_fa_sweep_table,
    )

    assert 0.10 in REAL_MARGIN_GRID
    rows = _rows_for([*REAL_SSOT, RISK_SSOT])
    members = _members()
    extra: list[RealFaRow] = []
    for i in range(25):
        runner_up = next(m for m in members if m != "person-01")
        gal, probe = _synthetic_gallery(members, "person-01", runner_up, 0.30, 0.29)
        repo_e = _harness(members, len(probe))
        extra.extend(
            real_fa_rows(gal, [probe], repo_e, "m", names=[f"quiet-{i:02d}.jpg"])
        )
    full = rows + extra
    target = [
        TargetProbeScore(
            probe_index=i,
            person23_score=0.65 if i < 2 else (0.57 if i == 2 else 0.20),
            top_is_person23=i < 3,
            margin=0.15 if i < 3 else None,
        )
        for i in range(13)
    ]
    sweep = real_fa_sweep_table(
        target=target, real_rows=full, match_grid=REAL_MATCH_GRID
    )
    grid = real_fa_grid_table(
        target=target,
        real_rows=full,
        match_grid=REAL_MATCH_GRID,
        margin_grid=REAL_MARGIN_GRID,
    )
    assert len(grid) == len(REAL_MATCH_GRID) * len(REAL_MARGIN_GRID)
    guard = [g for g in grid if g.margin_threshold == 0.10]
    assert len(guard) == len(SWEEP_SSOT)
    for grow, (mt, hits, fa) in zip(guard, SWEEP_SSOT, strict=True):
        assert grow.match_threshold == mt
        assert (grow.target_hits, grow.real_fa) == (hits, fa), mt
        srow = next(s for s in sweep if s.match_threshold == mt)
        assert (grow.target_hits, grow.real_fa) == (
            srow.target_hits,
            srow.real_fa,
        ), mt
        assert grow.target_denom == 13
        assert grow.real_denom == 30


def test_real_fa_grid_none_margin_never_counts() -> None:
    """Boundary: None margin on either arm never counts at any grid cell."""
    from facecore.eval.fa_matrix import TargetProbeScore
    from facecore.eval.nontarget_fa import (
        REAL_MARGIN_GRID,
        REAL_MATCH_GRID,
        RealFaRow,
        real_fa_grid_table,
    )

    target = [
        TargetProbeScore(
            probe_index=0,
            person23_score=0.99,
            top_is_person23=True,
            margin=None,
        )
    ]
    real = [
        RealFaRow(
            probe_name="x.jpg",
            top_identity="person-01",
            top_score=0.99,
            margin=None,
            is_false_accept=False,
        )
    ]
    grid = real_fa_grid_table(
        target=target,
        real_rows=real,
        match_grid=REAL_MATCH_GRID,
        margin_grid=REAL_MARGIN_GRID,
    )
    assert all(g.target_hits == 0 and g.real_fa == 0 for g in grid)


def test_render_real_fa_grid_section_skeleton_shape() -> None:
    """R3 renders skeleton §1.1 line shape with exact denominators."""
    from facecore.eval.nontarget_fa import (
        RealFaGridRow,
        render_real_fa_grid_section,
    )

    section = render_real_fa_grid_section(
        grid_rows=[
            RealFaGridRow(
                match_threshold=0.30,
                margin_threshold=0.10,
                target_hits=3,
                target_denom=13,
                real_fa=4,
                real_denom=30,
            )
        ],
    )
    assert "R3" in section
    assert "match>=0.30 margin>=0.10" in section
    assert "3/13" in section
    assert "4/30" in section
