"""Task 10 RED/GREEN: exact denominators, 0/N never a rate, <30 counts-only."""

from facecore.eval.sweep import format_cell, sweep_thresholds


def test_zero_failures_renders_counts_not_rate() -> None:
    """Failing case from the plan: denominator 4 must render 0/4, not 0.0%."""
    assert format_cell(0, 4) == "0/4"


def test_below_30_shows_counts_only() -> None:
    assert format_cell(3, 6) == "3/6"
    assert format_cell(29, 29) == "29/29"


def test_at_30_rate_may_appear_with_denominator() -> None:
    cell = format_cell(3, 30)
    assert "3/30" in cell


def test_sweep_rows_carry_exact_denominators() -> None:
    rows = sweep_thresholds(
        target_scores=[0.9, 0.2],
        target_margins=[0.3, 0.0],
        nontarget_scores=[0.95, 0.1],
        match_grid=[0.76],
        margin_grid=[0.1],
        review_threshold=0.5,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.target_denom == 2
    assert row.nontarget_denom == 2
    assert row.matched + row.review + row.unknown == 2
