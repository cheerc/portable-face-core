"""D4: the SOP's round-counting formula must point at the right column.

Source of truth: decision d-20260929235132588526-14 item 2 — the operator
must report 「已執行輪次」 and 「已標註輪次」 as two separate numbers,
with `unlabeled` rounds counted in the first and not the second.

S1 (reviewer-found): the SOP said `label_kind` was column 20 and wrote
the formula against `R:R`. `label_kind` is column 22 (V); column 18 (R)
is `margin`, a float. `margin` never equals the string "unlabeled", so
the condition is always true and **「已標註輪次」 collapses into
「已執行輪次」** — the two numbers the decision asks to distinguish
become indistinguishable, silently.

**Why this needs a test at all.** The existing demo-CSV tests all read
rows by *name* (`dict(zip(header, rows[0]))`), which is correct and
robust — and is exactly why nothing caught a document that names the
wrong *position*. Nothing pinned the order. So the guard here computes
with the same positional logic a spreadsheet would, and asserts the two
counts actually differ.

**The failure this must catch is a number that silently equals another
number**, so the mutation is exactly that: point the formula at a
non-label column and the two counts merge.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from string import ascii_uppercase


# No Qt dependency: this file only reads the column tuple and a document,
# so it runs in CI's `verify` job on both platforms, which is exactly
# where the SOP's arithmetic needs checking.


def _columns() -> tuple[str, ...]:
    from facecore.research.cli import G3_DEMO_RESULTS_CSV_COLUMNS

    return G3_DEMO_RESULTS_CSV_COLUMNS


def _letter(index: int) -> str:
    """Spreadsheet column letter for a 1-based column position."""
    return ascii_uppercase[index - 1] if index <= 26 else "?"


def _sop_text() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "docs" / "g3-local-test-sop.md").read_text(encoding="utf-8")


def _demo_csv(rows: list[dict[str, str]]) -> list[list[str]]:
    cols = _columns()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(cols))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in cols})
    return list(csv.reader(io.StringIO(buf.getvalue())))


class TestLabelKindColumn:
    def test_label_kind_is_column_22_not_20(self) -> None:
        """The exact number the SOP must state."""
        cols = _columns()
        assert cols.index("label_kind") == 21, (
            f"label_kind moved to column {cols.index('label_kind') + 1}; "
            "the SOP quotes a literal position and must be updated with it"
        )
        assert _letter(cols.index("label_kind") + 1) == "V"

    def test_column_R_is_margin_not_label_kind(self) -> None:
        """R is what S1 mistakenly used — pin what R actually is.

        `margin` is a float column, so any `<>unlabeled` test against it
        is vacuously true. This assertion documents why R is wrong, so
        the next reader does not "simplify" the formula back to it.
        """
        cols = _columns()
        assert cols[17] == "margin", f"column R is now {cols[17]!r}"
        assert _letter(18) == "R"


class TestColumnOrderIsPinnedIndependently:
    def test_the_full_order_is_a_frozen_literal(self) -> None:
        """The independent anchor the other assertions lack.

        `test_sop_quotes_the_right_column_number` and
        `test_label_kind_is_column_22_not_20` compare the SOP against the
        live tuple. A mutation that inserts a column ahead of
        `label_kind` moves both together, so those two stay green while
        the SOP's `V:V` silently starts reading the wrong field — the
        failure mode S1 already exhibited once.

        A literal cannot drift in step with the schema. Adding a column
        now fails HERE first, which forces whoever did it to update the
        schema, this literal, and the SOP in one conscious pass rather
        than discovering the break at the operator's spreadsheet.

        Reorder or insert/remove freely, but update this list in the same
        commit and re-check the two SOP assertions above.
        """
        assert list(_columns()) == [
            "mode",
            "app_version",
            "profile_version",
            "model_generation",
            "gallery_digest",
            "profile_digest",
            "round_id",
            "session_id",
            "started_utc",
            "labeled_at_utc",
            "elapsed_ms",
            "result",
            "reason_codes",
            "top1_identity",
            "top1_score",
            "top2_identity",
            "top2_score",
            "margin",
            "frames_sampled",
            "frames_usable",
            "required_support",
            "label_kind",
            "label_identity",
        ]


class TestSopFormula:
    def test_sop_quotes_the_right_column_number(self) -> None:
        """The SOP says 「第 20 欄」; the truth is 「第 22 欄」."""
        text = _sop_text()
        assert "第 22 欄" in text, "the SOP must state label_kind is column 22"
        assert "第 20 欄" not in text, (
            "the SOP still calls label_kind column 20; that is margin's "
            "neighbourhood and the formula derived from it is wrong"
        )

    def test_sop_formula_uses_column_v(self) -> None:
        """The formula must read `V:V`, not `R:R`."""
        text = _sop_text()
        assert 'V:V, "<>unlabeled"' in text, (
            "the 已標註輪次 formula must filter on V (label_kind)"
        )
        assert 'R:R, "<>unlabeled"' not in text, (
            "R is margin; a <>unlabeled test on a float column is always "
            "true and makes 已標註輪次 equal 已執行輪次"
        )


class TestFormulaActuallySeparatesTheTwoCounts:
    def test_unlabeled_rounds_split_the_counts(self) -> None:
        """Run the SOP's own arithmetic on a CSV containing unlabeled rows.

        This is the behavioural version of the two assertions above. The
        document checks pin what it *says*; this pins what it *does* —
        and a spreadsheet evaluating a wrong column would produce 5 = 5.
        """
        cols = _columns()
        label_idx = cols.index("label_kind")
        executed = 5
        rows = _demo_csv(
            [
                {"mode": "demo", "margin": "0.10", "label_kind": "enrolled"},
                {"mode": "demo", "margin": "0.12", "label_kind": "unenrolled"},
                {"mode": "demo", "margin": "0.08", "label_kind": "unlabeled"},
                {"mode": "demo", "margin": "0.09", "label_kind": "uncertain"},
                {"mode": "demo", "margin": "0.11", "label_kind": "unlabeled"},
            ]
        )
        body = rows[1:]

        correct = sum(1 for r in body if r[0] == "demo" and r[label_idx] != "unlabeled")
        wrong = sum(1 for r in body if r[0] == "demo" and r[17] != "unlabeled")

        assert executed == len(body) == 5
        assert correct == 3, (
            f"2 unlabeled rounds must be excluded, got {correct}"
        )
        assert wrong == executed, (
            "precondition: filtering column R reproduces S1 — the two "
            "counts collapse, which is the bug the SOP rewrite fixed"
        )
        assert correct != executed, (
            "if these are equal the formula is pointing at a column that "
            "never contains the string 'unlabeled' — the two numbers the "
            "decision asks to distinguish have merged"
        )
