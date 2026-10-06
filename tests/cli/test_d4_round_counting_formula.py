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
import re
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
            # D7-A W3 appended 12 columns. They are added strictly AFTER
            # `label_identity`, so every position 1-23 is unchanged and the
            # SOP's 「第 22 欄」 for label_kind (column V) still holds — see
            # TestLabelKindColumn above, which re-asserts that index
            # independently. Appending is the only shape W3 may take; an
            # insertion ahead of `label_kind` would move V:V onto the
            # wrong field, which is the S1 failure this file exists for.
            "recognition_duration_ms",
            "frames_rejected",
            # D7-A W3 rework 2 (R1): replaced the pair of raw event-type
            # counters as the answer to D4 §11 18b. `support_clear_reasons`
            # buckets by `reset_reason` and only counts genuine clears;
            # `score_reset_count` / `interval_skip_count` remain for
            # per-event-type inspection. Appended, so positions 1-23 and
            # the SOP's column 22 are untouched.
            "support_clear_reasons",
            "score_reset_count",
            "interval_skip_count",
            "probe_kind",
            "presenting_identity",
            "match_threshold",
            "review_threshold",
            "margin_threshold",
            # `required_support` is not re-declared: it already sits at
            # position 21 and is reused in the snapshot block below, so a
            # second declaration would silently override the first in the
            # row dict.
            "min_support_interval_ms",
            "timeout_ms",
            # D7-A W1 appended 3 more, again strictly AFTER
            # `timeout_ms`. Same rule as W3: appending is the only shape
            # permitted, because positions 1-23 (margin at 18,
            # label_kind at 22) are what the operator's spreadsheet
            # formulas read.
            "expected_count",
            "loaded_count",
            "gallery_rejected",
            # G3-w appended one more, again strictly AFTER
            # `gallery_rejected`. Same rule as W3 and W1, for the same
            # reason: appending is the only shape permitted, because
            # positions 1-23 (margin at 18, label_kind at 22) are what the
            # operator's spreadsheet formulas read. Appended at the end,
            # so the SOP's column 22 for label_kind still holds — the two
            # SOP assertions above re-assert that independently.
            "operator_verdict",
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


class TestTheSopsOwnFormulaDoesTheWork:
    """The behavioural guard, pointed at the deliverable.

    Round 2 review (S1 + N1): the previous version of this guard
    recomputed the counts from `cols.index("label_kind")` with its own
    `sum()`, so it tested *this file's* arithmetic rather than the
    SOP's. Two consequences, both confirmed by the reviewer:

    - It could not name S1's wrong column. All 22 non-`label_kind`
      columns make `wrong == executed` true, so `assert wrong ==
      executed` was a tautology: it proves "some column that is not
      label_kind collapses the counts", not "R is the wrong one".
    - Restoring the SOP's `R:R` left this guard green, because the
      two were only ever linked by an assumption nobody checked.

    So this version *parses the column letter out of the SOP's own
    formula* and evaluates that. If the document points at a column
    that never holds "unlabeled", the counts collapse here — the
    failure lands on the artefact the operator will actually use, and
    no frozen letter is involved on this path at all.
    """

    @staticmethod
    def _formula_column() -> str:
        """The column letter the SOP tells the operator to filter on."""
        match = re.search(
            r'COUNTIFS\([^)]*?,\s*([A-Z]{1,2}):\1,\s*"<>unlabeled"\)', _sop_text()
        )
        assert match, (
            "the SOP's 已標註輪次 formula was not found in the expected "
            "COUNTIFS(<mode column>, \"<label column>:<label column>\", "
            "\"<>unlabeled\") shape"
        )
        return match.group(1)

    def test_the_sops_own_column_separates_the_counts(self) -> None:
        """Evaluate the SOP's formula against a CSV with unlabeled rows.

        If the formula's column is wrong the two numbers merge, exactly
        as they would in the operator's spreadsheet — and this fails
        with the column name, not just a count.
        """
        cols = _columns()
        letter = self._formula_column()
        idx = ord(letter) - ord("A")

        assert idx < len(cols), f"the SOP references column {letter}, beyond the CSV"
        assert cols[idx] == "label_kind", (
            f"the SOP filters on {letter}, which holds {cols[idx]!r} — that "
            "column never contains 'unlabeled', so 已標註輪次 silently "
            "equals 已執行輪次 (the S1 bug)"
        )

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

        executed = len(body)
        labeled = sum(1 for row in body if row[idx] != "unlabeled")

        assert executed == 5
        assert labeled == 3, (
            f"the SOP's formula over column {letter} counted {labeled} of "
            f"{executed} rounds as labeled; 2 unlabeled rounds must be excluded"
        )
        assert labeled != executed, (
            "the two numbers the decision asks to distinguish have merged"
        )

    def test_the_formula_also_matches_the_prose_column_number(self) -> None:
        """The formula letter and the prose must name the same column.

        Without this, a future edit could leave `V:V` in the formula
        while the prose says "第 23 欄" — the document would contradict
        itself and the operator would have no way to tell which to trust.
        """
        letter = self._formula_column()
        text = _sop_text()
        position = _columns().index("label_kind") + 1
        assert f"第 {position} 欄" in text, (
            f"the SOP prose must state label_kind is column {position} "
            f"(letter {letter})"
        )
        assert _letter(position) == letter, (
            f"the formula uses {letter} but label_kind is column {position} "
            f"({_letter(position)}) — the document contradicts itself"
        )
