"""Every demo-row writer must forward the W0-b ground-truth columns.

`probe_kind` / `presenting_identity` are the operator's ground truth for a
round, and more than one place writes a demo row. If a writer forgets
them, the round still lands in the CSV — with empty cells that read as
「沒有記錄」 — and nothing downstream can tell that apart from a round the
operator genuinely recorded nothing.

⚠️ Why this file exists, and why it parses the AST instead of counting
strings.

An earlier guard in `test_d7_w0b_probe_kind_input.py` counted how many
times the literal `probe_kind=self._probe_kind_value()` appeared in the
module source and asserted `>= 2`. It did catch removing one of the two
real writers. But it cannot see a third one:

    def _future_export_path(self):
        append_g3_demo_results_csv(csv, round_, required_support=3,
                                   labeled_at_utc=now)   # no ground truth

    → the literal still appears twice, so that guard stays green.

That is precisely the failure its own docstring claimed to defend
against ("the gap is only findable at the call site"), and a count is
not a call site. Counting also over-counts the other way: one mention
in a comment or docstring satisfies the threshold while both real
writers are broken.

So these guards walk `ast.Call` nodes and speak about structure. A second
reason for structure over text: a hard-coded `probe_kind="target"`
satisfies any keyword-presence check while being wrong, because the
whole point is that the value differs per round.

⚠️ This file deliberately needs no Qt. The behavioural assertions about
the value path live in `test_d7_w0b_probe_kind_input.py`, which requires
PySide6 and therefore only runs in `qt-smoke`. This file parses source,
so it also runs in `verify` — the job where a wiring mistake is cheapest
to catch, and where a guard that silently never executes would do the
most damage.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "facecore"
    / "live"
    / "qt_window.py"
)
_WRITER = "append_g3_demo_results_csv"
_REQUIRED = ("probe_kind", "presenting_identity")
# The accessors that read the window's live combo state. A forwarding
# site must call these rather than pass a constant: the columns are
# per-round, and 「per round」 is enforced only by reading them at write
# time.
_ACCESSORS = frozenset({"_probe_kind_value", "_presenting_identity_value"})


def _callee(node: ast.Call) -> str | None:
    """The called function's name, whether attribute or bare."""
    func = node.func
    return getattr(func, "attr", None) or getattr(func, "id", None)


def _writer_calls() -> list[ast.Call]:
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _callee(node) == _WRITER
    ]


def _kwargs(node: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in node.keywords if kw.arg}


def test_the_writer_is_actually_called_somewhere() -> None:
    """Anchor the assertions below.

    Without this, moving or deleting every writer would leave
    `test_every_writer_forwards_the_ground_truth` vacuously true — the
    same 「a guard that cannot fail」 shape this file exists to remove.
    """
    calls = _writer_calls()
    assert calls, (
        f"no {_WRITER}(...) call found in {_SRC.name}; the wiring guards "
        "below would pass on an empty list"
    )


def test_every_writer_forwards_both_ground_truth_columns() -> None:
    """Each call site passes both columns — including future ones."""
    missing = {
        node.lineno: sorted(set(_REQUIRED) - set(_kwargs(node)))
        for node in _writer_calls()
        if set(_REQUIRED) - set(_kwargs(node))
    }
    assert not missing, (
        f"every {_WRITER}(...) call must pass probe_kind and "
        f"presenting_identity; missing at {missing}. A writer that omits "
        "them records empty cells, which read as 「沒有記錄」 — "
        "indistinguishable from a round the operator deliberately left "
        "unrecorded. This guard is structural, so a writer added later "
        "without the columns fails here."
    )


def test_no_writer_passes_a_constant_ground_truth() -> None:
    """The forwarded value must be read live, not baked into the call.

    Keyword presence alone is not enough: `probe_kind="target"` passes a
    presence check and is wrong, because the value has to differ per
    round. Each required kwarg must therefore be a call to the window
    accessor that reads the combo at write time.
    """
    constants = {
        node.lineno: sorted(
            name for name in _REQUIRED if not _is_live_read(_kwargs(node).get(name))
        )
        for node in _writer_calls()
        if set(_REQUIRED) <= set(_kwargs(node))
    }
    hard_coded = {ln: names for ln, names in constants.items() if names}
    assert not hard_coded, (
        "ground-truth columns must be forwarded as "
        "self._probe_kind_value() / self._presenting_identity_value() "
        f"(read at write time), not constants. Found at {hard_coded}"
    )


def _is_live_read(value: ast.expr | None) -> bool:
    """True when the expression is a call to one of the live accessors."""
    if not isinstance(value, ast.Call):
        return False
    return _callee(value) in _ACCESSORS
