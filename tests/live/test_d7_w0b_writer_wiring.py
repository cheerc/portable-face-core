"""Every demo-row writer must forward the W0-b ground-truth columns.

`probe_kind` / `presenting_identity` are the operator's ground truth for a
round, and more than one place writes a demo row. Two things can go wrong
at those call sites, and they are different failures:

· a writer that omits the columns records empty cells, which read as
  「沒有記錄」 and are indistinguishable from a round the operator
  deliberately left unrecorded;
· a writer that is deleted outright stops recording at all.

⚠️ Why this file parses the AST instead of counting strings.

An earlier guard counted how many times the literal
`probe_kind=self._probe_kind_value()` appeared in the module source and
asserted `>= 2`. It caught removing one of the two real writers, and it
missed two other shapes:

    # a third writer that omits the columns
    append_g3_demo_results_csv(csv, r, required_support=3, ...)
    → the literal still appears twice, so the guard stays green

    # a comment satisfies the count while both real writers are broken
    # probe_kind=self._probe_kind_value()
    → count reaches 2 with nothing forwarded anywhere

So counting text is replaced by counting structure: the calls themselves,
identified by the function each one sits in.

⚠️ The three guards below are deliberately not redundant:

`test_both_demo_row_paths_are_present`
    「there are two writers」 is what the `>= 2` count used to provide by
    accident. Without it, deleting a writer is invisible: 「every writer
    forwards both columns」 stays true, because the survivor forwards
    them. That survivor is the unlabeled path, so deleting the labeled
    one means no 正確／錯誤 round reaches the demo CSV at all.

`test_every_writer_forwards_both_ground_truth_columns`
    the F2 concern — a writer, present or future, that forgets the
    columns.

`test_no_writer_passes_a_constant_ground_truth`
    keyword presence alone is not enough: `probe_kind="target"` passes
    any presence check and is wrong, since the value must differ per
    round.

⚠️ This file needs no Qt. The behavioural assertions about the value path
live in `test_d7_w0b_probe_kind_input.py`, which requires PySide6 and so
only runs in `qt-smoke`. This one parses source, so it also runs in
`verify` — the job where a wiring mistake is cheapest to catch, and where
a guard that silently never executes would do the most damage.
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
# The two demo-row writers, named by the function each lives in. The
# labeled path is the operator's 正確／錯誤 button, so its absence is
# user-visible; the unlabeled one is 再次辨識 with no verdict.
_EXPECTED_WRITERS = ("_press_key", "_record_unlabeled_round")


def _callee(node: ast.Call) -> str | None:
    """The called function's name, whether attribute or bare."""
    func = node.func
    return getattr(func, "attr", None) or getattr(func, "id", None)


def _annotated_tree() -> ast.Module:
    """Parse the module, tagging each node with its parent.

    `ast` carries no parent links, and threading one through by hand is
    easy to get subtly wrong — which is why it is done in one place
    rather than at each call site.
    """
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]
    return tree


def _enclosing_function(node: ast.AST) -> str | None:
    """The nearest `def` around `node`, via the parent chain."""
    current = getattr(node, "_parent", None)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = getattr(current, "_parent", None)
    return None


def _kwargs(node: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in node.keywords if kw.arg}


def _writer_calls() -> list[tuple[int, str | None, dict[str, ast.expr]]]:
    """Every writer call: line, enclosing function, keyword arguments."""
    found = [
        (node.lineno, _enclosing_function(node), _kwargs(node))
        for node in ast.walk(_annotated_tree())
        if isinstance(node, ast.Call) and _callee(node) == _WRITER
    ]
    return sorted(found)


def _is_live_read(value: ast.expr | None) -> bool:
    """True when the expression calls one of the live accessors."""
    if not isinstance(value, ast.Call):
        return False
    return _callee(value) in _ACCESSORS


def test_both_demo_row_paths_are_present() -> None:
    """Exactly the two writers that exist, identified by function name.

    ⚠️ The assertion that 「every writer forwards both columns」 cannot
    make on its own. Deleting one writer leaves it true — vacuously,
    because the survivor still forwards them — and the survivor is the
    unlabeled path, so deleting the labeled one means no 正確／錯誤
    round reaches the demo CSV. Naming the two paths makes deletion
    fail here instead.
    """
    calls = _writer_calls()
    functions = sorted(name for _, name, _ in calls)
    assert functions == sorted(_EXPECTED_WRITERS), (
        f"expected exactly {_EXPECTED_WRITERS} to call {_WRITER}(...) — "
        f"the labeled 正確／錯誤 path and the 再次辨識 path — but found "
        f"{functions} at lines {[ln for ln, _, _ in calls]}. A missing "
        "writer is invisible to the forwarding guard below."
    )


def test_every_writer_forwards_both_ground_truth_columns() -> None:
    """Each call site passes both columns — including future ones."""
    missing = {
        lineno: sorted(set(_REQUIRED) - set(kwargs))
        for lineno, _, kwargs in _writer_calls()
        if set(_REQUIRED) - set(kwargs)
    }
    assert not missing, (
        f"every {_WRITER}(...) call must pass probe_kind and "
        f"presenting_identity; missing at {missing}. A writer that omits "
        "them records empty cells, which read as 「沒有記錄」 — "
        "indistinguishable from a round the operator deliberately left "
        "unrecorded."
    )


def test_no_writer_passes_a_constant_ground_truth() -> None:
    """The forwarded value must be read live, not baked into the call."""
    offenders = {
        lineno: sorted(
            name for name in _REQUIRED if not _is_live_read(kwargs.get(name))
        )
        for lineno, _, kwargs in _writer_calls()
        if set(_REQUIRED) <= set(kwargs)
    }
    hard_coded = {ln: names for ln, names in offenders.items() if names}
    assert not hard_coded, (
        "ground-truth columns must be forwarded as "
        "self._probe_kind_value() / self._presenting_identity_value() "
        f"(read at write time), not constants. Found at {hard_coded}"
    )
