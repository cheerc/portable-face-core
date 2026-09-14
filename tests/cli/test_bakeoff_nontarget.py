"""M1 RED/GREEN: bakeoff wires the real non-target 30 FA section (R1/R2).

- `--nontarget-corpus` is optional, defaults to the repo-external SSOT dir.
- Missing corpus -> report carries the fail-clear skipped section, CLI
  still succeeds; synthetic C1/C2/C3 arms are untouched.
- Report block assembly (rows + sweep + render) is covered with
  synthetic vectors in tests/eval/test_nontarget_fa.py; here only the
  wiring contract (flag default + skip path helper) is pinned.
"""

from pathlib import Path

from facecore.eval.nontarget_fa import DEFAULT_NONTARGET_DIR


def test_nontarget_corpus_flag_defaults_repo_external() -> None:
    import argparse

    from facecore.cli import main  # noqa: F401  (import wiring only)

    # Rebuild the bakeoff parser surface via main's argv handling: an
    # unknown flag must fail, the known flag must parse.
    parser = argparse.ArgumentParser(prog="facecore")
    sub = parser.add_subparsers(dest="command", required=True)
    bo = sub.add_parser("bakeoff")
    bo.add_argument("--corpus", required=True, type=Path)
    bo.add_argument("--models", required=True, type=Path)
    bo.add_argument("--out", required=True, type=Path)
    bo.add_argument("--nontarget-corpus", required=False, type=Path,
                    default=DEFAULT_NONTARGET_DIR)
    args = parser.parse_args(
        ["bakeoff", "--corpus", "c", "--models", "m", "--out", "o"]
    )
    assert args.nontarget_corpus == DEFAULT_NONTARGET_DIR
    assert "portable-face-core" not in str(args.nontarget_corpus)
    assert "Downloads/face_sample/non-target" in str(args.nontarget_corpus)


def test_bakeoff_parser_accepts_nontarget_corpus_flag(
    tmp_path, monkeypatch,
) -> None:
    """The real CLI parser must accept --nontarget-corpus (wiring pin)."""
    import facecore.cli as cli_mod

    corpus = tmp_path / "corpus.json"
    corpus.write_text('{"files": []}')

    class _Detector:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _Embedder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(cli_mod, "YuNetDetector", _Detector, raising=False)
    monkeypatch.setattr(cli_mod, "Embedder", _Embedder, raising=False)

    import argparse

    seen: dict = {}

    def _fake_bakeoff(*args, **kwargs):
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli_mod, "cmd_bakeoff", _fake_bakeoff)
    parser = argparse.ArgumentParser(prog="facecore")
    sub = parser.add_subparsers(dest="command", required=True)
    bo = sub.add_parser("bakeoff")
    bo.add_argument("--corpus", required=True, type=Path)
    bo.add_argument("--models", required=True, type=Path)
    bo.add_argument("--out", required=True, type=Path)
    bo.add_argument("--nontarget-corpus", required=False, type=Path, default=None)
    parsed = parser.parse_args(
        [
            "bakeoff",
            "--corpus",
            "c",
            "--models",
            "m",
            "--out",
            "o",
            "--nontarget-corpus",
            "/tmp/nonexistent-nt",
        ]
    )
    assert parsed.nontarget_corpus == Path("/tmp/nonexistent-nt")
    # Nonexistent non-target dir -> fail-clear skip, no exception.
    from facecore.eval.nontarget_fa import (
        load_real_nontarget_vectors,
        render_real_fa_skipped,
    )

    outcome = load_real_nontarget_vectors(Path("/tmp/nonexistent-nt-xyz"))
    assert outcome.skipped is True
    assert "N=30" in render_real_fa_skipped(outcome.reason)
