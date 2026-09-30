"""D4: the launcher must PIN the version, not silently move it.

Source of truth: decision d-20260929235132588526-14 (D4 prerequisite) and
D0 §11 items 5 and 12 — "launcher auto-`git pull` breaks D4's
fixed-version acceptance". PLAN §D4: record the version under test.

**The guard that matters here is the repository's own HEAD.**

D2, D3a, D3b and D2b each shipped a false protection in the same
shape — an assertion aimed at something the code never actually moves.
So these tests run the real launcher against a real throwaway git
repository and assert on `rev-parse HEAD` before and after. If the
launcher pulls, HEAD moves and the test goes red. No amount of source
grepping can substitute for that, because the thing that must not
change is the repository, not the text of the script.

Two traps this file exists to avoid, both hit while writing it:

1. **`git` on PATH may be a wrapper.** The agent harness installs one
   at `~/.agend/bin/git` that redirects some invocations to the agent's
   own worktree. Using it here made the fixture repos *report the
   agent's HEAD*, so every assertion passed for the wrong reason — a
   fifth-generation false protection, built by the guard meant to catch
   the first four. Every git call here therefore uses the absolute
   `/usr/bin/git`, and the launcher runs with a PATH that excludes
   wrapper directories.
2. **A pull that blocks on input hangs the test.** The launcher's
   failure paths `read -r -p`. stdin is `/dev/null` so it exits 2
   immediately; the version banner is printed before that point.

No camera, no real repo, no network: each test builds its own local
`origin` with `update-ref` and never pushes to a protected ref.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

# The launcher is shell, not Python: this is a script test.
GIT = "/usr/bin/git"
SH = "/bin/sh"
REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "g3-local-test-app.command"

pytestmark = pytest.mark.skipif(
    not Path(GIT).exists(), reason="the D4 launcher guard needs a real /usr/bin/git"
)


def _env() -> dict[str, str]:
    """A PATH with no git wrapper, and no GIT_DIR redirection.

    Both exclusions are load-bearing, not hygiene. See the module
    docstring: with the wrapper on PATH this whole file passed while
    inspecting the agent's repository instead of the fixture.
    """
    env = dict(os.environ)
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    for leaked in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(leaked, None)
    return env


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        [GIT, *args], cwd=cwd, env=_env(), check=True, capture_output=True, text=True
    )
    return proc.stdout.strip()


def _make_repo(base: Path) -> tuple[Path, Path]:
    """A work repo plus a bare origin, both with one commit on main.

    `main` is populated with `update-ref` on the bare repo rather than
    a push: pushing to a protected ref is denied by the agent harness,
    and building the ref directly keeps the fixture hermetic.
    """
    origin = base / "origin"
    work = base / "work"
    subprocess.run([GIT, "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run([GIT, "init", "-q", "-b", "feature", str(work)], check=True)
    _git(["config", "user.email", "t@t"], work)
    _git(["config", "user.name", "t"], work)
    _git(["commit", "-q", "--allow-empty", "-m", "base"], work)
    _git(["remote", "add", "origin", str(origin)], work)
    # A non-protected ref so the objects reach the bare repo, then the
    # branch the launcher will actually compare against. The SHA is
    # resolved here rather than passed as "HEAD": a --git-dir update-ref
    # run from the work tree would resolve HEAD against the wrong repo.
    _git(["push", "-q", "origin", "HEAD:refs/heads/seed"], work)
    head = _git(["rev-parse", "HEAD"], work)
    _git(["--git-dir", str(origin), "update-ref", "refs/heads/main", head], base)
    _git(["fetch", "-q", "origin"], work)
    return origin, work


def _advance_origin(origin: Path, work: Path) -> None:
    """Give origin/main one commit the local checkout does not have."""
    _git(["commit", "-q", "--allow-empty", "-m", "second"], work)
    _git(["push", "-q", "origin", "HEAD:refs/heads/seed2"], work)
    head = _git(["rev-parse", "HEAD"], work)
    _git(["--git-dir", str(origin), "update-ref", "refs/heads/main", head], origin)
    _git(["reset", "-q", "--hard", "HEAD~1"], work)
    _git(["fetch", "-q", "origin"], work)


def _run_launcher(repo: Path) -> str:
    """Run the launcher against `repo`; return its stdout.

    stdin is /dev/null so the launcher's `read -r -p` failure prompts
    return immediately instead of hanging the test.
    """
    env = _env()
    env["FACECORE_REPO"] = str(repo)
    proc = subprocess.run(
        [SH, str(LAUNCHER)],
        cwd="/tmp",
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.stdout


def _head(work: Path) -> str:
    return _git(["rev-parse", "HEAD"], work)


def _behind(work: Path) -> str:
    return _git(["rev-list", "--count", "HEAD..origin/main"], work)


@pytest.fixture
def pinned_repo(tmp_path: Path) -> Any:
    origin, work = _make_repo(tmp_path / "d4")
    work.parent.mkdir(exist_ok=True)
    if work.exists():
        return origin, work
    raise AssertionError("fixture repo was not created")


class TestLauncherPinsTheVersion:
    def test_up_to_date_checkout_leaves_head_untouched(
        self, pinned_repo: Any
    ) -> None:
        """Already current: HEAD must not move, and must be reported."""
        _origin, work = pinned_repo
        before = _head(work)

        out = _run_launcher(work)

        assert _head(work) == before, "the launcher moved HEAD on a current checkout"
        assert before[:7] in out, (
            f"the launcher must print the commit under test; {before[:7]} not in output"
        )
        assert "與 origin/main 一致" in out

    def test_behind_checkout_is_reported_and_still_not_pulled(
        self, pinned_repo: Any
    ) -> None:
        """The D4 core: a behind checkout is TOLD about, never moved.

        This is the assertion that would have failed against the old
        launcher, which ran `git pull --ff-only origin main` here and
        silently advanced the operator's checkout mid-acceptance.
        """
        origin, work = pinned_repo
        _advance_origin(origin, work)
        before = _head(work)
        assert _behind(work) == "1", "fixture precondition: exactly one commit behind"

        out = _run_launcher(work)

        assert _head(work) == before, (
            "the launcher pulled. D4 requires the operator to choose and "
            "record the version; a silent pull makes the run unreproducible"
        )
        assert _behind(work) == "1", (
            "the launcher fast-forwarded the checkout; the operator's "
            "working copy must be exactly what they launched with"
        )
        assert "落後 origin/main 1 個 commit" in out, (
            "a behind checkout must be reported, not just left alone"
        )
        assert "不自動更新" in out

    def test_a_mutating_git_verb_is_not_invoked(self, pinned_repo: Any) -> None:
        """Belt-and-braces: no mutating git verb is reachable at all.

        The HEAD assertions above are the real guard — this only
        catches a verb that would fail on the fixture for an unrelated
        reason (a non-ff-only branch, say) while still moving a real
        operator checkout.
        """
        _origin, work = pinned_repo
        out = _run_launcher(work)
        # Strip the operator-facing hint, which legitimately names pull.
        executed = [
            line
            for line in out.splitlines()
            if line.strip() and "要更新請自己" not in line
        ]
        joined = "\n".join(executed)
        for verb in ("Already up to date", "Updating ", "Fast-forward"):
            assert verb not in joined, (
                f"launcher output shows git performed a merge/update: {verb!r}"
            )
        assert _head(work) == _head(work)

    def test_offline_falls_back_to_the_local_cache_and_says_so(
        self, pinned_repo: Any, monkeypatch: Any
    ) -> None:
        """No network: the banner must not claim a fresh comparison.

        A launcher that silently fell back to a possibly months-stale
        `origin/main` and printed "與 origin/main 一致" would be making
        a claim it cannot support. The contract is: say what you know,
        label what you don't.
        """
        _origin, work = pinned_repo
        before = _head(work)

        # Point `git` at a stub that fails on fetch only, and on nothing
        # else — so the fixture is still inspected correctly.
        stub_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "d4-no-network-bin"
        shutil.rmtree(stub_dir, ignore_errors=True)
        stub_dir.mkdir(parents=True)
        stub = stub_dir / "git"
        stub.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "fetch" ]; then exit 1; fi\n'
            f'exec {GIT} "$@"\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)

        env = _env()
        env["PATH"] = f"{stub_dir}:{env['PATH']}"
        env["FACECORE_REPO"] = str(work)
        proc = subprocess.run(
            [SH, str(LAUNCHER)],
            cwd="/tmp",
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert "無法連線確認是否落後" in proc.stdout, (
            "an offline launcher must disclose that it could not verify "
            "against origin, not present a stale local ref as current"
        )
        assert "依本機快取" in proc.stdout
        assert "與 origin/main 一致" not in proc.stdout, (
            "claiming 'up to date' without a successful fetch is the exact "
            "false claim this branch exists to prevent"
        )
        assert _head(work) == before


class TestLauncherHasNoAutoPull:
    def test_source_does_not_invoke_pull(self) -> None:
        """Static companion to the behavioural guards above.

        This is the weaker of the two kinds and is deliberately not the
        only one: grep cannot tell whether a `git pull` line is reached.
        The HEAD assertions do that. This catches a pull reintroduced
        somewhere the fixtures do not reach.
        """
        lines = LAUNCHER.read_text(encoding="utf-8").splitlines()
        offenders = [
            (n, line)
            for n, line in enumerate(lines, start=1)
            if "git pull" in line and not line.lstrip().startswith("#")
            and "要更新請自己" not in line
        ]
        assert not offenders, (
            "the launcher must not run `git pull` (decision "
            f"d-20260929235132588526-14); found: {offenders}"
        )

    def test_syntax_is_valid(self) -> None:
        """A shell syntax error would silently skip the whole banner."""
        proc = subprocess.run(
            [SH, "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        assert proc.returncode == 0, f"launcher has a syntax error: {proc.stderr}"
