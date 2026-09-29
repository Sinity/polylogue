"""The pre-push hook (`.githooks/pre-push`) driven through a real `git push`.

Each test pushes from a scratch repository whose `core.hooksPath` is this
checkout's `.githooks`, to a bare remote, with a recording `devtools` stub on
PATH. Git itself feeds the hook its ref lines, so the tests cover the route an
agent's push takes. They fail when the hook stops running the quick gate on the
pushed HEAD, lets a failed gate through, gates a working tree that differs from
the pushed commit, gates a ref that is not HEAD, runs for a deletion, or passes
when devtools is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOKS = REPO_ROOT / ".githooks"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="the hook test requires bash and git",
)


class Checkout:
    def __init__(self, root: Path, *, with_devtools: bool = True, gate_exit: int = 0) -> None:
        self.root = root
        self.work = root / "work"
        self.remote = root / "remote.git"
        self.calls = root / "devtools.calls"
        tools = root / "tools"
        tools.mkdir()
        for name in ("git", "bash"):
            found = shutil.which(name)
            assert found is not None
            (tools / name).symlink_to(found)
        path = [str(tools)]
        if with_devtools:
            stub = root / "stub"
            stub.mkdir()
            devtools = stub / "devtools"
            devtools.write_text(f"#!{tools / 'bash'}\nprintf '%s\\n' \"$*\" >> '{self.calls}'\nexit {gate_exit}\n")
            devtools.chmod(0o755)
            path.insert(0, str(stub))
        self.env = {
            "PATH": os.pathsep.join(path),
            "HOME": str(root),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CEILING_DIRECTORIES": str(root.parent),
        }
        self.git("init", "-q", "--bare", str(self.remote), cwd=root)
        self.git("init", "-q", "-b", "feature", str(self.work), cwd=root)
        self.git("config", "core.hooksPath", str(HOOKS))
        self.git("config", "user.name", "fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.git("remote", "add", "origin", str(self.remote))

    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd or self.work),
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
            check=check,
        )

    def commit(self, text: str) -> str:
        (self.work / "file.txt").write_text(text)
        self.git("add", "file.txt")
        self.git("commit", "-q", "-m", text)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def push(self, *refspecs: str) -> subprocess.CompletedProcess[str]:
        return self.git("push", "origin", *refspecs, check=False)

    def remote_head(self, branch: str) -> str | None:
        done = self.git("rev-parse", "--verify", "-q", f"refs/heads/{branch}", cwd=self.remote, check=False)
        return done.stdout.strip() or None

    def gate_calls(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []


def test_a_clean_head_push_runs_the_quick_gate(tmp_path: Path) -> None:
    checkout = Checkout(tmp_path)
    head = checkout.commit("one")

    done = checkout.push("feature")

    assert done.returncode == 0, done.stderr
    assert checkout.gate_calls() == ["verify --quick"]
    assert checkout.remote_head("feature") == head


def test_a_failed_gate_refuses_the_push(tmp_path: Path) -> None:
    checkout = Checkout(tmp_path, gate_exit=1)
    checkout.commit("one")

    done = checkout.push("feature")

    assert done.returncode != 0
    assert checkout.gate_calls() == ["verify --quick"]
    assert checkout.remote_head("feature") is None


def test_tracked_changes_on_top_of_head_refuse_the_push(tmp_path: Path) -> None:
    checkout = Checkout(tmp_path)
    checkout.commit("one")
    (checkout.work / "file.txt").write_text("uncommitted")

    done = checkout.push("feature")

    assert done.returncode != 0
    assert checkout.gate_calls() == []
    assert checkout.remote_head("feature") is None


def test_a_ref_that_is_not_head_is_left_to_the_hosted_gate(tmp_path: Path) -> None:
    checkout = Checkout(tmp_path)
    older = checkout.commit("one")
    checkout.commit("two")

    done = checkout.push(f"{older}:refs/heads/older")

    assert done.returncode == 0, done.stderr
    assert checkout.gate_calls() == []
    assert checkout.remote_head("older") == older


def test_a_deletion_runs_no_gate(tmp_path: Path) -> None:
    checkout = Checkout(tmp_path)
    older = checkout.commit("one")
    checkout.commit("two")
    assert checkout.push(f"{older}:refs/heads/older").returncode == 0

    done = checkout.push(":refs/heads/older")

    assert done.returncode == 0, done.stderr
    assert checkout.gate_calls() == []
    assert checkout.remote_head("older") is None


def test_a_missing_devtools_refuses_the_push(tmp_path: Path) -> None:
    checkout = Checkout(tmp_path, with_devtools=False)
    checkout.commit("one")

    done = checkout.push("feature")

    assert done.returncode != 0
    assert checkout.remote_head("feature") is None
