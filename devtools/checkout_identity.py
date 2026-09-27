"""Which checkout a verification runs in, and the refusal to run on the base.

An agent's shell working directory can reset to the primary checkout between
commands. That checkout sits on the default branch, so a ``devtools test`` or
``devtools verify`` issued without ``env -C <worktree>`` silently tested the
base instead of the change and reported a green that proved nothing about it.
Both runners therefore refuse on the default branch unless the caller opts in
with :data:`ON_DEFAULT_BRANCH_FLAG`, and both name the checkout, branch and head
in their first and last output lines and in the receipt, so a cited receipt
says what it tested.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Opt-in for a deliberate run on the default branch: a base comparison, or
#: the hosted gate on a push to the default branch.
ON_DEFAULT_BRANCH_FLAG = "--on-default-branch"
#: Exit status of the refusal; the same status the agent-tier refusal uses.
REFUSAL_EXIT = 2
REFUSAL_DIAGNOSIS = "default_branch_refused"
#: Carries the opt-in to a queued run's slot, which re-checks the branch when
#: the run actually starts: a checkout can switch branch while its run waits.
ALLOW_DEFAULT_BRANCH_ENV = "POLYLOGUE_ALLOW_DEFAULT_BRANCH"
_FALLBACK_DEFAULT_BRANCH = "master"


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None


@dataclass(frozen=True, slots=True)
class CheckoutIdentity:
    """The checkout a run executes in."""

    root: Path
    #: ``None`` for a detached HEAD.
    branch: str | None
    head: str | None
    default_branch: str
    #: Commits the default branch names locally and on ``origin``.
    default_tips: frozenset[str] = frozenset()

    @property
    def on_default_branch(self) -> bool:
        """On the default branch, or detached at one of its tips."""
        if self.branch is not None:
            return self.branch == self.default_branch
        return self.head is not None and self.head in self.default_tips

    def describe(self) -> str:
        head = self.head[:12] if self.head else "unknown"
        return f"checkout={self.root} branch={self.branch or '(detached)'} head={head}"


def checkout_identity(root: Path) -> CheckoutIdentity:
    remote_default = _git(root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD")
    default = (remote_default.split("/", 1)[1] if remote_default and "/" in remote_default else None) or (
        _FALLBACK_DEFAULT_BRANCH
    )
    tips = {
        _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        for ref in (f"refs/heads/{default}", f"refs/remotes/origin/{default}")
    }
    return CheckoutIdentity(
        root=root.resolve(),
        branch=_git(root, "symbolic-ref", "--quiet", "--short", "HEAD"),
        head=_git(root, "rev-parse", "HEAD"),
        default_branch=default,
        default_tips=frozenset(tip for tip in tips if tip),
    )


def default_branch_refusal(identity: CheckoutIdentity, *, command: str, allowed: bool) -> str | None:
    """Why *command* refuses to run in this checkout, or ``None``."""
    if allowed or not identity.on_default_branch:
        return None
    where = identity.branch or f"{identity.default_branch} (detached at its tip)"
    return (
        f"{command}: refused on the default branch `{where}` in {identity.root}.\n"
        f"  A run here tests `{identity.default_branch}`, not your change. Run it in your worktree:\n"
        f"    env -C /path/to/your/worktree {command} ...\n"
        f"  For a deliberate run on the base, pass {ON_DEFAULT_BRANCH_FLAG}."
    )


__all__ = [
    "ALLOW_DEFAULT_BRANCH_ENV",
    "ON_DEFAULT_BRANCH_FLAG",
    "REFUSAL_DIAGNOSIS",
    "REFUSAL_EXIT",
    "CheckoutIdentity",
    "checkout_identity",
    "default_branch_refusal",
]
