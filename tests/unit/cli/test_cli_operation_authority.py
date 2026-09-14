"""No daemon means a typed, actionable refusal — never a second CLI writer.

The operator ruling behind step S10 is that there is no requirement for the
CLI to ever work standalone: every durable ``user.db`` write lowers to a
declared daemon operation, and with no daemon answering the command must
refuse in a way the operator can act on — naming the operation it could not
run and the command that makes it runnable (``polylogued run``).

Anti-vacuity: restoring a CLI-side writable ``ArchiveStore``, or adding a local
fallback inside ``submit_cli_mutation``, makes every command here exit 0 and
mutate ``user.db``, turning every test in this module red — both the exit-code
assertion and the byte-identical ``user.db`` digest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder

_SESSION_ID = "claude-code-session:ext-conv-authority"


@pytest.fixture
def authority_archive(tmp_path: Path) -> Path:
    """An archive root holding one plain session and a bootstrapped user tier."""
    initialize_active_archive_root(tmp_path)
    (
        SessionBuilder(tmp_path / "index.db", "conv-authority")
        .provider("claude-code")
        .title("Authority session")
        .add_message("m0", role="user", text="hello alpha bravo")
        .save()
    )
    return tmp_path


def _run(archive_root: Path, *args: str) -> Result:
    """Invoke the CLI with every daemon route explicitly switched off."""
    env = {
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_DB_PATH": str(archive_root / "index.db"),
        "POLYLOGUE_NO_DAEMON": "1",
        "POLYLOGUE_FORCE_PLAIN": "1",
    }
    return CliRunner().invoke(cli, list(args), env=env)


def _user_tier_digest(archive_root: Path) -> str:
    """Digest ``user.db`` and its journal: any local write changes it."""
    digest = hashlib.sha256()
    for path in sorted(archive_root.glob("user.db*")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _refusal_text(result: Result) -> str:
    return f"{result.output}\n{result.exception}"


@pytest.mark.parametrize(
    ("verb_args", "operation"),
    [
        (("mark", "--star"), "mutation.session.mark"),
        (("mark", "--tag-add", "X"), "mutation.session.tag"),
        (("mark", "--tag-remove", "X"), "mutation.session.tag"),
        (("mark", "--note", "n"), "mutation.annotation.save"),
    ],
)
def test_mark_mutation_refuses_without_a_daemon(
    authority_archive: Path, verb_args: tuple[str, ...], operation: str
) -> None:
    """Each ``mark`` branch refuses by name and leaves ``user.db`` untouched."""
    before = _user_tier_digest(authority_archive)

    result = _run(authority_archive, "--no-daemon", "find", f"id:{_SESSION_ID}", "then", *verb_args)

    text = _refusal_text(result)
    assert result.exit_code != 0, result.output
    assert operation in text
    assert "polylogued run" in text
    assert _user_tier_digest(authority_archive) == before


def test_judge_accept_refuses_without_a_daemon(authority_archive: Path) -> None:
    """Recording a review is a declared write, so it refuses the same way.

    ``judge`` is a leaf command without the root ``--no-daemon`` flag, so the
    environment variable alone carries the daemon-off condition here.
    """
    before = _user_tier_digest(authority_archive)

    result = _run(authority_archive, "judge", "--accept", "assertion:candidate-authority-1")

    text = _refusal_text(result)
    assert result.exit_code != 0, result.output
    assert "mutation.judgment.record" in text
    assert "polylogued run" in text
    assert _user_tier_digest(authority_archive) == before
