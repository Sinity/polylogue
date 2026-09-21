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
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from polylogue.cli.machine_main import run_machine_entry
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


def _run_machine(
    archive_root: Path, argv: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, object]]:
    """Run one invocation through the REAL machine entry and parse its envelope.

    ``CliRunner().invoke(cli, ...)`` is deliberately not used here:
    ``run_machine_entry`` is what maps an exception onto a machine error code,
    and it reads ``--format json`` off ``sys.argv``. Invoking the Click group
    directly skips the entire mapping, so a test that did so would assert a
    code no operator invocation ever produces.
    """
    full_argv = ["polylogue", *argv, "--format", "json"]
    monkeypatch.setattr(sys, "argv", full_argv)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_DB_PATH", str(archive_root / "index.db"))
    monkeypatch.setenv("POLYLOGUE_NO_DAEMON", "1")
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_info:
        run_machine_entry(cli, full_argv[1:])
    stdout = capsys.readouterr().out
    code = exit_info.value.code
    payload = json.loads(stdout)
    assert isinstance(payload, dict), stdout
    return (code if isinstance(code, int) else 1), payload


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


# --------------------------------------------------------------------------
# Daemon-down matrix over every CLI-bound mutating operation
# --------------------------------------------------------------------------

#: One invocation per CLI-bound mutating operation, and the operation each one
#: must name when it refuses.  Hand-written because no declaration knows what
#: argv reaches a verb; kept honest by
#: :func:`test_the_matrix_covers_every_cli_bound_mutating_operation`, which
#: fails when a new mutating operation gains a CLI binding without a row here.
_MUTATING_INVOCATIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("mark-star", ("find", f"id:{_SESSION_ID}", "then", "mark", "--star"), "mutation.session.mark"),
    ("mark-tag-add", ("find", f"id:{_SESSION_ID}", "then", "mark", "--tag-add", "X"), "mutation.session.tag"),
    ("mark-note", ("find", f"id:{_SESSION_ID}", "then", "mark", "--note", "n"), "mutation.annotation.save"),
    ("root-set-meta", ("--set", "lane", "triage", "find", f"id:{_SESSION_ID}"), "mutation.session.metadata"),
    ("root-add-tag", ("--add-tag", "triage", "find", f"id:{_SESSION_ID}"), "mutation.session.tag"),
    ("delete", ("find", f"id:{_SESSION_ID}", "then", "delete", "--yes"), "mutation.session.delete"),
    ("note", ("note", "a terminal note"), "mutation.assertion.candidate.capture"),
    ("setting-set", ("setting", "set", "subscription_tier", "max"), "mutation.user.setting.set"),
    ("judge", ("judge", "--accept", "assertion:candidate-authority-1"), "mutation.judgment.record"),
    (
        "excise",
        ("ops", "excise", "--session", _SESSION_ID, "--reason", "r", "--actor", "user:local", "--yes"),
        "mutation.session",
    ),
    ("reset-identity", ("ops", "reset", "--session", _SESSION_ID, "--yes"), "mutation.identity-reset"),
)

#: Mutating operations whose CLI route refuses before it reaches the daemon
#: probe, so a daemon-down run never produces the ``daemon_required`` refusal.
#: Each needs an argument this matrix cannot synthesize (a manifest on disk, a
#: staged export, a live blob-GC generation id), so the row would be asserting
#: the argument check rather than the authority check.
_MATRIX_EXEMPT: Mapping[str, str] = {
    "ingest": "needs a real staged export path; `import` refuses an absent path first",
    "maintenance.demo.augment": "only reachable behind `import --demo`, which seeds a fixture world first",
    "maintenance.blob-gc.recover": "needs a live pending GC generation id read from source.db",
    "maintenance.blob-publications.abandon": "needs live publication ids read from source.db",
    "maintenance.blob-refs.replace-from-source": "needs a manifest file produced by a prior scan",
    "maintenance.blob-refs.prune-orphans": "needs a quarantine path and a prior orphan scan",
    "mutation.raw-authority-blocker.resolve": "needs a live blocker id read from source.db",
    # `ops reset` accepts --format/--json only alongside --session/--source,
    # so its tier-reset branch has no machine envelope to carry a code in. The
    # identity-reset branch IS covered above, which is what proves the shared
    # translator reached this command.
    "maintenance.reset": "the tier-reset branch of `ops reset` has no --format json route to assert a code on",
}


def test_the_matrix_covers_every_cli_bound_mutating_operation() -> None:
    """A new mutating CLI route cannot be added without a daemon-down row.

    Anti-vacuity: bind a mutating operation in ``CLI_OPERATION_BINDINGS``
    without adding it to ``_MUTATING_INVOCATIONS`` or ``_MATRIX_EXEMPT`` and
    this is red. Without it the matrix below silently stops covering the newest
    writer, which is precisely how a bypass survives a green suite.
    """
    from polylogue.cli.operation_bindings import CLI_OPERATION_BINDINGS
    from polylogue.operations.daemon_protocol import MUTATION_OPERATION_NAMES

    bound_mutations = set(CLI_OPERATION_BINDINGS) & set(MUTATION_OPERATION_NAMES)
    covered = {operation for _, _, operation in _MUTATING_INVOCATIONS}
    uncovered = {
        operation
        for operation in bound_mutations
        if operation not in _MATRIX_EXEMPT and not any(operation.startswith(prefix) for prefix in covered)
    }
    assert not uncovered, uncovered
    # The exemption list may not outlive its entries either.
    assert set(_MATRIX_EXEMPT) <= bound_mutations, set(_MATRIX_EXEMPT) - bound_mutations


@pytest.mark.parametrize(
    ("name", "argv", "operation"),
    _MUTATING_INVOCATIONS,
    ids=[row[0] for row in _MUTATING_INVOCATIONS],
)
def test_daemon_down_refusal_names_polylogued_run_in_terminal_format(
    authority_archive: Path, name: str, argv: tuple[str, ...], operation: str
) -> None:
    """Every mutating verb refuses by name, never as a traceback or a no-op.

    Anti-vacuity: give ``submit_cli_mutation`` a local writer and these exit 0;
    drop ``polylogued run`` from ``_mutation_refusal`` and every row goes red
    on the remedy assertion while still exiting non-zero, which is the failure
    mode worth separating -- a refusal with no next action is barely better
    than a traceback.
    """
    del name
    before = _user_tier_digest(authority_archive)

    result = _run(authority_archive, *argv)

    text = _refusal_text(result)
    assert result.exit_code != 0, result.output
    assert "polylogued run" in text, text
    assert "Traceback" not in result.output, result.output
    assert _user_tier_digest(authority_archive) == before


@pytest.mark.parametrize(
    ("name", "argv", "operation"),
    _MUTATING_INVOCATIONS,
    ids=[row[0] for row in _MUTATING_INVOCATIONS],
)
def test_daemon_down_refusal_is_typed_daemon_required_in_machine_format(
    authority_archive: Path,
    name: str,
    argv: tuple[str, ...],
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--format json`` reports ``daemon_required``, not ``runtime_error``.

    ``machine_errors`` carried no ``daemon_required`` code at all, so every
    daemon-absent refusal reached a machine caller as ``runtime_error`` -- the
    same code an unexpected exception and a corrupt tier produce -- and the
    remedy the terminal format prints was simply absent from the wire
    (polylogue-re6s3 AC4, polylogue-3eexy AC4).

    Anti-vacuity: route ``DaemonRequiredError`` back through ``error_runtime``
    in ``machine_main`` (or drop the typed exception from
    ``_mutation_refusal``) and every row here goes red on the ``code``
    assertion while the terminal test above stays green -- the exact asymmetry
    that let the gap survive.
    """
    del name, operation
    before = _user_tier_digest(authority_archive)

    exit_code, payload = _run_machine(authority_archive, argv, monkeypatch, capsys)

    assert exit_code != 0, payload
    assert payload["status"] == "error", payload
    assert payload["code"] == "daemon_required", payload
    assert "polylogued run" in str(payload["message"]), payload
    details = payload.get("details")
    assert isinstance(details, dict) and details.get("remedy") == "polylogued run", payload
    assert _user_tier_digest(authority_archive) == before
