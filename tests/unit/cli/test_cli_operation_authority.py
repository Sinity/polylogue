"""No daemon means a typed, actionable refusal — never a second CLI writer.

The operator ruling behind step S10 is that there is no requirement for the
CLI to ever work standalone: every durable ``user.db`` write lowers to a
declared daemon operation, and with no daemon answering the command must
refuse in a way the operator can act on — naming the operation it could not
run and the command that makes it runnable (``polylogued run``).

Anti-vacuity: restoring a CLI-side writable ``ArchiveStore``, or adding a local
fallback inside ``submit_cli_mutation``, makes every command here exit 0 and
mutate ``user.db``, turning every test in this module red — both the exit-code
assertion and the byte-identical ``user.db`` digest. The matrix rows also watch
every ``sqlite3.connect`` in the process through the production interception
seam, so a local writer of *any* archive tier (``index.db``, ``ops.db``,
``embeddings.db``) is red even when its bytes happen to come out unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
    """Digest ``user.db`` and its journal: any local write changes it.

    A committed or pending write lands in the main file, in WAL frames, or in
    a rollback journal. The ``-shm`` WAL index and an empty ``-wal`` are what
    a read-only open of a WAL database creates (an excision plan and a
    materialization preview read the archive before submitting), so they are
    not evidence of a write and are left out.
    """
    digest = hashlib.sha256()
    for path in sorted(archive_root.glob("user.db*")):
        if path.name.endswith("-shm"):
            continue
        content = path.read_bytes()
        if path.name.endswith("-wal") and not content:
            continue
        digest.update(path.name.encode())
        digest.update(content)
    return digest.hexdigest()


@contextmanager
def _recording_writable_tier_opens() -> Iterator[list[Path]]:
    """Record every writable archive-tier open this process attempts.

    Uses :func:`~polylogue.maintenance.offline_guard.refuse_writable_tier_opens`,
    the seam the CLI's own ownership boundary installs, so "writable archive-tier
    open" is storage's definition rather than one restated here. The recorder
    does not raise: a command that writes is observed writing, not stopped
    part-way into a different failure.
    """
    from polylogue.maintenance.offline_guard import refuse_writable_tier_opens

    opened: list[Path] = []
    with refuse_writable_tier_opens(opened.append):
        yield opened


def _refusal_text(result: Result) -> str:
    return f"{result.output}\n{result.exception}"


def _run_machine(
    archive_root: Path,
    argv: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    format_flag: str = "--format",
) -> tuple[int, dict[str, object]]:
    """Run one invocation through the REAL machine entry and parse its envelope.

    ``CliRunner().invoke(cli, ...)`` is deliberately not used here:
    ``run_machine_entry`` is what maps an exception onto a machine error code,
    and it reads ``--format json`` off ``sys.argv``. Invoking the Click group
    directly skips the entire mapping, so a test that did so would assert a
    code no operator invocation ever produces.
    """
    full_argv = ["polylogue", *argv, format_flag, "json"]
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
        # ``mark`` lowers its query selection into the one resident
        # ``mutation.session.mark`` operation; with no daemon that operation
        # is what each composed branch names in its refusal.
        (("mark", "--star"), "mutation.session.mark"),
        (("mark", "--tag-add", "X"), "mutation.session.mark"),
        (("mark", "--tag-remove", "X"), "mutation.session.mark"),
        (("mark", "--note", "n"), "mutation.session.mark"),
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

#: One invocation per mutating operation the CLI submits, and the operation
#: each one must name when it refuses.  Hand-written because no declaration
#: knows what argv reaches a verb; kept honest by
#: :func:`test_the_matrix_covers_every_mutating_operation_the_cli_submits`,
#: which fails when CLI code starts submitting a mutating operation without a
#: row here.
#: A real provider export: ``import`` stages it and runs the admissibility
#: preflight before it reaches the daemon probe.
_IMPORTABLE_EXPORT = Path(__file__).parents[2] / "fixtures" / "origin-capability" / "codex-session.jsonl"

_MUTATING_INVOCATIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("import", ("import", str(_IMPORTABLE_EXPORT)), "ingest"),
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
    ("backup", ("ops", "backup", "--output-dir", "./backup-matrix"), "maintenance.backup"),
    (
        "restore-verified-backup",
        (
            "ops",
            "maintenance",
            "restore-verified-backup",
            "--backup-dir",
            str(_IMPORTABLE_EXPORT.parent),
            "--destination",
            "./restore-matrix",
        ),
        "maintenance.restore_verified_backup",
    ),
    ("embed-backfill", ("ops", "embed", "backfill", "--yes"), "maintenance.embeddings.backfill"),
    ("scan-secrets", ("ops", "scan-secrets", "--session", _SESSION_ID), "maintenance.secret_scan"),
    (
        "embed-resolve-failure",
        ("ops", "embed", "resolve-failure", "failure:missing", "--action", "requeue", "--yes"),
        "maintenance.embeddings.failure.resolve",
    ),
    (
        # ``reconcile-work-effects --yes`` submits the same operation; it needs
        # a stored graph to reconcile before it reaches the daemon probe.
        "materialize-incident-evidence",
        (
            "ops",
            "materialize-incident-evidence",
            "--session-id",
            _SESSION_ID,
            "--graph-id",
            "incident:authority-matrix",
            "--yes",
        ),
        "mutation.work_evidence.graph.replace",
    ),
    (
        "annotations-import",
        (
            "annotations",
            "import",
            # ``click.Path(exists=True)`` and the command's own UTF-8 decode
            # both run before the daemon probe, so the row needs a real,
            # valid-UTF8 file on disk -- this module's own source stands in;
            # its content is never parsed as JSONL because the refusal fires
            # first.
            __file__,
            "--batch-id",
            "b1",
            "--schema-id",
            "seed.activity",
            "--schema-version",
            "2",
            "--target-ref",
            f"session:{_SESSION_ID}",
            "--source-result-ref",
            "result-set:r",
            "--actor-ref",
            "agent:a",
            "--model-ref",
            "agent:m",
            "--prompt-ref",
            "block:p:0",
        ),
        "mutation.annotation.import_batch",
    ),
    # The three maintenance rows submit before they read anything, so an
    # unknown id reaches the daemon probe exactly as a live one would.
    (
        "blob-publications-abandon",
        ("ops", "maintenance", "blob-publications", "--abandon", "publication:matrix", "--yes"),
        "maintenance.blob-publications.abandon",
    ),
    (
        "raw-authority-blocker-resolve",
        (
            "ops",
            "maintenance",
            "raw-authority-blocker-resolve",
            "--blocker-id",
            "raw-authority-blocker:matrix",
            "--reason",
            "r",
            "--yes",
        ),
        "mutation.raw-authority-blocker.resolve",
    ),
    (
        "raw-authority-frontier",
        ("ops", "maintenance", "raw-authority-frontier"),
        "maintenance.raw-authority-frontier",
    ),
    ("reset-tier", ("ops", "reset", "--index", "--yes"), "maintenance.reset"),
)

#: The machine-format flag of each row whose command spells it differently
#: from ``--format``; ``None`` marks a command with no machine envelope.
_MACHINE_FORMAT_FLAG: Mapping[str, str | None] = {
    "blob-publications-abandon": "--output-format",
    "raw-authority-blocker-resolve": "--output-format",
    "raw-authority-frontier": "--output-format",
    # `ops reset` accepts --format/--json only alongside --session/--source,
    # so its tier-reset branch has no machine envelope to carry a code in; the
    # terminal row above still proves the branch refuses without writing.
    "reset-tier": None,
}

#: Mutating operations the CLI submits whose route cannot reach the daemon
#: probe from a daemon-down invocation, with the reason.
_MATRIX_EXEMPT: Mapping[str, str] = {
    "maintenance.demo.augment": (
        "submitted only after `import --demo --wait` saw its ingest complete, and that ingest already needs the daemon"
    ),
    "maintenance.schema.quarantine": (
        "submitted by `ops doctor --schemas --schema-quarantine-malformed` only after verification "
        "found a malformed raw row, which the matrix archive does not hold"
    ),
    "mutation.facade.context_ledger": (
        "a best-effort receipt the `read context` views submit after a read the daemon already served"
    ),
    "operation.result": (
        "pages the retained receipt of an `ops excise` the daemon already accepted; it reruns and writes nothing"
    ),
}


def _cli_submitted_mutations() -> frozenset[str]:
    """Every declared mutating operation that ``polylogue/cli`` names as a literal.

    The CLI submits operations by name at its call sites, so the names spelled
    in its source are the real inventory of what it can ask the daemon to
    write -- no registry beside the code can drift from it.
    """
    import ast

    import polylogue.cli
    from polylogue.operations.daemon_protocol import MUTATION_OPERATION_NAMES

    names: set[str] = set()
    for path in Path(polylogue.cli.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in MUTATION_OPERATION_NAMES
            ):
                names.add(node.value)
    return frozenset(names)


def test_the_matrix_covers_every_mutating_operation_the_cli_submits() -> None:
    """A new mutating CLI route cannot be added without a daemon-down row.

    Anti-vacuity: submit a mutating operation from a new CLI command without
    adding a row to ``_MUTATING_INVOCATIONS`` (or ``_MATRIX_EXEMPT``) and this
    is red. Without it the matrix below silently stops covering the newest
    writer, which is precisely how a bypass survives a green suite.
    """
    submitted = _cli_submitted_mutations()
    covered = {operation for _, _, operation in _MUTATING_INVOCATIONS}
    uncovered = {
        operation
        for operation in submitted
        if operation not in _MATRIX_EXEMPT and not any(operation.startswith(prefix) for prefix in covered)
    }
    assert not uncovered, uncovered
    # The exemption list may not outlive its entries either.
    assert set(_MATRIX_EXEMPT) <= submitted, set(_MATRIX_EXEMPT) - submitted
    assert set(_MACHINE_FORMAT_FLAG) <= {name for name, _, _ in _MUTATING_INVOCATIONS}


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
    drop ``polylogued run`` from ``mutation_refusal`` and every row goes red
    on the remedy assertion while still exiting non-zero, which is the failure
    mode worth separating -- a refusal with no next action is barely better
    than a traceback.
    """
    del name
    before = _user_tier_digest(authority_archive)

    with _recording_writable_tier_opens() as opened:
        result = _run(authority_archive, *argv)

    text = _refusal_text(result)
    assert result.exit_code != 0, result.output
    assert "polylogued run" in text, text
    assert "Traceback" not in result.output, result.output
    assert _user_tier_digest(authority_archive) == before
    assert opened == [], f"{operation} opened writable archive tiers in the CLI process: {opened}"


_MACHINE_INVOCATIONS = tuple(row for row in _MUTATING_INVOCATIONS if _MACHINE_FORMAT_FLAG.get(row[0], "--format"))


@pytest.mark.parametrize(
    ("name", "argv", "operation"),
    _MACHINE_INVOCATIONS,
    ids=[row[0] for row in _MACHINE_INVOCATIONS],
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
    ``mutation_refusal``) and every row here goes red on the ``code``
    assertion while the terminal test above stays green -- the exact asymmetry
    that let the gap survive.
    """
    flag = _MACHINE_FORMAT_FLAG.get(name) or "--format"
    before = _user_tier_digest(authority_archive)

    with _recording_writable_tier_opens() as opened:
        exit_code, payload = _run_machine(authority_archive, argv, monkeypatch, capsys, format_flag=flag)

    assert exit_code != 0, payload
    assert payload["status"] == "error", payload
    assert payload["code"] == "daemon_required", payload
    assert "polylogued run" in str(payload["message"]), payload
    details = payload.get("details")
    assert isinstance(details, dict) and details.get("remedy") == "polylogued run", payload
    assert _user_tier_digest(authority_archive) == before
    assert opened == [], f"{operation} opened writable archive tiers in the CLI process: {opened}"


@pytest.mark.parametrize(
    "argv",
    [("demo", "verify"), ("ops", "maintenance", "gc-history")],
    ids=["demo-verify", "gc-history"],
)
def test_read_only_commands_open_no_writable_tier(authority_archive: Path, argv: tuple[str, ...]) -> None:
    """A verification or history view reads; it never opens a tier for writing.

    Both opened plain ``sqlite3.connect``/write-profile connections, which
    create a missing tier file and contend with the daemon's writer, and a
    sweep of every CLI leaf command found them as the only read commands that
    did so.

    Anti-vacuity: open ``demo/verify.py``'s connections with
    ``sqlite3.connect(path)`` again (or ``read_gc_history`` with
    ``open_connection``) and the recorded opens are non-empty.
    """
    with _recording_writable_tier_opens() as opened:
        _run(authority_archive, *argv)

    assert opened == [], opened
