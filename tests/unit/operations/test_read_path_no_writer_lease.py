"""A pure read must not take the writer lease nor write to the archive root.

Regression surface for the #4867 read-path regression: ``open_operation_read``
fell through to ``OwnedArchiveLocation.acquire`` whenever no publication guard
was supplied, and the CLI read adapter minted the daemon bearer token before
knowing whether a daemon was listening. Both made a read mutate -- or outright
fail against -- the archive it was reading.

Anti-vacuity for each test is named in its docstring: every assertion here goes
red if the exclusive acquisition or the eager mint is reintroduced, and none of
them is satisfied by the five CLI snapshot tests merely passing.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from tests.infra.archive_templates import bootstrap_archive_root

OWNERSHIP_LOCK = ".archive-ownership.lock"


def _root_entries(root: Path) -> set[str]:
    return {entry.name for entry in root.iterdir()}


def test_unguarded_operation_read_never_acquires_the_writer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Red if the direct read path calls ``OwnedArchiveLocation.acquire`` again.

    The lease is the *maintenance/campaign writer* proof. A reader taking it
    is the defect, so the strongest statement is that the constructor is never
    reached at all -- not that the read happens to succeed.
    """
    bootstrap_archive_root(tmp_path)

    def refuse(location: ArchiveLocation) -> OwnedArchiveLocation:
        raise AssertionError("a read acquired the exclusive maintenance-writer lease")

    monkeypatch.setattr(OwnedArchiveLocation, "acquire", staticmethod(refuse))
    with open_operation_read(tmp_path) as pinned:
        assert pinned.archive.source_connection is not None


def test_unguarded_operation_read_writes_nothing_into_the_archive_root(tmp_path: Path) -> None:
    """Red if the read creates ``.archive-ownership.lock`` (or anything else) in the root.

    A read that mutates the thing it reads is the second harm. Comparing the
    whole directory listing, not just the lock name, also catches a future
    reader that leaves some other bookkeeping file behind.
    """
    bootstrap_archive_root(tmp_path)
    # ``bootstrap_archive_root`` is a legitimate *writer* and leaves its own
    # lease record behind; the property under test is that a subsequent READ
    # neither adds root entries nor rewrites that record. The old path called
    # ``ftruncate`` + ``write`` on it, so the sentinel is what goes red.
    lock_path = tmp_path / OWNERSHIP_LOCK
    lock_path.write_bytes(b"pid=previous-writer")
    before = _root_entries(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        assert pinned.archive.source_connection is not None
    assert _root_entries(tmp_path) == before
    assert lock_path.read_bytes() == b"pid=previous-writer"


def test_operation_read_succeeds_while_another_process_owns_the_archive(tmp_path: Path) -> None:
    """Red if the read contends for the lease held by a legitimate owner.

    ``flock`` is per open-file-description, so this second descriptor conflicts
    exactly as a separate daemon or migrate-tier process would. This is the
    campaign rollback case: reading a preserved aside archive that something
    else legitimately owns.
    """
    bootstrap_archive_root(tmp_path)
    lock_path = tmp_path / OWNERSHIP_LOCK
    owner_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(owner_fd, b"pid=foreign-owner")
        with open_operation_read(tmp_path) as pinned:
            assert pinned.archive.source_connection is not None
        # The foreign owner's record survives: the read neither truncated nor
        # rewrote the lease file.
        assert os.pread(owner_fd, 64, 0) == b"pid=foreign-owner"
    finally:
        os.close(owner_fd)


def test_operation_read_succeeds_against_a_read_only_archive_root(tmp_path: Path) -> None:
    """Red if any part of the read needs write access to the root or its tiers.

    Mode 0500 on the root and 0400 on the tier files is the shape the harness
    seals ``.bootstrap-archive-template`` directories with, and the shape a
    preserved rollback archive has.
    """
    root = tmp_path / "sealed"
    root.mkdir()
    bootstrap_archive_root(root)
    # A preserved rollback archive carries no live lease record; drop the one
    # the bootstrap writer left so the assertion below is about this read.
    (root / OWNERSHIP_LOCK).unlink(missing_ok=True)
    for entry in root.iterdir():
        if entry.is_file():
            entry.chmod(0o400)
    root.chmod(0o500)
    try:
        with open_operation_read(root) as pinned:
            assert pinned.archive.source_connection is not None
        assert OWNERSHIP_LOCK not in _root_entries(root)
    finally:
        root.chmod(0o700)
        for entry in root.iterdir():
            if entry.is_file():
                entry.chmod(0o600)


def test_cli_read_verb_mints_no_api_token_when_no_daemon_is_listening(
    tmp_path: Path,
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Red if the bearer token is resolved before a daemon socket answers.

    Deliberately does NOT patch ``load_or_mint_api_auth_token`` out (the CLI
    snapshot module does, which is why it never caught this): the mint is
    replaced by a tripwire. With the eager ``resolve_api_auth_token`` call
    restored in ``configured_read_operation`` this fails on the tripwire, and
    in production it failed with ``PermissionError`` on a read-only root.
    """
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    bootstrap_archive_root(tmp_path)

    def tripwire(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("a CLI read minted the daemon API token with no daemon listening")

    monkeypatch.setattr("polylogue.daemon.api_auth.load_or_mint_api_auth_token", tripwire)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    before = _root_entries(tmp_path)

    result = CliRunner().invoke(cli, ["--plain", "read", "--all"], catch_exceptions=False)

    # An empty archive legitimately exits 2 (``empty``); what must not happen
    # is a credential write. Anything above 2 is a real failure.
    assert result.exit_code in {0, 2}, result.output
    assert _root_entries(tmp_path) == before
