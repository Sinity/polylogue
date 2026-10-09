"""One readiness census workspace retains exact per-Raw identity laws."""

import asyncio
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.revision_authority import (
    ParserCensusIdentityMeasurement,
    _measure_parser_census_identity,
    parser_census_identity_measurement,
    parser_census_identity_workspace,
)
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot
from polylogue.storage.raw_authority import iter_parser_census_logical_keys, raw_authority_parser_fingerprint
from polylogue.storage.sqlite import connection_profile
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
from tests.unit.storage.test_archive_readiness import _seed_rows


@pytest.mark.parametrize(
    ("raw_key", "members", "receipt", "non_session", "fragment", "expected"),
    [
        ("codex:n:exact tail", (), '["codex-session:n:exact tail"]', False, False, True),
        ("codex:key", (), '["codex:key", "codex-session:key"]', False, False, False),
        ("codex:key", ("codex:other",), '["codex:key"]', False, False, False),
        (None, ("invalid:key",), "[]", True, False, False),
        ("codex:key", (), '["codex:key", 17]', False, False, False),
        (None, (), "[]", False, False, False),
        (None, (), "[]", True, False, True),
        (None, (), "[]", False, True, True),
    ],
)
def test_shared_census_workspace_preserves_exact_measurement_and_clears_previous_raw(
    raw_key: str | None,
    members: tuple[str, ...],
    receipt: str,
    non_session: bool,
    fragment: bool,
    expected: bool,
) -> None:
    with parser_census_identity_measurement(
        raw_logical_key=raw_key,
        revision_kind="full",
        membership_logical_keys=members,
        observed_logical_keys=iter_parser_census_logical_keys(receipt),
        observed_are_receipt=True,
    ) as standalone:
        standalone_keys = tuple(standalone.iter_keys(observed=True))
        standalone_flags = (
            standalone.durable_valid,
            standalone.observed_valid,
            standalone.identities_match,
            standalone.observed_count,
        )
        assert (
            standalone.complete(
                typed_non_session=non_session, parser_confirmed_non_session=False, byte_governed_fragment=fragment
            )
            is expected
        )

    with parser_census_identity_workspace() as owner:
        first = _measure_parser_census_identity(
            owner,
            raw_logical_key="codex:previous",
            revision_kind="full",
            membership_logical_keys=(),
            observed_logical_keys=("codex:previous",),
        )
        with first.keys_json_stream(sqlite_encoding=True) as (byte_length, chunks):
            assert byte_length > 0 and b"previous" in b"".join(chunks)
        measured = _measure_parser_census_identity(
            owner,
            raw_logical_key=raw_key,
            revision_kind="full",
            membership_logical_keys=members,
            observed_logical_keys=iter_parser_census_logical_keys(receipt),
            observed_are_receipt=True,
        )
        assert tuple(measured.iter_keys(observed=True)) == standalone_keys
        assert (
            measured.durable_valid,
            measured.observed_valid,
            measured.identities_match,
            measured.observed_count,
        ) == standalone_flags
        assert (
            measured.complete(
                typed_non_session=non_session, parser_confirmed_non_session=False, byte_governed_fragment=fragment
            )
            is expected
        )
        with closing(owner.require_connection().execute("SELECT COUNT(*) FROM census_encoded_keys")) as rows:
            assert rows.fetchone()[0] == 0
    assert owner.connection is None


def _seed_census_projection(root: Path) -> None:
    initialize_active_archive_root(root)
    with sqlite3.connect(root / "source.db") as source:
        _seed_rows(
            source,
            "raw_sessions",
            ("raw_id", "logical_source_key", "revision_kind"),
            [("a-valid", "codex:valid", "full"), ("b-duplicate", "codex:duplicate", "full")],
        )
        source.executemany(
            "INSERT INTO raw_authority_parser_census"
            "(raw_id,parser_fingerprint,status,logical_keys_json,detail) VALUES (?,?,'complete',?,'')",
            [
                ("a-valid", raw_authority_parser_fingerprint(), '["codex-session:valid"]'),
                ("b-duplicate", raw_authority_parser_fingerprint(), '["codex:duplicate","codex:duplicate"]'),
            ],
        )


@pytest.mark.parametrize("empty", [False, True])
def test_readiness_projection_owns_one_physical_census_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    if empty:
        initialize_active_archive_root(tmp_path)
    else:
        _seed_census_projection(tmp_path)
    original = connection_profile.scratch_connection_context
    connections: list[sqlite3.Connection] = []

    @contextmanager
    def counted_workspace(*, prefix: str, filename: str, directory: Path | None = None) -> Iterator[sqlite3.Connection]:
        with original(prefix=prefix, filename=filename, directory=directory) as connection:
            if prefix == "polylogue-parser-census-":
                connections.append(connection)
            yield connection

    monkeypatch.setattr(connection_profile, "scratch_connection_context", counted_workspace)
    snapshot = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)
    census = cast(Mapping[str, object], snapshot["raw_authority_parser_census"])
    assert census["complete_count"] == (0 if empty else 1)
    assert census["incomplete_count"] == (0 if empty else 1)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize("cancelled", [False, True])
def test_census_workspace_settles_on_projection_exception_or_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    import threading

    import polylogue.storage.archive_readiness as readiness
    from polylogue.core.compute_cancel import compute_cancel

    _seed_census_projection(tmp_path)
    original = connection_profile.scratch_connection_context
    connections: list[sqlite3.Connection] = []
    measure = _measure_parser_census_identity
    measurements = 0
    cancellation = threading.Event()
    token = compute_cancel.set(cancellation)

    @contextmanager
    def counted_workspace(*, prefix: str, filename: str, directory: Path | None = None) -> Iterator[sqlite3.Connection]:
        with original(prefix=prefix, filename=filename, directory=directory) as connection:
            if prefix == "polylogue-parser-census-":
                connections.append(connection)
            yield connection

    def failed_second_measurement(
        owner: NativeSQLCustodyOwner,
        *,
        raw_logical_key: object,
        revision_kind: object,
        membership_logical_keys: Iterable[object],
        observed_logical_keys: Iterable[str] | None,
        observed_are_receipt: bool = False,
    ) -> ParserCensusIdentityMeasurement:
        nonlocal measurements
        measurements += 1
        if measurements == 2:
            if cancelled:
                cancellation.set()
            else:
                raise RuntimeError("synthetic projection failure")
        return measure(
            owner,
            raw_logical_key=raw_logical_key,
            revision_kind=revision_kind,
            membership_logical_keys=membership_logical_keys,
            observed_logical_keys=observed_logical_keys,
            observed_are_receipt=observed_are_receipt,
        )

    monkeypatch.setattr(connection_profile, "scratch_connection_context", counted_workspace)
    monkeypatch.setattr(readiness, "_measure_parser_census_identity", failed_second_measurement)
    try:
        if cancelled:
            with pytest.raises(asyncio.CancelledError, match="cancelled"):
                readiness.raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)
        else:
            snapshot = readiness.raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)
            assert snapshot["available"] is False and "projection failure" in str(snapshot["error"])
    finally:
        cancellation.clear()
        compute_cancel.reset(token)
    assert measurements == 2
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
