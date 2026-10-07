"""SQLite provider parsing into the caller's complete preparation store."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from contextlib import closing
from pathlib import Path

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.sources.parse_accounting_spool import SqliteParseAccountingWriter
from polylogue.sources.parsers import antigravity, hermes_state
from polylogue.sources.parsers.base import AdmissionOutcome, AdmissionUnit, ParseAccounting, ParsedSession
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.sources.streamed_event_payload import iter_json_value


class _AccountingBuilder:
    def __init__(self, store: SqliteMessageStore, expected: dict[AdmissionUnit, int]) -> None:
        self.expected = expected
        writer_expected: dict[object, int] = {}
        for unit, count in expected.items():
            writer_expected[unit] = count
        self.writer = SqliteParseAccountingWriter(store.conn, writer_expected)

    def append(self, outcome: AdmissionOutcome) -> None:
        self.writer.append(outcome)

    def finish(self) -> ParseAccounting:
        return ParseAccounting.model_construct(expected=self.expected, outcomes=self.writer.finish())


def iter_sqlite_sessions(
    provider: Provider,
    path: Path,
    store: SqliteMessageStore,
    *,
    fallback_id: str,
    profile_root: Path | None = None,
    profile_identity: str | None = None,
) -> Generator[ParsedSession, None, None]:
    """Keep all parser streams on the preparation owner's live connection.

    The caller consumes, hashes, and writes each session before closing its
    store. Parent-reference arrays and accounting borrow that same connection.
    """
    if provider is Provider.ANTIGRAVITY:
        yield from antigravity.parse_trajectory_db(
            path,
            fallback_id=fallback_id,
            immutable=True,
            grouping=store.conn,
            message_sink_factory=store.new_sink,
            event_sink_factory=store.new_event_sink,
            accounting_factory=lambda expected: _AccountingBuilder(store, expected),
            check_cancelled=check_compute_cancelled,
        )
    elif provider is Provider.HERMES:
        with hermes_state._readonly_context(path, immutable=True) as connection:
            yield from hermes_state.iter_state_db_sessions(
                connection,
                path,
                message_sink_factory=store.new_sink,
                event_sink_factory=store.new_event_sink,
                profile_root=profile_root,
                profile_identity=profile_identity,
                check_cancelled=check_compute_cancelled,
            )
    else:
        raise ValueError(f"SQLite session parsing is not supported for {provider}")


def collect_sqlite_sessions(
    provider: Provider,
    path: Path,
    *,
    fallback_id: str,
    profile_identity: str | None = None,
) -> list[ParsedSession]:
    """Collect the explicit resident replay API before its scratch closes.

    Production preparation and live acquisition use ``iter_sqlite_sessions``.
    This API returns independent resident models, including every event array
    and admission outcome, to callers that explicitly request a session list.
    """
    with (
        tempfile.TemporaryDirectory(prefix="polylogue-sqlite-collect-") as directory,
        closing(SqliteMessageStore(Path(directory) / "sessions.db")) as store,
    ):
        result = []
        for session in iter_sqlite_sessions(
            provider, path, store, fallback_id=fallback_id, profile_identity=profile_identity
        ):
            result.append(resident_sqlite_session(session))
        return result


def resident_sqlite_session(session: ParsedSession) -> ParsedSession:
    """Copy every borrowed stream for the independent-model public API."""
    events = [
        event.model_copy(update={"payload": json.loads("".join(iter_json_value(event.payload, ensure_ascii=False)))})
        for event in session.session_events
    ]
    accounting = session.unit_accounting
    if accounting is not None:
        accounting = ParseAccounting.model_construct(expected=accounting.expected, outcomes=list(accounting.outcomes))
    return session.model_copy(
        update={"messages": list(session.messages), "session_events": events, "unit_accounting": accounting}
    )
