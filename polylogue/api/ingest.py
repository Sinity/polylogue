"""Daemon-backed ingest methods for the async Polylogue facade."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Source, active_archive_root
from polylogue.core.errors import PolylogueError
from polylogue.pipeline.services.parsing_models import ParseResult

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.storage.repository import SessionRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend


class IngestDaemonRequiredError(PolylogueError):
    """The archive's accepted ingest owner is unavailable."""

    code = "daemon_required"


class PolylogueIngestMixin:
    if TYPE_CHECKING:

        @property
        def config(self) -> Config: ...

        @property
        def backend(self) -> SQLiteBackend: ...

        @property
        def repository(self) -> SessionRepository: ...

    async def parse_file(self, path: str | Path, *, source_name: str | None = None) -> ParseResult:
        file_path = Path(path).expanduser().resolve()
        return await self.parse_sources([Source(name=source_name or file_path.stem, path=file_path)])

    async def parse_sources(self, sources: list[Source] | None = None) -> ParseResult:
        """Submit local sources to the resident daemon's accepted ingest owner."""
        from polylogue.daemon.api_auth import resolve_api_auth_token
        from polylogue.daemon.socket_path import daemon_socket_path
        from polylogue.daemon_client import DaemonClient
        from polylogue.operations.daemon_protocol import daemon_operation_spec

        selected = self.config.sources if sources is None else sources
        local_sources = [source for source in selected if source.path is not None]
        result = ParseResult()
        if not local_sources:
            return result
        root = active_archive_root(self.config)
        spec = daemon_operation_spec("ingest")
        assert spec is not None
        client = DaemonClient(
            daemon_socket_path(root),
            timeout_s=spec.deadline_s,
            auth_token=lambda: resolve_api_auth_token(
                getattr(self.config, "api_auth_token", None),
                allow_no_auth=getattr(self.config, "api_allow_no_auth", False),
            ),
        )
        for source in local_sources:
            assert source.path is not None
            path = source.path.expanduser().resolve()
            envelope = await asyncio.to_thread(
                client.operation_to_completion,
                "ingest",
                {
                    "path": str(path),
                    "source_path": str(path) if path.is_file() else None,
                    "source_name": source.name,
                    "idempotency_key": None,
                },
                archive_root=str(root),
            )
            if envelope is None:
                raise IngestDaemonRequiredError(
                    f"start `polylogued run` for {root} and submit the accepted `ingest` operation"
                )
            if envelope.get("outcome") != "completed":
                raise RuntimeError(f"daemon ingest did not complete: {envelope.get('outcome')}")
            body = envelope.get("result")
            if not isinstance(body, dict):
                raise RuntimeError("daemon ingest returned no terminal receipt")
            history = body.get("historical_receipt")
            if not isinstance(history, dict):
                raise RuntimeError("daemon ingest returned no historical receipt")
            summary = history.get("summary")
            if not isinstance(summary, dict) or not summary.get("enumeration_complete"):
                raise RuntimeError("daemon ingest enumeration did not complete")
            if not summary.get("parse_projection_known"):
                raise RuntimeError("daemon ingest terminal receipt lacks parse projection")
            session_ids = [str(session_id) for session_id in summary["processed_session_ids"]]
            pages_ref = summary.get("processed_session_id_pages_ref")
            if pages_ref is not None:
                from polylogue.operations.audit import AuditRepository

                audit = AuditRepository(root / "audit.db")
                with audit.settled_machine_read():
                    session_ids = audit.read_ingest_session_id_pages(
                        str(pages_ref),
                        page_count=int(summary["processed_session_id_page_count"]),
                        session_count=int(summary["changed_session_count"]),
                        digest=str(summary["processed_session_ids_digest"]),
                    )
            message_count = int(summary["processed_message_count"])
            changed_sessions = int(summary["changed_session_count"])
            changed_messages = int(summary["changed_message_count"])
            result.processed_ids.update(session_ids)
            result.counts["sessions"] += len(session_ids)
            result.counts["messages"] += message_count
            result.changed_counts["sessions"] += changed_sessions
            result.changed_counts["messages"] += changed_messages
            result.parse_failures += int(summary.get("refused_membership_count", 0))
            result.parse_failures += int(summary.get("unresolved_raw_count", 0))
            result.batch_observations.append(
                {
                    "primary_ingest_store": "archive_file_set",
                    "archive_primary_write": True,
                    "archive_write_mode": "archive",
                    "archive_root": str(root),
                    "archive_write_targets": ["source.db", "index.db"],
                    "archive_source_rows": int(summary["confirmed_raw_count"]),
                    "archive_index_rows": changed_sessions,
                    "sessions": len(session_ids),
                    "messages": message_count,
                    "changed_sessions": changed_sessions,
                    "failed_raw_count": int(summary.get("unresolved_raw_count", 0)),
                }
            )
        return result


__all__ = ["IngestDaemonRequiredError", "PolylogueIngestMixin"]
