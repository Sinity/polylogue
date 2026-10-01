"""Domain inspection results from one physically bound SQLite read transaction."""

from __future__ import annotations

import errno
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.core.enums import SOURCE_FIDELITY_STATUS_VALUES, Provider
from polylogue.sources.parsers import antigravity, hermes_identity, hermes_state, hermes_verification
from polylogue.sources.parsers.base import ParsedSession


@dataclass(frozen=True, slots=True)
class SQLiteInspection:
    """Preview evidence only; parsed transcript bodies never cross the worker pipe."""

    domain: str | None
    produced: dict[str, Any]
    admitted: int = 0
    degraded: bool = False
    fidelity: hermes_state.HermesImportFidelity | None = None


def _inspect_connection(
    conn: sqlite3.Connection,
    path: Path,
    *,
    preflight: bool,
    grouping: sqlite3.Connection | None = None,
    profile_identity: str | None = None,
) -> SQLiteInspection:
    conn.text_factory = str
    conn.row_factory = sqlite3.Row
    fidelity = None
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"} and antigravity._trajectory_schema_matches(conn):
        domain = "antigravity_trajectory_db"
        sessions: Iterable[ParsedSession] = antigravity._parse_trajectory_connection(conn, path, path.stem)
    elif not preflight and hermes_state._has_required_tables(conn):
        domain = "hermes_state_db"
        state_sessions = hermes_state._parse_state_connection(
            conn, path, profile_root=hermes_identity.profile_root_for_artifact(path), profile_identity=profile_identity
        )
        fidelity = hermes_state.import_fidelity_declaration(state_sessions, acquisition_method="logical_export")
        sessions = state_sessions
    elif not preflight and hermes_verification._has_required_tables(conn):
        domain = "hermes_verification_evidence_db"
        if grouping is None:
            raise OSError(errno.EPROTO, "verification inspection has no private grouping store")
        produced, fidelity = hermes_verification._inspect_verification_connection(
            conn,
            grouping,
            path,
            profile_root=hermes_identity.profile_root_for_artifact(path),
            profile_identity=profile_identity,
        )
        return SQLiteInspection(domain, produced, degraded=bool(produced["sessions"]), fidelity=fidelity)
    else:
        return SQLiteInspection(None, {})
    produced: dict[str, Any] = {
        "sessions": 0,
        "messages": 0,
        "blocks": 0,
        "actions": 0,
        "raw_records": 0,
        "session_refs": [],
    }
    admitted = 0
    degraded = False
    for session in sessions:
        produced["sessions"] += 1
        produced["raw_records"] += 1
        if not preflight:
            produced["session_refs"].append(f"session:{session.source_name.value}:{session.provider_session_id}")
        produced["messages"] += len(session.messages)
        for message in session.messages:
            produced["blocks"] += len(message.blocks)
            produced["actions"] += sum(block.type.value == "tool_use" for block in message.blocks)
        degraded |= bool(session.ingest_flags)
        if preflight:
            from polylogue.sources.dispatch import require_positive_conversational_evidence

            positive = require_positive_conversational_evidence(
                [session], provider=Provider.ANTIGRAVITY, source_path=str(path)
            )
            admitted += len(positive)
            degraded |= not positive
    return SQLiteInspection(domain, produced, admitted, degraded, fidelity)


def _decode_inspection(value: dict[str, Any], *, preflight: bool = False) -> SQLiteInspection:
    """Validate the private protocol before exposing a successful preview."""
    try:
        if set(value) != {"domain", "produced", "admitted", "degraded", "fidelity"}:
            raise ValueError("unexpected inspection fields")
        domain = value["domain"]
        if domain not in {None, "antigravity_trajectory_db", "hermes_state_db", "hermes_verification_evidence_db"}:
            raise ValueError("unexpected inspection domain")
        produced = value["produced"]
        if not isinstance(produced, dict):
            raise ValueError("invalid produced evidence")
        if domain is not None:
            if set(produced) != {"sessions", "messages", "blocks", "actions", "raw_records", "session_refs"}:
                raise ValueError("invalid produced fields")
            if any(
                type(produced[name]) is not int or produced[name] < 0 for name in produced if name != "session_refs"
            ):
                raise ValueError("invalid produced counts")
            refs = produced["session_refs"]
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                raise ValueError("invalid session references")
            if len(refs) != (0 if preflight else produced["sessions"]):
                raise ValueError("inconsistent session references")
        elif produced:
            raise ValueError("unmatched inspection produced evidence")
        if type(value["admitted"]) is not int or value["admitted"] < 0 or type(value["degraded"]) is not bool:
            raise ValueError("invalid admission evidence")
        if value["admitted"] > produced.get("sessions", 0) or produced.get("raw_records", 0) != produced.get(
            "sessions", 0
        ):
            raise ValueError("inconsistent admission evidence")
        if domain is None and (value["admitted"] or value["degraded"]):
            raise ValueError("unmatched inspection has admission evidence")
        fidelity_value = value["fidelity"]
        if (domain in {"hermes_state_db", "hermes_verification_evidence_db"}) != (fidelity_value is not None):
            raise ValueError("inconsistent inspection fidelity")
        fidelity = None
        if fidelity_value is not None:

            def capability(payload: dict[str, Any]) -> hermes_state.HermesFidelityCapability:
                if set(payload) != {"status", "observed", "expected", "counts", "detail"}:
                    raise ValueError("invalid fidelity capability fields")
                if payload["status"] not in SOURCE_FIDELITY_STATUS_VALUES or not isinstance(payload["detail"], str):
                    raise ValueError("invalid fidelity capability")
                if any(type(payload[name]) is not int or payload[name] < 0 for name in ("observed", "expected")):
                    raise ValueError("invalid fidelity counts")
                counts = payload["counts"]
                if not isinstance(counts, dict) or any(
                    not isinstance(name, str) or type(count) is not int or count < 0 for name, count in counts.items()
                ):
                    raise ValueError("invalid fidelity count map")
                return hermes_state.HermesFidelityCapability(**payload)

            fidelity = hermes_state.HermesImportFidelity(
                **{
                    key: item
                    for key, item in fidelity_value.items()
                    if key not in {"retained_blob_reproducibility", "capabilities", "caveats"}
                },
                retained_blob_reproducibility=capability(fidelity_value["retained_blob_reproducibility"]),
                capabilities={name: capability(item) for name, item in fidelity_value["capabilities"].items()},
                caveats=tuple(fidelity_value["caveats"]),
            )
        return SQLiteInspection(domain, produced, value["admitted"], value["degraded"], fidelity)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise OSError(errno.EPROTO, "invalid SQLite inspection result") from exc


def inspect_sqlite_source(path: Path, *, preflight: bool = False) -> SQLiteInspection:
    """Inspect supported provider domains on one bound connection and snapshot."""
    from polylogue.sources.sqlite_export import _run_source_worker

    operation = "inspect_preflight" if preflight else "inspect_explain"
    return _decode_inspection(_run_source_worker(path, operation), preflight=preflight)
