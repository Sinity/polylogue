"""Bounded read-only AgentCTL owner adapter for the Polylogue archive."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from typing import Any

from polylogue.archive.query.execution_control import InterruptibleSQLiteRead
from polylogue.archive.query.transaction import QueryTransaction, QueryTransactionRequest
from polylogue.mcp.payloads import MCPArchiveStatsPayload
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

OWNER = "polylogue-archive"
OPERATION = "polylogue.archive.status"
SOURCE_REF = "sinnix://polylogue/archive"
SCHEMA = 1
INLINE_PAYLOAD_LIMIT_BYTES = 262_144


class AdapterError(ValueError):
    """One stable, protocol-visible failure from the owner adapter."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _inline(value: Any) -> dict[str, Any]:
    if len(_canonical_json(value)) > INLINE_PAYLOAD_LIMIT_BYTES:
        raise AdapterError("RESOURCE_EXHAUSTED", "archive status exceeds the inline response limit")
    return {"kind": "inline", "value": value}


def _response(
    request: Mapping[str, Any],
    *,
    payload: Any = None,
    error: AdapterError | None = None,
    binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "request_id": request.get("request_id"),
        "correlation_id": request.get("correlation_id"),
        "owner": OWNER,
        "ok": error is None,
        "source_bindings": [binding] if binding is not None else [],
        "receipt_ref": None,
    }
    if error is None:
        result["payload"] = _inline(payload)
    else:
        result["error"] = {
            "schema": SCHEMA,
            "code": error.code,
            "message": error.message,
            "details": _inline({}),
        }
    return result


def _validate_request(request: Mapping[str, Any]) -> dict[str, str] | None:
    if request.get("owner") != OWNER:
        raise AdapterError("AUTHORITY_MISMATCH", "request owner does not match this adapter")
    if request.get("operation") != OPERATION:
        raise AdapterError("AUTHORITY_MISMATCH", "request operation does not match this adapter")
    arguments = request.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise AdapterError("INVALID_ARGUMENT", "arguments must be an object")
    if arguments:
        raise AdapterError("INVALID_ARGUMENT", "polylogue.archive.status does not accept arguments")

    expected = request.get("expected_source_binding")
    if expected is None:
        return None
    if not isinstance(expected, Mapping) or set(expected) != {"source_ref", "generation", "root_digest"}:
        raise AdapterError("INVALID_ARGUMENT", "expected_source_binding has invalid fields")
    if not all(isinstance(expected[field], str) for field in expected):
        raise AdapterError("INVALID_ARGUMENT", "expected_source_binding fields must be strings")
    if expected["source_ref"] != SOURCE_REF:
        raise AdapterError("AUTHORITY_MISMATCH", "expected_source_binding names a different source")
    return dict(expected)


def _binding(identity: ArchiveIdentity) -> dict[str, str]:
    return {
        "source_ref": SOURCE_REF,
        "generation": identity.active_generation,
        "root_digest": f"sha256:{identity.authority_identity_digest}",
    }


def _status_payload(archive: ArchiveStore) -> dict[str, Any]:
    """Use the same archive-stat primitive and status projection as MCP."""
    stats = MCPArchiveStatsPayload.from_archive_stats(
        archive.stats(), include_embedded=False, include_db_size=False
    ).model_dump(mode="json", exclude_none=True)
    return {"operation": OPERATION, "archive": stats}


def _read_pinned_status(location: ArchiveLocation, identity: ArchiveIdentity) -> dict[str, Any]:
    index = identity.tier("index")
    if not index.exists or index.device is None or index.inode is None:
        raise AdapterError("OWNER_UNAVAILABLE", "active archive index is unavailable")
    try:
        index_fd = os.open(index.resolved_path, os.O_RDONLY)
    except OSError as error:
        raise AdapterError("OWNER_UNAVAILABLE", "active archive index is unavailable") from error
    try:
        opened = os.fstat(index_fd)
        if (opened.st_dev, opened.st_ino) != (index.device, index.inode):
            raise AdapterError("OWNER_UNAVAILABLE", "active archive generation changed while opening")
        transaction = QueryTransaction(
            location.configured_root,
            QueryTransactionRequest(
                operation="status", arguments={"scope": "archive"}, page_size=1, projection="status"
            ),
        )
        return InterruptibleSQLiteRead(transaction.context).run(
            location.configured_root,
            _status_payload,
            index_path=index.resolved_path,
            opened_main_fd=index_fd,
        )
    except AdapterError:
        raise
    except Exception as error:
        raise AdapterError("OWNER_UNAVAILABLE", "archive status is unavailable") from error
    finally:
        os.close(index_fd)


def _read_status(request: Mapping[str, Any]) -> dict[str, Any]:
    expected_binding = _validate_request(request)
    from polylogue.paths import archive_root as resolve_archive_root

    location = ArchiveLocation.resolve(resolve_archive_root())
    identity = ArchiveIdentity.resolve_location(location)
    binding = _binding(identity)
    if expected_binding is not None and expected_binding != binding:
        raise AdapterError("AUTHORITY_MISMATCH", "expected_source_binding does not match the active archive generation")
    payload = _read_pinned_status(location, identity)
    return _response(request, payload=payload, binding=binding)


def main() -> int:
    request: Mapping[str, Any] = {}
    try:
        decoded = json.loads(sys.stdin.read())
        if not isinstance(decoded, Mapping):
            raise AdapterError("INVALID_ARGUMENT", "request must be an object")
        request = decoded
        response = _read_status(request)
    except AdapterError as error:
        response = _response(request, error=error)
    except Exception:
        response = _response(request, error=AdapterError("OPERATION_FAILED", "archive status operation failed"))
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
