"""Bounded read-only AgentCTL owner adapter for the Polylogue archive."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from polylogue.api import Polylogue
from polylogue.paths import archive_root as resolve_archive_root
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

OWNER = "polylogue-archive"
OPERATION = "polylogue.archive.status"
SOURCE_REF = "sinnix://polylogue/archive"
SCHEMA = 1


def _inline(value: Any) -> dict[str, Any]:
    return {"kind": "inline", "value": value}


def _response(
    request: dict[str, Any],
    *,
    payload: Any = None,
    error: tuple[str, str] | None = None,
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
        code, message = error
        result["error"] = {
            "schema": SCHEMA,
            "code": code,
            "message": message,
            "details": _inline({}),
        }
    return result


def _binding(location: ArchiveLocation) -> dict[str, str]:
    identity = ArchiveIdentity.resolve_location(location)
    return {
        "source_ref": SOURCE_REF,
        "generation": identity.active_generation,
        "root_digest": f"sha256:{identity.authority_identity_digest}",
    }


def _read_status(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("operation") != OPERATION or request.get("owner") != OWNER:
        raise ValueError("unsupported owner operation")
    arguments = request.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    root = resolve_archive_root()
    location = ArchiveLocation.resolve(root)
    binding = _binding(location)

    async def read() -> dict[str, Any]:
        async with Polylogue(archive_root=root) as archive:
            stats = await archive.stats()
        return {
            "operation": OPERATION,
            "archive": {
                "session_count": stats.session_count,
                "message_count": stats.message_count,
                "word_count": stats.word_count,
                "origins": stats.origins,
                "tags": stats.tags,
                "last_sync": stats.last_sync,
            },
            "location": {
                "configured_root": str(location.configured_root),
                "active_index": location.active_index.as_dict(),
                "active_pointer": str(location.active_pointer) if location.active_pointer else None,
            },
        }

    payload = asyncio.run(read())
    # Keep the binding digest independently reproducible from the returned value.
    json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _response(request, payload=payload, binding=binding)


def main() -> int:
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        response = _read_status(request)
    except Exception as exc:
        request = request if "request" in locals() and isinstance(request, dict) else {}
        response = _response(request, error=("OPERATION_FAILED", str(exc)))
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
