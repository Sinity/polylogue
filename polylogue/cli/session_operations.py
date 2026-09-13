"""Machine adapter for the session owner's exact operation contracts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pydantic import BaseModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("contracts", "execute"))
    args = parser.parse_args()
    from polylogue.operations.session_contracts import (
        SESSION_OPERATION_ADAPTER,
        SessionOperationError,
        session_operation_contracts,
    )

    if args.action == "contracts":
        print(json.dumps(session_operation_contracts(), sort_keys=True, indent=2))
        return 0

    async def execute() -> BaseModel:
        from polylogue.api import Polylogue
        from polylogue.operations.session_reads import session_operation_response

        data = sys.stdin.buffer.read(65_537)
        if len(data) > 65_536:
            raise ValueError("session request exceeds 65536 bytes")
        request = SESSION_OPERATION_ADAPTER.validate_json(data)
        async with Polylogue() as api:
            return await session_operation_response(api, request)

    try:
        result = asyncio.run(execute())
    except (ValueError, OSError) as exc:
        result = SessionOperationError(code=getattr(exc, "code", "invalid_argument"), message=str(exc))
        print(result.model_dump_json())
        return 1
    print(result.model_dump_json())
    return 1 if isinstance(result, SessionOperationError) else 0


if __name__ == "__main__":
    raise SystemExit(main())
