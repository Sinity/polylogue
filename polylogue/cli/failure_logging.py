from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..paths import STATE_HOME
from ..schema import stamp_payload


def _classify_exit_reason(exc: BaseException) -> str:
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)):
        return "io"
    if isinstance(exc, ValueError):
        return "schema"
    return "error"


def record_failure(args: object, exc: BaseException, *, phase: str = "cli") -> None:
    exit_reason = _classify_exit_reason(exc)
    os.environ["POLYLOGUE_EXIT_REASON"] = exit_reason
    try:
        STATE_HOME.mkdir(parents=True, exist_ok=True)
        provider = getattr(args, "provider", None) or getattr(args, "providers", None)
        file_hint = None
        for attr in ("input", "source", "dir"):
            value = getattr(args, attr, None)
            if value:
                file_hint = str(value)
                break
        hints = {
            "io": "Check file paths, permissions, and available disk space.",
            "schema": "Validate input/export schema and retry with updated tooling.",
            "error": "Re-run with --verbose for a traceback and file a bug if reproducible.",
        }
        record = stamp_payload(
            {
                "id": f"fail-{int(time.time() * 1000)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cmd": getattr(args, "cmd", None),
                "provider": provider,
                "file": file_hint,
                "phase": phase,
                "exception": exc.__class__.__name__,
                "message": str(exc),
                "exit_reason": exit_reason,
                "hint": hints.get(exit_reason),
                "cwd": str(Path.cwd()),
                "argv": sys.argv[1:],
            }
        )
        path = STATE_HOME / "failures.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
    except Exception:
        return


__all__ = ["record_failure"]

