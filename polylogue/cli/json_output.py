"""JSON output and error handling for --json mode.

Provides consistent JSON output format and error handling for CLI commands.
When --json mode is active, all errors are emitted as JSON instead of colored text.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..ui import UI


class JSONModeError(Exception):
    """Error that should be rendered as JSON in --json mode.

    Attributes:
        code: Machine-readable error code (e.g., "file_not_found")
        message: Human-readable error message
        details: Additional error context as key-value pairs
    """

    def __init__(self, code: str, message: str, **kwargs):
        """Create a JSON mode error.

        Args:
            code: Error code for programmatic handling
            message: Human-readable error description
            **kwargs: Additional error details to include in JSON output
        """
        self.code = code
        self.message = message
        self.details = kwargs
        super().__init__(message)


def json_error(code: str, message: str, **details) -> Dict[str, Any]:
    """Format error as JSON.

    Args:
        code: Error code
        message: Error message
        **details: Additional error details

    Returns:
        JSON-serializable error dictionary

    Example:
        >>> json_error("file_not_found", "File not found: export.zip", path="/tmp/export.zip")
        {"status": "error", "code": "file_not_found", "message": "File not found: export.zip", "path": "/tmp/export.zip"}
    """
    return {"status": "error", "code": code, "message": message, **details}


def json_success(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format success response as JSON.

    Args:
        data: Response data

    Returns:
        JSON-serializable success dictionary
    """
    return {"status": "success", **data}


def emit_json_or_error(payload: Optional[Dict], error: Optional[Exception], ui: UI, *, json_mode: bool = False) -> None:
    """Emit JSON output or error based on mode.

    In JSON mode, emits structured JSON. In interactive mode, uses console formatting.

    Args:
        payload: Success payload (if no error)
        error: Exception that occurred (if any)
        ui: UI instance
        json_mode: Whether JSON mode is active
    """
    if json_mode or ui.plain:
        if error:
            if isinstance(error, JSONModeError):
                output = json_error(error.code, error.message, **error.details)
            else:
                output = json_error("unknown", str(error))
        else:
            output = payload or {}

        print(json.dumps(output, indent=2, sort_keys=True))
    elif error:
        ui.console.print(f"[red]Error: {error}")


def safe_json_handler(handler):
    """Decorator to handle JSON mode errors.

    Wraps command handlers to catch JSONModeError exceptions and emit
    them as JSON when in --json mode.

    Args:
        handler: Command handler function

    Returns:
        Wrapped handler function

    Example:
        @safe_json_handler
        def run_import_cli(args, env):
            if not file.exists():
                raise JSONModeError("file_not_found", f"File not found: {file}")
    """

    def wrapper(args, env):
        json_mode = getattr(args, "json", False)
        try:
            return handler(args, env)
        except JSONModeError as e:
            if json_mode:
                emit_json_or_error(None, e, env.ui, json_mode=True)
                raise SystemExit(1)
            # In non-JSON mode, let it propagate as a regular exception
            raise
        except SystemExit:
            # Don't catch SystemExit - let it propagate
            raise
        except Exception as e:
            if json_mode:
                emit_json_or_error(None, e, env.ui, json_mode=True)
                raise SystemExit(1)
            # In non-JSON mode, let the exception propagate normally
            raise

    return wrapper


__all__ = [
    "JSONModeError",
    "json_error",
    "json_success",
    "emit_json_or_error",
    "safe_json_handler",
]
