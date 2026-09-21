"""The machine-consumable success envelope every surface writes.

Lives here rather than under ``cli/`` because it has more than one caller
ring: ``analysis/registry.py`` emits it for the JSON insight route, and a
function-local import is still a full closure edge (``_import_bases`` walks
the AST, not the runtime import order). Keeping ``emit_success`` in
``cli/shared/machine_errors.py`` therefore pulled that module -- and
``surfaces/outcome.py`` behind it -- into the derived schema identity, so an
ordinary CLI edit moved the identity and demanded an archive reconvergence.

The error half stays in ``cli/shared/machine_errors.py``: its error-code
vocabulary and argv pre-scanning tables describe the CLI's own surface and
have no caller outside it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from polylogue.core.json import JSONDocument, require_json_document


class MachineSuccessEnvelope(TypedDict):
    """Serialized machine-success envelope."""

    status: Literal["ok"]
    result: JSONDocument


@dataclass(frozen=True, slots=True)
class MachineSuccess:
    """Machine-readable success envelope."""

    result: Mapping[str, object] = field(default_factory=dict)
    status: Literal["ok"] = "ok"

    def to_dict(self) -> MachineSuccessEnvelope:
        return {
            "status": self.status,
            "result": require_json_document(dict(self.result), context="machine success result"),
        }

    def to_json(self, *, exclude_none: bool = False) -> str:
        del exclude_none
        return json.dumps(self.to_dict(), indent=2)


def _normalize_result_payload(
    result: Mapping[str, object] | MachineSuccess | None,
) -> JSONDocument:
    if result is None:
        return {}
    if isinstance(result, MachineSuccess):
        return require_json_document(result.result, context="machine success result")
    return require_json_document(
        {str(key): value for key, value in result.items()},
        context="machine success result",
    )


def success(result: Mapping[str, object] | MachineSuccess | None = None) -> MachineSuccess:
    return MachineSuccess(result=_normalize_result_payload(result))


def emit_success(result: Mapping[str, object] | MachineSuccess | None = None) -> None:
    """Write a ``{"status": "ok", "result": …}`` envelope to stdout."""
    sys.stdout.write(success(result).to_json(exclude_none=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


__all__ = [
    "MachineSuccess",
    "MachineSuccessEnvelope",
    "emit_success",
    "success",
]
