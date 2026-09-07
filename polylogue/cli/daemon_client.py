"""Compatibility import for the lightweight daemon UDS client."""

from polylogue.daemon_client import (
    DaemonClient,
    DaemonMutationIndeterminateError,
    DaemonOperationProtocolError,
    DaemonOperationRejected,
)

__all__ = [
    "DaemonClient",
    "DaemonMutationIndeterminateError",
    "DaemonOperationProtocolError",
    "DaemonOperationRejected",
]
