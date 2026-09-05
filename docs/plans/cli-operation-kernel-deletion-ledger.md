# CLI operation kernel deletion ledger

This ledger covers the status, bare recent-session, and first query transport slice.

## Owners

| Concern | Canonical owner |
| --- | --- |
| Lowered operation request and typed outcome handling | `polylogue/cli/operation_kernel.py` |
| Archive-scoped UDS transport | `polylogue/daemon_client.py` |
| Query rendering and direct read execution | `polylogue/cli/archive_query.py` |
| Status and bare-invocation presentation | `polylogue/cli/commands/status.py`, `polylogue/cli/click_app.py` |

## LOC accounting

| disposition | lines |
| --- | ---: |
| gross deleted | 127 |
| relocated | 0 |
| added production, tests, and ledger | 94 |
| net maintained | -33 |

The deleted code was the forked HTTP session fetch, archive-root status probe, and duplicate HTTP status fetch. The additions validate operation envelopes and route bare recent-session and fast-status calls through the shared kernel. Existing direct status computation and query renderers remain the owners of local read semantics.

## Retired routes

- The query slice has no multiprocessing deadline worker or `/api/sessions` HTTP adapter.
- Archive identity is carried by the operation request and response instead of a separate status probe.
- A terminal daemon outcome is mapped once by the kernel and is never retried through a direct executor.
